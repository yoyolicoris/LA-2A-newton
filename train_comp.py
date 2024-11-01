import logging

nb_logger = logging.getLogger("numba")
nb_logger.setLevel(logging.ERROR)  # only show error
import torch
from torch.nn import ParameterDict, Parameter
from torch.utils.data import DataLoader, TensorDataset
import wandb
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
from functools import partial, reduce
from itertools import chain, starmap, accumulate
from typing import Any, Dict, List, Tuple
import yaml
from torchaudio import load
from torchaudio.functional import lfilter
from torchcomp import ms2coef, coef2ms, db2amp
from torchlpc import sample_wise_lpc
import pyloudnorm as pyln

from utils import (
    arcsigmoid,
    compressor,
    simple_compressor,
    freq_simple_compressor,
    esr,
    SPSACompressor,
    chain_functions,
    logits2comp_params,
)


def simple_filter(x: torch.Tensor, a1: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    return sample_wise_lpc(
        x + b1 * torch.cat([x.new_zeros(x.shape[0], 1), x[:, :-1]], dim=1),
        a1.broadcast_to(x.shape + (1,)),
    )


@hydra.main(config_path="cfg", config_name="config")
def train(cfg: DictConfig):
    # TODO: Add a proper logger

    tr_cfg = cfg.data.train
    duration = cfg.data.duration
    overlap = cfg.data.overlap
    batch_size = cfg.data.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_input, sr = load(tr_cfg.input)
    train_target, sr2 = load(tr_cfg.target)
    assert sr == sr2, "Sample rates must match"
    if tr_cfg.start is not None and tr_cfg.end:
        train_input = train_input[:, int(sr * tr_cfg.start) : int(sr * tr_cfg.end)]
        train_target = train_target[:, int(sr * tr_cfg.start) : int(sr * tr_cfg.end)]

    assert train_input.shape == train_target.shape, "Input and target shapes must match"

    frame_size = int(sr * duration)
    hop_size = frame_size - int(sr * overlap)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(train_input.numpy().T)
    print(f"Train input loudness: {loudness}")
    target_loudness = meter.integrated_loudness(train_target.numpy().T)
    print(f"Train target loudness: {target_loudness}")

    m2c = partial(ms2coef, sr=sr)
    c2m = partial(coef2ms, sr=sr)

    # config: Any = OmegaConf.to_container(cfg)
    # wandb_init = config.pop("wandb_init", {})
    # run: Any = wandb.init(config=config, **wandb_init)

    # initialize model
    inits = cfg.compressor.inits
    init_th = torch.tensor(inits.threshold, dtype=torch.float32)
    init_ratio = torch.tensor(inits.ratio, dtype=torch.float32)
    init_at = m2c(torch.tensor(inits.attack_ms, dtype=torch.float32))
    init_rt = m2c(torch.tensor(inits.release_ms, dtype=torch.float32))
    init_rms_avg = torch.tensor(inits.rms_avg, dtype=torch.float32)
    init_make_up_gain = torch.tensor(inits.make_up_gain, dtype=torch.float32)

    th = init_th
    ratio_logit = torch.log(init_ratio - 1)
    at_logit = arcsigmoid(init_at)
    rt_logit = arcsigmoid(init_rt)
    rms_avg_logit = arcsigmoid(init_rms_avg)
    make_up_gain = init_make_up_gain

    params = torch.stack(
        [rms_avg_logit, th, ratio_logit, at_logit, rt_logit, make_up_gain]
    ).to(device)

    train_input = train_input.to(device)
    train_target = train_target.to(device)

    if cfg.compressor.init_config:
        init_cfg = yaml.safe_load(open(cfg.compressor.init_config))
        init_cfg.pop("formated_params", None)
        init_params = {k: Parameter(torch.tensor(v)) for k, v in init_cfg.items()}
        params.load_state_dict(init_params, strict=False)

    comp_delay = cfg.compressor.delay
    # infer = lambda x: compressor(x, *logits2comp_params(params), delay=comp_delay)

    # initialize loss function
    # loss_fn = hydra.utils.instantiate(cfg.loss_fn).to(device)
    loss_fn = partial(torch.nn.functional.mse_loss, reduction="sum")

    prefilt = partial(
        simple_filter,
        a1=torch.tensor(-0.995, device=device),
        b1=torch.tensor(-1, device=device),
    )

    unfold_input = train_input.unfold(1, frame_size, hop_size).reshape(-1, frame_size)
    unfold_target = train_target.unfold(1, frame_size, hop_size).reshape(-1, frame_size)
    unfold_target = prefilt(unfold_target)

    def get_param2loss(x, y):
        return chain_functions(
            logits2comp_params,
            lambda d: compressor(x, **d, delay=comp_delay),
            prefilt,
            lambda x: loss_fn(
                x[:, frame_size - hop_size :], y[:, frame_size - hop_size :]
            ),
        )

    param2loss = get_param2loss(unfold_input, unfold_target)
    # grad = torch.func.grad(param2loss)
    # hess = torch.func.hessian(param2loss)

    # loader = DataLoader(
    #     TensorDataset(unfold_input, unfold_target), batch_size=batch_size, shuffle=True
    # )

    # h_inv = torch.eye(params.shape[0], device=device)
    # h_avg = torch.eye(params.shape[0], device=device)
    # g_avg = params.new_zeros(params.shape)

    reg_lambda = cfg.optimiser.reg_lambda
    alpha = cfg.optimiser.alpha
    beta = cfg.optimiser.beta
    max_iter = cfg.optimiser.max_iter

    prev_loss = param2loss(params)

    with tqdm(range(cfg.epochs)) as pbar:
        for global_step in pbar:
            g = sum(
                map(
                    lambda f: f(params),
                    map(
                        torch.func.grad,
                        map(
                            get_param2loss,
                            unfold_input.split(batch_size),
                            unfold_target.split(batch_size),
                        ),
                    ),
                )
            )
            hess = sum(
                map(
                    lambda f: f(params),
                    map(
                        torch.func.hessian,
                        map(
                            get_param2loss,
                            unfold_input.split(batch_size),
                            unfold_target.split(batch_size),
                        ),
                    ),
                ),
            )

            h_inv = torch.linalg.inv(hess.diagonal_scatter(hess.diag() + reg_lambda))
            # h_inv = torch.diag(1 / (hess.diag() + reg_lambda))
            step = -h_inv @ g
            lambda_norm = -g @ step

            # perform backtracking line search
            t = 1
            i = 0
            upper_bound = prev_loss + alpha * t * lambda_norm
            for i in range(max_iter):
                params_new = params + t * step
                loss = param2loss(params_new)
                if loss < upper_bound:
                    break
                t *= beta
            if i == max_iter - 1:
                print("Line search failed")
                print(f"Loss: {loss}, Upper bound: {upper_bound}, norm: {lambda_norm}")
                break

            params = params_new
            prev_loss = loss

            pbar_dict = {
                "loss": loss,
                "norm": lambda_norm,
                "t": t,
                "inner_iter": i,
            } | logits2comp_params(params)

            pbar_dict["at"] = coef2ms(pbar_dict["at"], sr=sr)
            pbar_dict["rt"] = coef2ms(pbar_dict["rt"], sr=sr)
            pbar_dict = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in pbar_dict.items()
            }

            pbar.set_postfix(**pbar_dict)

            # wandb.log(pbar_dict, step=global_step)

        #     return lowest_loss

        # try:
        #     losses = list(accumulate(pbar, step, initial=torch.inf))
        # except KeyboardInterrupt:
        #     print("Training interrupted")

    print("Training complete. Saving model...")

    return


if __name__ == "__main__":
    train()
