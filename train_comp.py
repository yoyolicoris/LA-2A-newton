import logging

nb_logger = logging.getLogger("numba")
nb_logger.setLevel(logging.ERROR)  # only show error
import torch
import wandb
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
from functools import partial, reduce
from typing import Any, Dict, List, Tuple
import yaml
from torchaudio import load
from torchcomp import ms2coef, coef2ms, db2amp, amp2db
from torchlpc import sample_wise_lpc
import pyloudnorm as pyln

from utils import (
    arcsigmoid,
    compressor,
    compressor_inverse_filter,
    esr,
    chain_functions,
    logits2comp_params,
)


def direct_hess_step(hess_func, g):
    hess = hess_func()
    try:
        step = -torch.linalg.solve(hess, g)
    except RuntimeError:
        print("Singular matrix detected, using lstsq")
        step = -torch.linalg.lstsq(hess, g).solution
    return step


def conjugate_gradient(p, func, g):
    r_t = -g
    p_t = g
    v_t = torch.zeros_like(g)
    while torch.norm(r_t) > torch.finfo(r_t.dtype).eps:
        _, Ap_t = torch.autograd.functional.vhp(func, p, p_t)
        alpha_t = (r_t @ r_t) / (p_t @ Ap_t)
        v_t += alpha_t * p_t
        r_new = r_t + alpha_t * Ap_t
        beta_t = (r_new @ r_new) / (r_t @ r_t)
        p_t = -r_new + beta_t * p_t
        r_t = r_new
    return -v_t


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
    overlap = int(sr * overlap)
    hop_size = frame_size - overlap

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(train_input.numpy().T)
    print(f"Train input loudness: {loudness}")
    target_loudness = meter.integrated_loudness(train_target.numpy().T)
    print(f"Train target loudness: {target_loudness}")

    m2c = partial(ms2coef, sr=sr)
    c2m = partial(coef2ms, sr=sr)

    config: Any = OmegaConf.to_container(cfg)
    # wandb_init = config.pop("wandb_init", {})
    # run: Any = wandb.init(config=config, **wandb_init)

    # initialize model

    if cfg.compressor.range.ratio:
        ratio_min, ratio_max = (
            cfg.compressor.range.ratio.min,
            cfg.compressor.range.ratio.max,
        )
        ratio_func = lambda x: ratio_min + (ratio_max - ratio_min) * torch.sigmoid(x)
        ratio_func_inv = lambda x: torch.log((x - ratio_min) / (ratio_max - x))
    else:
        ratio_func = lambda x: 1 + torch.exp(x)
        ratio_func_inv = lambda x: torch.log(x - 1)

    if cfg.compressor.range.attack_ms:
        at_min, at_max = (
            m2c(torch.tensor(cfg.compressor.range.attack_ms.min, dtype=torch.float32)),
            m2c(torch.tensor(cfg.compressor.range.attack_ms.max, dtype=torch.float32)),
        )
        at_func = lambda x: at_min + (at_max - at_min) * torch.sigmoid(x)
        at_func_inv = lambda x: torch.log((x - at_min) / (at_max - x))
    else:
        at_func = torch.sigmoid
        at_func_inv = arcsigmoid
    if cfg.compressor.range.release_ms:
        rt_min, rt_max = (
            m2c(torch.tensor(cfg.compressor.range.release_ms.min, dtype=torch.float32)),
            m2c(torch.tensor(cfg.compressor.range.release_ms.max, dtype=torch.float32)),
        )
        rt_func = lambda x: rt_min + (rt_max - rt_min) * torch.sigmoid(x)
        rt_func_inv = lambda x: torch.log((x - rt_min) / (rt_max - x))
    else:
        rt_func = torch.sigmoid
        rt_func_inv = arcsigmoid

    my_logits2comp_params = partial(
        logits2comp_params,
        ratio_func=ratio_func,
        at_func=at_func,
        rt_func=rt_func,
    )

    if cfg.compressor.init_ckpt:
        params = torch.load(cfg.compressor.init_ckpt, map_location=device)
    else:
        inits = cfg.compressor.inits
        init_th = torch.tensor(inits.threshold, dtype=torch.float32)
        init_ratio = torch.tensor(inits.ratio, dtype=torch.float32)
        init_at = m2c(torch.tensor(inits.attack_ms, dtype=torch.float32))
        init_rt = m2c(torch.tensor(inits.release_ms, dtype=torch.float32))
        init_make_up_gain = torch.tensor(inits.make_up_gain, dtype=torch.float32)

        th = init_th
        make_up_gain = init_make_up_gain
        ratio_logit = ratio_func_inv(init_ratio)
        at_logit = at_func_inv(init_at)
        rt_logit = rt_func_inv(init_rt)

        params = torch.stack([th, ratio_logit, at_logit, rt_logit, make_up_gain]).to(
            device
        )

    train_input = train_input.to(device)
    train_target = train_target.to(device)

    comp_delay = cfg.compressor.delay
    loss_fn = partial(torch.nn.functional.mse_loss, reduction="sum")

    prefilt = partial(
        simple_filter,
        a1=torch.tensor(-0.995, device=device),
        b1=torch.tensor(-1, device=device),
    )

    mode = cfg.mode
    method = cfg.optimiser.method

    unfold_input = train_input.unfold(1, frame_size, hop_size).reshape(-1, frame_size)
    unfold_target = train_target.unfold(1, frame_size, hop_size).reshape(-1, frame_size)

    def predict(x, params):
        return prefilt(compressor(x, **my_logits2comp_params(params), delay=comp_delay))

    match mode:
        case "forward":
            unfold_target = prefilt(unfold_target)

            def get_param2loss(x, y):
                return chain_functions(
                    # logits2comp_params,
                    # lambda d: compressor(x, **d, delay=comp_delay),
                    # prefilt,
                    partial(predict, x),
                    lambda pred: loss_fn(pred[:, overlap:], y[:, overlap:]),
                )

        case "inverse":
            unfold_input = prefilt(unfold_input)

            def get_param2loss(x, y):
                return chain_functions(
                    logits2comp_params,
                    lambda d: compressor_inverse_filter(y, x, **d, delay=comp_delay),
                    prefilt,
                    lambda inv_y: loss_fn(x[:, overlap:], inv_y[:, overlap:]),
                )

        case _:
            raise ValueError(f"Invalid mode: {mode}")

    param2loss = get_param2loss(unfold_input, unfold_target)

    # scale output gain to match target loudness
    init_preds = predict(unfold_input, params)
    scaler = amp2db(
        torch.sqrt(
            unfold_target[:, overlap:].square().mean()
            / init_preds[:, overlap:].square().mean()
        )
    )
    params[-1] += scaler
    print(
        f"Increasing make-up gain by {scaler.item()} dB"
        if scaler > 0
        else f"Decreasing make-up gain by {scaler.item()} dB"
    )

    alpha = cfg.optimiser.alpha
    beta = cfg.optimiser.beta
    max_iter = cfg.optimiser.max_iter

    prev_loss = param2loss(params)
    optimal_params = params.clone()
    lowest_loss = prev_loss
    t = 1
    i = 0

    with tqdm(range(cfg.epochs)) as pbar:
        for global_step in pbar:
            # grad_func = lambda p: sum(
            #     map(
            #         lambda f: f(p),
            #         map(
            #             torch.func.grad,
            #             map(
            #                 get_param2loss,
            #                 unfold_input.split(batch_size * params.numel()),
            #                 unfold_target.split(batch_size * params.numel()),
            #             ),
            #         ),
            #     )
            # )
            # g = grad_func(params)
            g = torch.autograd.functional.jacobian(param2loss, params)

            if batch_size > 1:
                hess_func = lambda: sum(
                    map(
                        partial(torch.autograd.functional.hessian, inputs=params),
                        map(
                            get_param2loss,
                            unfold_input.split(batch_size),
                            unfold_target.split(batch_size),
                        ),
                    ),
                )
            else:
                hess_func = lambda: torch.autograd.functional.hessian(
                    param2loss, params
                )

            match method:
                case "direct":
                    step = direct_hess_step(hess_func, g)
                case "cg":
                    step = conjugate_gradient(params, param2loss, g)
                case _:
                    raise ValueError(f"Invalid optimiser method: {method}")

            lambda_norm = -g @ step
            if lambda_norm < 0:
                print(f"Negative curvature detected, {lambda_norm}")
                # print(g, h_inv, step)
                # step = -g / g.norm()
                random_step = torch.randn_like(step) / step.numel() ** 0.5
                ortho_step = random_step - random_step @ step / step.norm()
                step = ortho_step
                # step /= step.norm()
                # lambda_norm = -g @ step
                # break
                params_new = params + step
                loss = param2loss(params_new)
            else:

                # perform backtracking line search
                t = 1
                i = 0
                upper_bound = prev_loss + alpha * t * lambda_norm.relu()
                for i in range(max_iter):
                    params_new = params + t * step
                    loss = param2loss(params_new)
                    if loss < upper_bound:
                        break
                    t *= beta

                if i == max_iter - 1:
                    print("Line search failed")
                    print(
                        f"Loss: {loss}, Upper bound: {upper_bound}, norm: {lambda_norm}"
                    )
                    break

            params = params_new
            if loss < lowest_loss:
                optimal_params.copy_(params)
                lowest_loss = loss
            prev_loss = loss

            esr_score = esr(
                prefilt(
                    compressor(
                        unfold_input, **my_logits2comp_params(params), delay=comp_delay
                    )
                )[:, overlap:],
                unfold_target[:, overlap:],
            )

            pbar_dict = {
                "loss": loss,
                "norm": lambda_norm,
                "t": t,
                "inner_iter": i,
                "esr": esr_score * 100,
            } | my_logits2comp_params(params)

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

    print(f"Lowest loss: {lowest_loss}")
    print("Training complete. Saving model...")

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=False, exist_ok=True)

    torch.save(optimal_params, ckpt_dir / "logits.pt")
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    return


if __name__ == "__main__":
    train()
