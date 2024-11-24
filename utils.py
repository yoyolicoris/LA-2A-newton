import torch
from torchcomp import compexp_gain, db2amp, amp2db
from torchlpc import sample_wise_lpc
from torch.autograd import Function
from functools import reduce


def chain_functions(*functions):
    return lambda x: reduce(lambda x, f: f(x), functions, x)


def arcsigmoid(x: torch.Tensor) -> torch.Tensor:
    return (x / (1 - x)).log()


def comp_gain(x, *args, **kwargs) -> torch.Tensor:
    return compexp_gain(x, *args, exp_ratio=0.9999, exp_thresh=-120, **kwargs)


def avg(rms: torch.Tensor, avg_coef: torch.Tensor):
    """Compute the running average of a signal.

    Args:
        rms (torch.Tensor): Input signal.
        avg_coef (torch.Tensor): Coefficient for the average RMS.

    Shape:
        - rms: :math:`(B, T)` where :math:`B` is the batch size and :math:`T` is the number of samples.
        - avg_coef: :math:`(B,)` or a scalar.

    """

    avg_coef = avg_coef.broadcast_to(rms.shape[0])
    assert torch.all(avg_coef > 0) and torch.all(avg_coef <= 1)

    return sample_wise_lpc(
        rms * avg_coef.unsqueeze(1),
        avg_coef[:, None, None].broadcast_to(rms.shape + (1,)) - 1,
    )


def avg_rms(audio: torch.Tensor, avg_coef: torch.Tensor) -> torch.Tensor:
    return avg(audio.square().clamp_min(1e-8), avg_coef).sqrt()


def gain_reduction(x, th, ratio, at, rt, make_up, delay: int = 0):
    # rms = avg_rms(x, avg_coef=avg_coef)
    gain = comp_gain(
        x.abs().clamp_min(1e-8),
        comp_ratio=ratio,
        comp_thresh=th,
        at=at,
        rt=rt,
    )
    if delay > 0:
        gain = torch.cat([gain[:, delay:], gain.new_ones(gain.shape[0], delay)], dim=1)
    return gain * db2amp(make_up.broadcast_to(x.shape[0], 1))


def compressor(x, *args, **kwargs):
    return gain_reduction(x, *args, **kwargs) * x


def compressor_inverse_filter(y, *args, **kwargs):
    return y / gain_reduction(*args, **kwargs)


def logits2comp_params(
    logits: torch.Tensor,
    ratio_func=lambda x: 1 + torch.exp(x),
    at_func=torch.sigmoid,
    rt_func=torch.sigmoid,
):
    th, ratio_logits, at_logits, rt_logits, make_up = torch.unbind(logits, dim=0)
    # avg_coef = torch.sigmoid(avg_coef_logits)
    # ratio = 1 + torch.exp(ratio_logits)
    ratio = ratio_func(ratio_logits)
    at = at_func(at_logits)
    rt = rt_func(rt_logits)
    # at = torch.sigmoid(at_logits)
    # rt = torch.sigmoid(rt_logits)
    return {
        # "avg_coef": avg_coef,
        "th": th,
        "ratio": ratio,
        "at": at,
        "rt": rt,
        "make_up": make_up,
    }


def simple_compressor(x, avg_coef, th, ratio, at, *args, **kwargs):
    return compressor(x, avg_coef, th, ratio, at, at, *args, **kwargs)


@torch.amp.autocast("cuda", False)
def freq_sampling(x, coef):
    # casting to double to avoid NaNs
    x_freq = torch.fft.rfft(x.double())
    coef = coef.double()
    freqs = torch.exp(
        -2j
        * torch.pi
        * torch.fft.rfftfreq(x.shape[1], dtype=torch.double, device=x_freq.device)
    )
    return torch.fft.irfft(
        x_freq * coef[:, None] / (1 - (1 - coef[:, None]) * freqs), n=x.shape[1]
    ).to(x.dtype)


def freq_simple_compressor(x, avg_coef, th, ratio, at, make_up, delay: int = 0):
    device, dtype = x.device, x.dtype
    factory_func = lambda y: torch.as_tensor(
        y, device=device, dtype=dtype
    ).broadcast_to(x.shape[0])
    avg_coef = factory_func(avg_coef)
    th = factory_func(th)
    ratio = factory_func(ratio)
    at = factory_func(at)
    make_up = factory_func(make_up)

    rms = freq_sampling(x.square().clamp_min(1e-8), avg_coef).sqrt()

    comp_slope = 1 - 1 / ratio

    log_x_rms = amp2db(rms)
    g = (comp_slope[:, None] * (log_x_rms - th[:, None])).relu().neg()

    f = db2amp(g)
    gain = freq_sampling(f - 1, at) + 1

    if delay > 0:
        gain = torch.cat([gain[:, delay:], gain.new_ones(gain.shape[0], delay)], dim=1)
    return x * gain * db2amp(make_up).unsqueeze(1)


def esr(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.flatten()
    target = target.flatten()
    diff = pred - target
    return (diff @ diff) / (target @ target)


class SPSACompressor(Function):
    @staticmethod
    def forward(ctx, x, avg_coef, th, ratio, at, rt, make_up, delay: int = 0):
        ctx.save_for_backward(x, avg_coef, th, ratio, at, rt, make_up)
        ctx.delay = delay
        return compressor(x, avg_coef, th, ratio, at, rt, make_up, delay)

    @staticmethod
    def backward(ctx, grad_output):
        x, avg_coef, th, ratio, at, rt, make_up = ctx.saved_tensors
        delay = ctx.delay
        requires_grad_mask = torch.tensor(
            [ctx.needs_input_grad[i] for i in range(1, 7)],
            device=x.device,
            dtype=torch.bool,
        )
        delta = torch.randint(0, 2, (6,), device=x.device) * 2 - 1
        eps = 0.00001

        avg_coef_plus = (
            torch.clamp(avg_coef + eps * delta[0], eps, 1 - eps)
            if requires_grad_mask[0]
            else avg_coef
        )
        avg_coef_minus = (
            torch.clamp(avg_coef - eps * delta[0], eps, 1 - eps)
            if requires_grad_mask[0]
            else avg_coef
        )

        th_plus = th + eps * delta[1] if requires_grad_mask[1] else th
        th_minus = th - eps * delta[1] if requires_grad_mask[1] else th

        ratio_plus = (
            torch.clamp_min(ratio + eps * delta[2], 1 + eps)
            if requires_grad_mask[2]
            else ratio
        )
        ratio_minus = (
            torch.clamp_min(ratio - eps * delta[2], 1 + eps)
            if requires_grad_mask[2]
            else ratio
        )

        at_plus = (
            torch.clamp(at + eps * delta[3], eps, 1 - eps)
            if requires_grad_mask[3]
            else at
        )
        at_minus = (
            torch.clamp(at - eps * delta[3], eps, 1 - eps)
            if requires_grad_mask[3]
            else at
        )

        rt_plus = (
            torch.clamp(rt + eps * delta[4], eps, 1 - eps)
            if requires_grad_mask[4]
            else rt
        )
        rt_minus = (
            torch.clamp(rt - eps * delta[4], eps, 1 - eps)
            if requires_grad_mask[4]
            else rt
        )

        make_up_plus = make_up + eps * delta[5] if requires_grad_mask[5] else make_up
        make_up_minus = make_up - eps * delta[5] if requires_grad_mask[5] else make_up

        y_plus = compressor(
            x, avg_coef_plus, th_plus, ratio_plus, at_plus, rt_plus, make_up_plus, delay
        )
        y_minus = compressor(
            x,
            avg_coef_minus,
            th_minus,
            ratio_minus,
            at_minus,
            rt_minus,
            make_up_minus,
            delay,
        )
        grad_num = (y_plus - y_minus).flatten() / (2 * eps)
        grad_output = grad_output.flatten()
        grad_params = grad_num @ grad_output

        if requires_grad_mask[0]:
            grad_avg_coef = grad_params / delta[0]
        else:
            grad_avg_coef = None

        if requires_grad_mask[1]:
            grad_th = grad_params / delta[1]
        else:
            grad_th = None

        if requires_grad_mask[2]:
            grad_ratio = grad_params / delta[2]
        else:
            grad_ratio = None

        if requires_grad_mask[3]:
            grad_at = grad_params / delta[3]
        else:
            grad_at = None

        if requires_grad_mask[4]:
            grad_rt = grad_params / delta[4]
        else:
            grad_rt = None

        if requires_grad_mask[5]:
            grad_make_up = grad_params / delta[5]
        else:
            grad_make_up = None

        return (
            None,
            grad_avg_coef,
            grad_th,
            grad_ratio,
            grad_at,
            grad_rt,
            grad_make_up,
            None,
        )
