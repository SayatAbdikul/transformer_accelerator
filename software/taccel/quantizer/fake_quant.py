"""Fake quantization utilities: simulate INT8 inference in FP32.

This module applies our exact per-channel INT8 quantization scheme
(quantize weights → dequantize back to FP32) to a PyTorch model, so
we can measure quantization error without running the full accelerator
simulator.

Two modes:
  - weight_only: only weights are quantized; activations stay FP32
  - full_int8: weights + activations are quantized (simulated with hooks)
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Callable

from .quantize import quantize_tensor, dequantize_tensor


def _quantize_dequantize_weight(weight: torch.Tensor) -> torch.Tensor:
    """Apply per-channel INT8 quantization then dequantize back to FP32.

    This exactly models the precision loss that would occur on hardware:
    the weight is rounded to the nearest INT8 value and then scaled back,
    leaving only the rounding error.
    """
    w_np = weight.detach().cpu().numpy().astype(np.float32)

    # Reshape to 2D if needed (conv2d: [out, in, H, W] → [out, in*H*W])
    orig_shape = w_np.shape
    if w_np.ndim > 2:
        w_np = w_np.reshape(orig_shape[0], -1)

    q, scales = quantize_tensor(w_np, per_channel=True)
    w_rec = dequantize_tensor(q, scales).astype(np.float32)
    w_rec = w_rec.reshape(orig_shape)

    return torch.from_numpy(w_rec).to(weight.device).to(weight.dtype)


def apply_weight_quantization(model: nn.Module) -> nn.Module:
    """Return a copy of model with all Linear/Conv2d weights fake-quantized.

    The returned model has weights that have been:
      1. Per-channel INT8 quantized (using our exact scheme)
      2. Dequantized back to FP32

    This captures all weight rounding error with zero code change to the
    model forward pass itself.
    """
    model_q = copy.deepcopy(model)

    quantized_count = 0
    total_params_before = 0
    total_params_after = 0

    for name, module in model_q.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            w_orig = module.weight.data
            w_q = _quantize_dequantize_weight(w_orig)

            total_params_before += w_orig.numel()
            total_params_after += w_q.numel()

            module.weight.data = w_q
            quantized_count += 1

    return model_q, quantized_count


class ActivationQuantizer:
    """Hook-based activation quantizer using calibrated per-tensor scales.

    Intercepts module outputs, quantizes to INT8, and dequantizes back.
    """

    def __init__(self, scales: Dict[str, float]):
        self.scales = scales
        self.hooks = []

    def _make_hook(self, name: str):
        scale = self.scales.get(name, None)

        def hook(module, input, output):
            if scale is None or scale <= 0:
                return output
            if not isinstance(output, torch.Tensor):
                return output
            # Quantize to INT8: clip(round(x / scale), -128, 127) * scale
            q = torch.clamp(torch.round(output / scale), -128, 127)
            return q * scale

        return hook

    def attach(self, model: nn.Module):
        """Attach quantization hooks to all Linear/Conv2d modules."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
                h = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def calibrate_activation_scales(
    model: nn.Module,
    sample_inputs: list,
    percentile: float = 99.99,
) -> Dict[str, float]:
    """Collect per-module activation ranges using sample inputs.

    Args:
        model: FP32 model
        sample_inputs: list of dict inputs (as passed to model(**inp))
        percentile: which percentile of abs(activation) to use as max

    Returns:
        dict mapping module name → INT8 scale (max_abs / 127)
    """
    records: Dict[str, list] = {}
    hooks = []

    def make_record_hook(name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                abs_vals = output.detach().abs().float().flatten()
                # Use percentile to ignore outliers
                p = float(abs_vals.numpy().max()) if percentile >= 100.0 \
                    else float(np.percentile(abs_vals.numpy(), percentile))
                records.setdefault(name, []).append(p)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            h = module.register_forward_hook(make_record_hook(name))
            hooks.append((name, h))

    model.eval()
    with torch.no_grad():
        for inp in sample_inputs:
            try:
                model(**inp)
            except TypeError:
                model(inp)

    for _, h in hooks:
        h.remove()

    # Convert to INT8 scales: max_abs / 127
    scales = {}
    for name, vals in records.items():
        max_abs = max(vals) if vals else 1.0
        scales[name] = max(max_abs, 1e-8) / 127.0

    return scales


def compute_metrics(
    logits_fp32: np.ndarray,
    logits_q: np.ndarray,
) -> Dict[str, float]:
    """Compute accuracy and similarity metrics between FP32 and quantized logits.

    Returns dict with: top1_match, top5_match, cosine_sim, logit_mse,
                       logit_mae, softmax_kl_div, logit_snr_db
    """
    from scipy.spatial.distance import cosine
    from scipy.special import softmax
    from scipy.stats import entropy

    results = {}

    # Top-1 and Top-5 match
    top1_fp32 = int(np.argmax(logits_fp32))
    top1_q = int(np.argmax(logits_q))
    results["top1_match"] = (top1_fp32 == top1_q)
    results["top1_fp32"] = top1_fp32
    results["top1_q"] = top1_q

    top5_fp32 = set(np.argsort(logits_fp32)[-5:])
    top5_q = set(np.argsort(logits_q)[-5:])
    results["top5_match"] = len(top5_fp32 & top5_q) >= 1

    # Cosine similarity of logit vectors
    cos_dist = cosine(logits_fp32.astype(np.float64), logits_q.astype(np.float64))
    results["cosine_sim"] = float(1.0 - cos_dist)

    # MSE and MAE of logits
    diff = logits_fp32.astype(np.float64) - logits_q.astype(np.float64)
    results["logit_mse"] = float(np.mean(diff ** 2))
    results["logit_mae"] = float(np.mean(np.abs(diff)))

    # Signal-to-noise ratio of logits (dB)
    signal_power = float(np.mean(logits_fp32.astype(np.float64) ** 2))
    noise_power = float(np.mean(diff ** 2))
    if noise_power > 0 and signal_power > 0:
        results["logit_snr_db"] = float(10.0 * np.log10(signal_power / noise_power))
    else:
        results["logit_snr_db"] = float("inf")

    # KL divergence of softmax distributions
    # Use temperature=1 softmax; shift by max for numerical stability
    def safe_softmax(x):
        x = x.astype(np.float64)
        x = x - x.max()
        e = np.exp(np.clip(x, -500, 0))
        return e / e.sum()

    p_fp32 = safe_softmax(logits_fp32)
    p_q = safe_softmax(logits_q)
    # KL(FP32 || Q): how much info is lost going from FP32 to quantized
    eps = 1e-12
    kl = float(np.sum(p_fp32 * np.log((p_fp32 + eps) / (p_q + eps))))
    results["softmax_kl_div"] = kl

    return results
