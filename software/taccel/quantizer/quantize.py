"""Per-channel symmetric INT8 weight quantization."""
import numpy as np
from typing import Dict, Tuple


def quantize_tensor(tensor: np.ndarray, per_channel: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a 2D tensor to INT8 with per-channel symmetric quantization.

    Args:
        tensor: FP32 tensor of shape [out_channels, in_features]
        per_channel: if True, compute scale per output channel

    Returns:
        (int8_tensor, scales): quantized tensor and per-channel FP16 scales
    """
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)

    if per_channel:
        # Per-channel: scale[ch] = max(abs(W[ch,:])) / 127
        max_vals = np.max(np.abs(tensor), axis=1)
        max_vals = np.maximum(max_vals, 1e-8)  # avoid division by zero
        scales = max_vals / 127.0
    else:
        # Per-tensor
        max_val = max(np.max(np.abs(tensor)), 1e-8)
        scales = np.full(tensor.shape[0], max_val / 127.0)

    # Quantize
    scales_expanded = scales.reshape(-1, 1)
    q = np.clip(np.round(tensor / scales_expanded), -128, 127).astype(np.int8)

    return q, scales.astype(np.float16)


def dequantize_tensor(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize INT8 tensor back to FP32."""
    if q.ndim == 1:
        return q.astype(np.float32) * float(scales[0])
    scales_expanded = scales.astype(np.float32).reshape(-1, 1)
    return q.astype(np.float32) * scales_expanded


def quantize_weights(state_dict: dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Quantize all weight tensors in a state dict.

    Returns dict mapping weight names → (int8_tensor, scale_per_channel_fp16).
    Conv2d patch embedding is reshaped to 2D before quantizing.
    """
    result = {}
    for name, tensor in state_dict.items():
        if not hasattr(tensor, 'numpy'):
            continue
        t = tensor.numpy().astype(np.float32)

        if 'weight' in name and t.ndim >= 2:
            # Reshape conv2d [out, in, H, W] → [out, in*H*W]
            if t.ndim == 4:
                t = t.reshape(t.shape[0], -1)
            elif t.ndim > 2:
                t = t.reshape(t.shape[0], -1)
            q, scales = quantize_tensor(t)
            result[name] = (q, scales)
        elif 'bias' in name and t.ndim == 1:
            # 1D biases are LayerNorm beta — store as FP16 to match gamma convention.
            # Matmul biases are also 1D but they are handled by _prescale_biases in
            # the compiler (which reads from state_dict directly, not from here).
            # The SFU (sfu.py) reads gamma then beta both as FP16, so both must be FP16.
            weight_name = name.replace('.bias', '.weight')
            if weight_name in state_dict:
                w = state_dict[weight_name]
                if hasattr(w, 'numpy') and w.numpy().ndim >= 2:
                    # Matmul/conv bias — keep as FP32 for pre-scaling
                    result[name] = (t, None)
                else:
                    # LayerNorm beta (weight is 1D) — store as FP16
                    result[name] = (t.astype(np.float16), None)
            else:
                result[name] = (t, None)
        elif t.ndim <= 2:
            # LayerNorm gamma, cls_token, pos_embed, etc. — store as FP16 for SFU
            result[name] = (t.astype(np.float16), None)

    return result
