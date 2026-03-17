"""Calibration: collect activation ranges from FP32 model."""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CalibrationResult:
    """Per-tensor activation scale from calibration."""
    # Maps tensor name → max absolute value
    max_abs: Dict[str, float] = field(default_factory=dict)
    # Maps tensor name → per-tensor scale (max_abs / 127)
    scales: Dict[str, float] = field(default_factory=dict)

    def add_observation(self, name: str, tensor_abs_max: float):
        """Record observed max absolute value, keeping the running maximum."""
        current = self.max_abs.get(name, 0.0)
        self.max_abs[name] = max(current, tensor_abs_max)

    def compute_scales(self):
        """Compute per-tensor INT8 scales from observed max values."""
        for name, max_val in self.max_abs.items():
            self.scales[name] = max(max_val, 1e-8) / 127.0

    def get_scale(self, name: str) -> float:
        return self.scales.get(name, 1.0 / 127.0)


def calibrate_model(model, sample_inputs: list) -> CalibrationResult:
    """Run calibration on a PyTorch model with sample inputs.

    Hooks every linear/layernorm layer to record activation ranges.
    """
    import torch

    result = CalibrationResult()
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                val = output.detach().abs().max().item()
                result.add_observation(name, val)
            # Also record input range
            if isinstance(input, tuple) and len(input) > 0 and isinstance(input[0], torch.Tensor):
                val = input[0].detach().abs().max().item()
                result.add_observation(f"{name}_input", val)
        return hook

    # Register hooks on all modules
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # Run forward passes
    model.eval()
    with torch.no_grad():
        for inp in sample_inputs:
            if hasattr(inp, 'items'):
                model(**dict(inp.items()))
            else:
                model(inp)

    # Remove hooks
    for h in hooks:
        h.remove()

    result.compute_scales()
    return result
