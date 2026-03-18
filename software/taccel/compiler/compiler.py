"""Top-level compiler: PyTorch model → ProgramBinary."""
import struct
import numpy as np
from typing import Dict, Optional, List, Tuple
from ..assembler.assembler import ProgramBinary
from ..isa.encoding import encode
from ..isa.opcodes import Opcode, OPCODE_SHIFT, OPCODE_MASK, A_IMM28_SHIFT, MASK_28BIT
from ..quantizer.quantize import quantize_weights, quantize_tensor
from ..quantizer.scales import ScalePropagator
from ..quantizer.calibrate import CalibrationResult, calibrate_model
from .ir import IRGraph
from .graph_extract import extract_deit_tiny, EMBED_DIM, SEQ_LEN, NUM_PATCHES, PATCH_DIM, MLP_DIM, HEAD_DIM, NUM_HEADS, NUM_CLASSES
from .codegen import CodeGenerator
from .tiler import pad_dim


class Compiler:
    """Compile a DeiT-tiny model to ProgramBinary."""

    def __init__(self):
        self.scale_prop = ScalePropagator()

    def compile(self, state_dict: dict, calibration: Optional[CalibrationResult] = None,
                sample_inputs: Optional[list] = None) -> ProgramBinary:
        """Compile a DeiT-tiny model.

        Args:
            state_dict: PyTorch state dict with FP32 weights
            calibration: pre-computed calibration result, or None to use defaults
            sample_inputs: if calibration is None and sample_inputs provided, calibrate
        """
        # Step 1: Quantize weights
        quant_weights = quantize_weights(state_dict)

        # Step 2: Get calibration scales
        if calibration is not None:
            cal_scales = calibration.scales
        else:
            # Use default scales based on weight ranges
            cal_scales = self._default_calibration_scales(state_dict)

        # Step 3: Pre-scale biases
        prescaled_biases = self._prescale_biases(state_dict, quant_weights, cal_scales)

        # Step 4: Prepare weight data for codegen
        weight_data = {}
        for name, (data, scales) in quant_weights.items():
            if scales is not None:
                # Pad weight matrix to multiples of 16, then transpose to [K_in, N_out].
                # PyTorch stores weights as [N_out, K_in]; the systolic array reads src2 as
                # [K, N], so we must transpose before storing in DRAM.
                if data.ndim == 2:
                    data = np.pad(data,
                                  ((0, (16 - data.shape[0] % 16) % 16),
                                   (0, (16 - data.shape[1] % 16) % 16)),
                                  mode='constant')
                    data = np.ascontiguousarray(data.T)  # [N_pad, K_pad] → [K_pad, N_pad]
            weight_data[name] = (data, scales)

        # Add per-head Q/K/V weight slices using per-tensor quantization.
        # Using per-tensor (not per-channel) scale for each head ensures the single-scale
        # REQUANT in codegen is exact. Per-channel weights with mean(w_scales) REQUANT
        # introduces a per-channel weighting factor that distorts dot products, causing
        # wrong attention scores (e.g. 100% CLS-self instead of ~21%).
        for layer_idx in range(12):
            prefix = f"vit.encoder.layer.{layer_idx}"
            ln1_scale = cal_scales.get(f"block{layer_idx}_ln1", 6.0 / 127.0)
            for proj in ["query", "key", "value"]:
                wname = f"{prefix}.attention.attention.{proj}.weight"
                if wname not in state_dict:
                    continue
                w_fp32_full = state_dict[wname].numpy().astype(np.float32)  # [192, 192]
                bname = f"{prefix}.attention.attention.{proj}.bias"
                bias_fp32_full = state_dict[bname].numpy().astype(np.float32) \
                    if bname in state_dict else None
                for h in range(NUM_HEADS):
                    w_h_fp32 = w_fp32_full[h * HEAD_DIM:(h + 1) * HEAD_DIM, :]  # [64, 192]
                    # Per-tensor scale for this head's weight matrix
                    max_abs_h = float(np.max(np.abs(w_h_fp32)))
                    w_scale_h = max(max_abs_h, 1e-8) / 127.0
                    w_h_int8 = np.clip(np.round(w_h_fp32 / w_scale_h), -128, 127).astype(np.int8)
                    # Pad to multiples of 16
                    w_h_int8 = np.pad(w_h_int8,
                                      ((0, (16 - w_h_int8.shape[0] % 16) % 16),
                                       (0, (16 - w_h_int8.shape[1] % 16) % 16)),
                                      mode='constant')
                    # Uniform scale array (all output channels use same scale)
                    s_h = np.full(w_h_int8.shape[0], w_scale_h, dtype=np.float16)
                    # Transpose [N_pad, K_pad] → [K_pad, N_pad] for systolic layout
                    w_h_int8 = np.ascontiguousarray(w_h_int8.T)
                    weight_data[f"{wname}_h{h}"] = (w_h_int8, s_h)

                    # Per-head bias prescaled with matching per-tensor weight scale
                    # and calibrated LN1 activation scale (not hardcoded default).
                    if bias_fp32_full is not None:
                        b_h = bias_fp32_full[h * HEAD_DIM:(h + 1) * HEAD_DIM]
                        s_h_arr = np.full(len(b_h), w_scale_h, dtype=np.float32)
                        bias_i32 = self.scale_prop.prescale_bias(
                            b_h, np.array([ln1_scale]), s_h_arr)
                        pad_len = (16 - len(bias_i32) % 16) % 16
                        if pad_len:
                            bias_i32 = np.pad(bias_i32, (0, pad_len), constant_values=0)
                        prescaled_biases[f"{bname}_h{h}"] = bias_i32

        # Override out_proj, FC1, FC2 with per-tensor weight quantization.
        # quantize_weights() uses per-channel scales, but codegen's single-scale REQUANT
        # can only apply one scale to all output channels.  Using mean(per_ch_scales)
        # ≈ 0.5 × per_tensor_scale introduces a systematic ~2× amplitude error.
        # Per-tensor quantization makes REQUANT exact (mean(uniform) = the value).
        # Also fix bias prescaling: use calibrated layer-input scale (not hardcoded
        # 6/127) so the bias units match the accumulator units seen by REQUANT.
        for layer_idx in range(12):
            prefix = f"vit.encoder.layer.{layer_idx}"
            b = f"block{layer_idx}"
            layer_specs = [
                # (dense_name_prefix, input_scale_key)
                ("attention.output.dense", f"{b}_concat"),
                ("intermediate.dense",    f"{b}_ln2"),
                ("output.dense",          f"{b}_gelu"),
            ]
            for dense_name, input_scale_key in layer_specs:
                wname = f"{prefix}.{dense_name}.weight"
                bname = f"{prefix}.{dense_name}.bias"
                if wname not in state_dict:
                    continue
                w_fp32 = state_dict[wname].numpy().astype(np.float32)
                max_abs = float(np.max(np.abs(w_fp32)))
                w_scale = max(max_abs, 1e-8) / 127.0
                # Per-tensor INT8 quantization
                w_int8 = np.clip(np.round(w_fp32 / w_scale), -128, 127).astype(np.int8)
                # Pad to multiples of 16
                w_int8 = np.pad(w_int8,
                                ((0, (16 - w_int8.shape[0] % 16) % 16),
                                 (0, (16 - w_int8.shape[1] % 16) % 16)),
                                mode='constant')
                # Uniform scale for all output channels (before transpose)
                n_out_pad = w_int8.shape[0]
                s = np.full(n_out_pad, w_scale, dtype=np.float16)
                # Transpose to [K_in_pad, N_out_pad] for systolic layout
                w_int8 = np.ascontiguousarray(w_int8.T)
                weight_data[wname] = (w_int8, s)

                # Override bias with calibrated act_scale × per-tensor w_scale so
                # that bias units match the REQUANT formula in codegen.
                if bname in state_dict and hasattr(state_dict[bname], 'numpy'):
                    bias_fp32 = state_dict[bname].numpy().astype(np.float32)
                    act_scale_b = cal_scales.get(input_scale_key, 6.0 / 127.0)
                    bias_i32 = self.scale_prop.prescale_bias(
                        bias_fp32,
                        np.array([act_scale_b], dtype=np.float32),
                        np.full(len(bias_fp32), w_scale, dtype=np.float32),
                    )
                    pad_len = (16 - len(bias_i32) % 16) % 16
                    if pad_len:
                        bias_i32 = np.pad(bias_i32, (0, pad_len), constant_values=0)
                    prescaled_biases[bname] = bias_i32

        # Override classifier with per-tensor weight quantization.
        # Classifier logits are read directly from ACCUM (INT32) before REQUANT.
        # Per-channel weights give acc[j] = dot[j] / (act_scale * w_scale[j]) where
        # w_scale[j] varies per class, distorting relative logit rankings.
        # Per-tensor makes acc[j] = dot[j] / (act_scale * w_scale_t) — all classes
        # at the same scale, so ranking is correct.
        if "classifier.weight" in state_dict:
            w_fp32 = state_dict["classifier.weight"].numpy().astype(np.float32)
            max_abs = float(np.max(np.abs(w_fp32)))
            w_scale = max(max_abs, 1e-8) / 127.0
            w_int8 = np.clip(np.round(w_fp32 / w_scale), -128, 127).astype(np.int8)
            w_int8 = np.pad(w_int8,
                            ((0, (16 - w_int8.shape[0] % 16) % 16),
                             (0, (16 - w_int8.shape[1] % 16) % 16)),
                            mode='constant')
            n_out_pad = w_int8.shape[0]
            s = np.full(n_out_pad, w_scale, dtype=np.float16)
            w_int8 = np.ascontiguousarray(w_int8.T)
            weight_data["classifier.weight"] = (w_int8, s)
            # Fix classifier bias: use final_ln_scale (= cls_extract scale) as act_scale
            if "classifier.bias" in state_dict and hasattr(state_dict["classifier.bias"], 'numpy'):
                bias_fp32 = state_dict["classifier.bias"].numpy().astype(np.float32)
                act_scale_b = cal_scales.get("cls_extract", 6.0 / 127.0)
                bias_i32 = self.scale_prop.prescale_bias(
                    bias_fp32,
                    np.array([act_scale_b], dtype=np.float32),
                    np.full(len(bias_fp32), w_scale, dtype=np.float32),
                )
                pad_len = (16 - len(bias_i32) % 16) % 16
                if pad_len:
                    bias_i32 = np.pad(bias_i32, (0, pad_len), constant_values=0)
                prescaled_biases["classifier.bias"] = bias_i32

        # Quantize cls_token and pos_embed to INT8 using the embedding output scale.
        # These tensors are 3D in DeiT ([1,1,192] / [1,197,192]) so quantize_weights
        # silently skips them (ndim > 2 doesn't match any branch). We handle them
        # here by squeezing to 2D and quantizing at the pos_embed_add output scale so
        # codegen can load them as INT8 via DMA and use them in INT8 VADD.
        #
        # IMPORTANT: The scale must cover (cls/pos_embed + patches) after VADD, not
        # just cls/pos_embed alone. Using 6.0/127.0 causes heavy clipping because
        # pos_embed max_abs ≈ 6.9 and patch max_abs ≈ 6.0, so their sum ≈ 13.8.
        # cal_scales["pos_embed_add"] from actual calibration captures this full range.
        # Fallback: 14.0/127.0 safely covers the observed max of ~13.8.
        act_scale = cal_scales.get("pos_embed_add", 14.0 / 127.0)
        for k, v in state_dict.items():
            if not hasattr(v, 'numpy'):
                continue
            if 'cls_token' not in k and 'position_embeddings' not in k:
                continue
            t = v.numpy().astype(np.float32)
            while t.ndim > 2:
                t = t.squeeze(0)
            q = np.clip(np.round(t / act_scale), -128, 127).astype(np.int8)
            row_pad = (16 - q.shape[0] % 16) % 16
            col_pad = (16 - q.shape[1] % 16) % 16
            if row_pad or col_pad:
                q = np.pad(q, ((0, row_pad), (0, col_pad)), mode='constant')
            weight_data[k] = (q, None)

        # Step 5: Extract IR
        graph = extract_deit_tiny()

        # Step 6: Generate code
        codegen = CodeGenerator(weight_data, cal_scales, prescaled_biases)
        instructions, dram_data = codegen.generate(graph)

        # Step 7: Extend dram_data to include DRAM temp region (strip-mine spill space)
        dram_temp_size = codegen.mem.dram_temp_total
        if dram_temp_size > 0:
            dram_data = dram_data + bytes(dram_temp_size)

        # Step 8: Assemble into ProgramBinary
        insn_bytes = bytearray()
        for insn in instructions:
            insn_bytes.extend(encode(insn))

        # Step 9: Build unified DRAM layout
        # data_base is the byte offset of the data section within the DRAM image,
        # aligned to 16 bytes so instructions don't bleed into parameter data.
        data_base = (len(insn_bytes) + 15) & ~15

        # Patch SET_ADDR_LO instructions: all DRAM addresses emitted by codegen are
        # data-relative (offset from start of dram_blob).  Add data_base to make them
        # DRAM-absolute (offset from start of the unified DRAM image).
        # SET_ADDR_HI needs no patching because all data fits within 24 bits, so the
        # high 28-bit half is always zero and adding data_base (<256 KB) still fits in
        # the low 28 bits with no carry.
        patched = bytearray(insn_bytes)
        for i in range(0, len(patched), 8):
            word = struct.unpack(">Q", patched[i:i + 8])[0]
            opcode_val = (word >> OPCODE_SHIFT) & OPCODE_MASK
            if opcode_val == Opcode.SET_ADDR_LO:
                old_imm28 = (word >> A_IMM28_SHIFT) & MASK_28BIT
                new_imm28 = (old_imm28 + data_base) & MASK_28BIT
                word = (word & ~(MASK_28BIT << A_IMM28_SHIFT)) | (new_imm28 << A_IMM28_SHIFT)
                patched[i:i + 8] = struct.pack(">Q", word)

        # input_offset: DRAM-absolute address where the host writes input patches
        input_patches_dram_off = codegen.dram_layout.get("__input_patches__", 0)
        input_offset = data_base + input_patches_dram_off

        return ProgramBinary(
            instructions=bytes(patched),
            data=dram_data,
            entry_point=0,
            insn_count=len(instructions),
            data_base=data_base,
            input_offset=input_offset,
        )

    def _default_calibration_scales(self, state_dict: dict) -> Dict[str, float]:
        """Generate default calibration scales from weight statistics."""
        scales = {}
        # Default activation scale based on typical transformer activations
        default_scale = 6.0 / 127.0  # ~6 max abs is typical
        for name in state_dict:
            scales[name] = default_scale
            scales[f"{name}_input"] = default_scale
        return scales

    def _prescale_biases(self, state_dict: dict, quant_weights: dict,
                          cal_scales: dict) -> Dict[str, np.ndarray]:
        """Pre-scale biases to INT32."""
        prescaled = {}
        default_act_scale = 6.0 / 127.0

        for name, tensor in state_dict.items():
            if 'bias' not in name:
                continue
            if not hasattr(tensor, 'numpy'):
                continue

            bias_fp32 = tensor.numpy().astype(np.float32)

            # Find corresponding weight
            weight_name = name.replace('.bias', '.weight')
            if weight_name in quant_weights:
                _, w_scales = quant_weights[weight_name]
                if w_scales is not None:
                    act_scale = np.array([default_act_scale])
                    w_scales_fp32 = w_scales.astype(np.float32)
                    # Trim or pad w_scales to match bias length
                    if len(w_scales_fp32) > len(bias_fp32):
                        w_scales_fp32 = w_scales_fp32[:len(bias_fp32)]
                    elif len(w_scales_fp32) < len(bias_fp32):
                        w_scales_fp32 = np.pad(w_scales_fp32,
                                                (0, len(bias_fp32) - len(w_scales_fp32)),
                                                constant_values=w_scales_fp32[-1])
                    bias_int32 = self.scale_prop.prescale_bias(bias_fp32, act_scale, w_scales_fp32)
                    prescaled[name] = bias_int32

        # Add per-head bias slices for Q/K/V
        for layer_idx in range(12):
            prefix = f"vit.encoder.layer.{layer_idx}"
            for proj in ["query", "key", "value"]:
                bname = f"{prefix}.attention.attention.{proj}.bias"
                wname = f"{prefix}.attention.attention.{proj}.weight"
                if bname not in state_dict or wname not in quant_weights:
                    continue
                bias_fp32 = state_dict[bname].numpy().astype(np.float32)
                _, w_scales_full = quant_weights[wname]
                if w_scales_full is None:
                    continue
                for h in range(NUM_HEADS):
                    b_h = bias_fp32[h * HEAD_DIM:(h + 1) * HEAD_DIM]
                    s_h = w_scales_full[h * HEAD_DIM:(h + 1) * HEAD_DIM]
                    act_scale = np.array([6.0 / 127.0])
                    bias_i32 = self.scale_prop.prescale_bias(b_h, act_scale, s_h)
                    prescaled[f"{bname}_h{h}"] = bias_i32

        return prescaled
