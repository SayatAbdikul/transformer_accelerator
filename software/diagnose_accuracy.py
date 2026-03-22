#!/usr/bin/env python3
"""Phase A diagnostics: identify where FP32 vs INT8 accuracy is lost.

Collects:
  A1. Per-block residual output ranges vs. constant block_input_scale (H1)
  A3. Per-head QKT ranges vs. hardcoded 6.0/127.0 (H5)
  VADD saturation % per residual add in the golden model (H4)

Usage:
    python3 diagnose_accuracy.py [--max-images 5]
"""
import argparse
import collections
import math
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_golden import (
    COCO_VAL_IDS,
    MODEL_NAME,
    collect_images,
    compile_model,
    load_model,
    patch_embed_int8,
)
from taccel.compiler.tiler import pad_dim
from taccel.golden_model import MachineState, Simulator
from taccel.golden_model import memory as mem_module
from taccel.isa.opcodes import BUF_ABUF
from transformers import AutoImageProcessor

DEPTH = 12
NUM_HEADS = 3


# ---------------------------------------------------------------------------
# A1 + A3: FP32 hooks
# ---------------------------------------------------------------------------

def collect_fp32_stats(model, processor, images):
    """Register hooks to measure per-block residual ranges and per-head QKT ranges."""
    residual1_maxabs = collections.defaultdict(list)  # block_idx -> [max_abs per image]
    residual2_maxabs = collections.defaultdict(list)  # block_idx -> [max_abs per image]
    qkt_maxabs = collections.defaultdict(list)        # (block, head) -> [max_abs per image]

    handles = []

    # --- A1: residual1 output range ---
    # The residual1 output is the input to layernorm_after (pre-MLP residual).
    for i in range(DEPTH):
        def _make_res1(block_idx):
            def hook(module, inputs):
                x = inputs[0]
                residual1_maxabs[block_idx].append(float(x.abs().max().item()))
            return hook
        h = model.vit.encoder.layer[i].layernorm_after.register_forward_pre_hook(_make_res1(i))
        handles.append(h)

    # --- A1: residual2 output range ---
    # The ViTLayer output is the residual2 output (after the MLP residual add).
    for i in range(DEPTH):
        def _make_res2(block_idx):
            def hook(module, inputs, output):
                x = output[0] if isinstance(output, tuple) else output
                residual2_maxabs[block_idx].append(float(x.abs().max().item()))
            return hook
        h = model.vit.encoder.layer[i].register_forward_hook(_make_res2(i))
        handles.append(h)

    # --- A3: QKT range per head ---
    # Hook ViTSelfAttention, recompute Q@K^T/sqrt(d) from the module's projections.
    # This runs Q/K projections twice per image (diagnostics only).
    for i in range(DEPTH):
        def _make_qkt(block_idx):
            def hook(module, inputs, output):
                hidden = inputs[0]  # post-LayerNorm activations fed into attention
                with torch.no_grad():
                    q = module.transpose_for_scores(module.query(hidden))
                    k = module.transpose_for_scores(module.key(hidden))
                    scale = math.sqrt(module.attention_head_size)
                    qkt = torch.matmul(q, k.transpose(-1, -2)) / scale  # [B, H, S, S]
                    for h in range(qkt.shape[1]):
                        max_abs = float(qkt[0, h].abs().max().item())
                        qkt_maxabs[(block_idx, h)].append(max_abs)
            return hook
        h = model.vit.encoder.layer[i].attention.attention.register_forward_hook(_make_qkt(i))
        handles.append(h)

    model.eval()
    with torch.no_grad():
        for _, img in images:
            inp = processor(images=img, return_tensors="pt")
            model(**inp)

    for h in handles:
        h.remove()

    return residual1_maxabs, residual2_maxabs, qkt_maxabs


# ---------------------------------------------------------------------------
# VADD saturation: golden model instrumentation
# ---------------------------------------------------------------------------

def run_golden_with_vadd_stats(program, patches_int8, num_classes=1000):
    """Run golden model and return (logits, vadd_stats).

    vadd_stats is a list of {'total': int, 'saturated': int} for each
    INT8 VADD instruction executed (pos_embed + residual adds).
    """
    state = MachineState()
    sim = Simulator(state)
    sim.load_program(program)

    M, N = patches_int8.shape
    N_pad = pad_dim(N)
    if N < N_pad:
        p = np.zeros((M, N_pad), dtype=np.int8)
        p[:M, :N] = patches_int8
        patches_int8 = p
    patch_bytes = patches_int8.tobytes()
    dram_off = program.input_offset
    state.dram[dram_off:dram_off + len(patch_bytes)] = patch_bytes

    vadd_stats = []
    _original = sim._exec_vadd  # bound method

    def _diagnostic_vadd(insn):
        if sim.state.tile_config is not None and insn.src1_buf == BUF_ABUF:
            m_tiles = sim.state.tile_config[0] + 1
            n_tiles = sim.state.tile_config[1] + 1
            M_t = m_tiles * 16
            N_t = n_tiles * 16
            src1 = mem_module.read_int8_tile(sim.state, insn.src1_buf, insn.src1_off, M_t, N_t)
            src2 = mem_module.read_int8_tile(sim.state, insn.src2_buf, insn.src2_off, M_t, N_t)
            sums = src1.astype(np.int16) + src2.astype(np.int16)
            saturated = int(np.sum((sums > 127) | (sums < -128)))
            vadd_stats.append({"total": M_t * N_t, "saturated": saturated})
        _original(insn)

    sim._exec_vadd = _diagnostic_vadd
    sim.run()

    logits = state.accum[:num_classes].copy().astype(np.float32)
    return logits, vadd_stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _vadd_label(idx):
    """Map VADD index to a human-readable label.

    Expected order: pos_embed_add(0), then per block: res1(2i+1), res2(2i+2).
    """
    if idx == 0:
        return "pos_embed_add"
    bidx = (idx - 1) // 2
    rtype = "res1" if (idx - 1) % 2 == 0 else "res2"
    return f"block{bidx:02d}_{rtype}"


def print_report(residual1_maxabs, residual2_maxabs, qkt_maxabs,
                 vadd_stats_per_image, block_input_scale):

    W = 90

    # --- A1: Per-block residual ranges ---
    print("\n" + "=" * W)
    print("  A1: Per-Block Residual Output Ranges vs. constant block_input_scale")
    print("=" * W)
    print(f"\n  Constant block_input_scale = {block_input_scale:.5f}  "
          f"(covers max_abs ≤ {block_input_scale * 127:.2f})\n")
    hdr = (f"  {'Blk':>4}  {'Res1 max (avg)':>15}  {'Ideal scale1':>13}  {'Ratio1':>7}  "
           f"{'Res2 max (avg)':>15}  {'Ideal scale2':>13}  {'Ratio2':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    any_mismatch = False
    for i in range(DEPTH):
        r1 = residual1_maxabs[i]
        r2 = residual2_maxabs[i]
        if not r1 or not r2:
            continue
        r1m = np.mean(r1)
        r2m = np.mean(r2)
        id1 = r1m / 127.0
        id2 = r2m / 127.0
        rat1 = id1 / block_input_scale if block_input_scale > 1e-9 else 0.0
        rat2 = id2 / block_input_scale if block_input_scale > 1e-9 else 0.0
        flag1 = " !" if (rat1 > 2.0 or rat1 < 0.5) else "  "
        flag2 = " !" if (rat2 > 2.0 or rat2 < 0.5) else "  "
        if "!" in flag1 or "!" in flag2:
            any_mismatch = True
        print(f"  {i:>4}  {r1m:>15.3f}  {id1:>13.5f}  {rat1:>6.2f}x{flag1}  "
              f"{r2m:>15.3f}  {id2:>13.5f}  {rat2:>6.2f}x{flag2}")

    print()
    if any_mismatch:
        print("  [!] Ratio > 2x  → scale too coarse → residual add CLIPS values")
        print("  [!] Ratio < 0.5x → scale too fine → wastes dynamic range")
    else:
        print("  All ratios within 2x: constant block_input_scale is a reasonable fit.")

    # --- A3: Per-head QKT ranges ---
    print("\n" + "=" * W)
    print("  A3: Per-Head QKT Ranges vs. Hardcoded 6.0/127.0 = 0.04724")
    print("=" * W)
    HARDCODED = 6.0 / 127.0
    print(f"\n  Hardcoded QKT scale = {HARDCODED:.5f}  (covers max_abs ≤ 6.0)\n")
    hdr2 = f"  {'Blk':>4}  {'Hd':>3}  {'QKT max (avg)':>14}  {'Ideal scale':>12}  {'Ratio':>7}  Verdict"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    clipping_heads = []
    for i in range(DEPTH):
        for h in range(NUM_HEADS):
            vals = qkt_maxabs[(i, h)]
            if not vals:
                continue
            mean_max = np.mean(vals)
            ideal = mean_max / 127.0
            ratio = ideal / HARDCODED
            if ratio > 1.3:
                verdict = "CLIPS  <-- quantizes with loss"
                clipping_heads.append((i, h, mean_max, ratio))
            elif ratio < 0.5:
                verdict = "wastes range"
            else:
                verdict = "OK"
            print(f"  {i:>4}  {h:>3}  {mean_max:>14.3f}  {ideal:>12.5f}  {ratio:>6.2f}x  {verdict}")

    print()
    if clipping_heads:
        print(f"  [!] {len(clipping_heads)} head(s) clip the QKT INT8 range → calibrate per-head QKT scales (fix B1)")
    else:
        print("  All QKT ranges fit within 6.0 → hardcoded scale is fine.")

    # --- VADD Saturation ---
    print("\n" + "=" * W)
    print("  VADD Saturation (INT8 residual/pos_embed adds)")
    print("=" * W)

    if not vadd_stats_per_image:
        print("  No INT8 VADD stats collected.")
    else:
        n_vadds = max(len(s) for s in vadd_stats_per_image)
        print(f"\n  {'VADD #':>7}  {'Label':<20}  {'Avg sat %':>10}  {'Max sat %':>10}  Note")
        print("  " + "-" * 70)
        any_clip = False
        for idx in range(n_vadds):
            pcts = []
            for stats in vadd_stats_per_image:
                if idx < len(stats):
                    s = stats[idx]
                    pcts.append(100.0 * s["saturated"] / max(s["total"], 1))
            if not pcts:
                continue
            avg_p = np.mean(pcts)
            max_p = np.max(pcts)
            note = ""
            if avg_p > 5.0:
                note = "!!! heavy clipping"
                any_clip = True
            elif avg_p > 1.0:
                note = "! moderate clipping"
                any_clip = True
            label = _vadd_label(idx)
            print(f"  {idx:>7}  {label:<20}  {avg_p:>10.2f}%  {max_p:>10.2f}%  {note}")
        print()
        if any_clip:
            print("  [!] Saturation > 1% → residual adds are clipping → fix B2 (per-block scales)")
        else:
            print("  Saturation < 1% in all residual adds → VADD clipping is not a major issue.")

    # --- Summary ---
    print("\n" + "=" * W)
    print("  SUMMARY & RECOMMENDED NEXT STEPS")
    print("=" * W)
    print()

    issues = []
    # Check residual scale mismatch
    for i in range(DEPTH):
        r2 = residual2_maxabs[i]
        if r2:
            ratio = (np.mean(r2) / 127.0) / block_input_scale
            if ratio > 2.0 or ratio < 0.5:
                issues.append(f"  H1: Block {i} residual2 scale ratio = {ratio:.2f}x → implement B2 (per-block scales)")
                break

    # Check QKT clipping
    if clipping_heads:
        issues.append(f"  H5: {len(clipping_heads)} heads have QKT max_abs > 6.0 → implement B1 (calibrate QKT scales)")

    # Check VADD saturation
    if vadd_stats_per_image:
        for idx in range(min(25, max(len(s) for s in vadd_stats_per_image))):
            pcts = [100.0 * s[idx]["saturated"] / max(s[idx]["total"], 1)
                    for s in vadd_stats_per_image if idx < len(s)]
            if pcts and np.mean(pcts) > 1.0:
                issues.append(f"  H4: VADD #{idx} ({_vadd_label(idx)}) avg saturation {np.mean(pcts):.1f}% → fix B2")
                break

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  No major issues detected in these diagnostics.")
        print("  Consider investigating H3 (softmax double-quantization) via A4.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase A diagnostics for INT8 accuracy")
    parser.add_argument("--max-images", type=int, default=5,
                        help="Number of images to use (default: 5 for speed)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TACCEL Accuracy Diagnostics — Phase A")
    print("=" * 60)

    print("\n[1/5] Loading model...")
    model, state_dict = load_model()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    print(f"\n[2/5] Downloading {args.max_images} images...")
    images = collect_images(args.max_images)
    if not images:
        print("  No images downloaded.")
        sys.exit(1)
    print(f"  Got {len(images)} images")

    print("\n[3/5] Compiling INT8 program (calibration from collected images)...")
    sample_imgs = [img for _, img in images]
    program, cal_scales = compile_model(model, state_dict, sample_imgs, processor)
    block_input_scale = cal_scales.get("pos_embed_add", 14.0 / 127.0)
    embed_scale = block_input_scale
    print(f"  block_input_scale (pos_embed_add) = {block_input_scale:.5f}")

    print("\n[4/5] Collecting FP32 activation statistics (hooks)...")
    res1, res2, qkt = collect_fp32_stats(model, processor, images)
    print("  Done.")

    print("\n[5/5] Running golden model with VADD saturation instrumentation...")
    vadd_stats_per_image = []
    for i, (img_id, img) in enumerate(images, 1):
        patches_int8, _ = patch_embed_int8(model, processor, img, embed_scale)
        _, vstats = run_golden_with_vadd_stats(program, patches_int8)
        vadd_stats_per_image.append(vstats)
        print(f"  Image {i}/{len(images)}: {len(vstats)} INT8 VADD calls recorded")

    print_report(res1, res2, qkt, vadd_stats_per_image, block_input_scale)


if __name__ == "__main__":
    main()
