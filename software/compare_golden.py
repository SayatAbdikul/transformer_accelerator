#!/usr/bin/env python3
"""Compare FP32 PyTorch inference vs INT8 golden model (hardware simulator).

Downloads COCO val images, runs both:
  1. FP32 DeiT-tiny via PyTorch (reference)
  2. INT8 compiled program via the TACCEL golden model simulator

Reports top-K predictions side-by-side and accuracy metrics.

Usage:
    python3 compare_golden.py [--max-images 5] [--top-k 5]
"""
import argparse
import sys
import os
import json
import time
import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoImageProcessor, AutoConfig, AutoModelForImageClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taccel.compiler.compiler import Compiler
from taccel.compiler.graph_extract import EMBED_DIM, NUM_PATCHES, SEQ_LEN, PATCH_SIZE, NUM_HEADS, DEPTH
from taccel.assembler.assembler import ProgramBinary
from taccel.golden_model import Simulator, MachineState
from taccel.quantizer.quantize import quantize_tensor
from taccel.quantizer.calibrate import calibrate_model
from taccel.compiler.tiler import pad_dim

MODEL_NAME = "facebook/deit-tiny-patch16-224"
WEIGHTS_PATH = "pytorch_model.bin"

COCO_VAL_IDS = [
    39769, 139, 285, 632, 724, 776, 785, 872, 1000, 1296,
    1353, 1503, 1761, 2006, 2153, 2473, 2685, 3501, 3845, 5037,
]
COCO_BASE = "http://images.cocodataset.org/val2017/{:012d}.jpg"


def load_model():
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_config(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, state_dict


def fetch_image(img_id: int):
    url = COCO_BASE.format(img_id)
    try:
        r = requests.get(url, timeout=15, stream=True)
        if r.status_code == 200:
            return img_id, Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        pass
    return img_id, None


def collect_images(max_images: int):
    print(f"  Downloading up to {max_images} COCO val2017 images...")
    collected = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_image, img_id): img_id for img_id in COCO_VAL_IDS}
        for future in as_completed(futures):
            img_id, img = future.result()
            if img is not None:
                collected.append((img_id, img))
                if len(collected) >= max_images:
                    for f in futures:
                        f.cancel()
                    break
    collected.sort(key=lambda x: x[0])
    return collected[:max_images]


def fp32_inference(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.squeeze(0).cpu().numpy()


def patch_embed_int8(model, processor, image, act_scale=None):
    """Run patch embedding on CPU and quantize to INT8 for the accelerator.

    Returns (int8_patches, scale) where int8_patches is [NUM_PATCHES, EMBED_DIM].

    act_scale must match the embedding scale used in the compiled program for
    CLS token and position embeddings (cal_scales["pos_embed_add"]).
    """
    if act_scale is None:
        act_scale = 14.0 / 127.0  # safe default: covers patch+pos_embed sum ~13.8
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # [1, 3, 224, 224]

    with torch.no_grad():
        # Extract the patch embedding layer
        patch_embed = model.vit.embeddings.patch_embeddings
        # Conv2d: [1, 3, 224, 224] -> [1, 192, 14, 14]
        embedded = patch_embed.projection(pixel_values)
        # Reshape to [196, 192]
        embedded = embedded.flatten(2).transpose(1, 2).squeeze(0)  # [196, 192]

    patches_fp32 = embedded.numpy().astype(np.float32)
    patches_int8 = np.clip(np.round(patches_fp32 / act_scale), -128, 127).astype(np.int8)
    return patches_int8, act_scale


def golden_inference(program, patches_int8, num_classes=1000):
    """Run the golden model simulator and return INT32 logits."""
    state = MachineState(dram_data=program.data)
    sim = Simulator(state)
    sim.load_program(program)

    # Write embedded patches to ABUF starting at row 1 (byte offset 192)
    # Row 0 is reserved for CLS token (loaded by the program from DRAM)
    M, N = patches_int8.shape  # [196, 192]
    N_pad = pad_dim(N)
    # Pad columns to multiple of 16 if needed
    if N < N_pad:
        patches_padded = np.zeros((M, N_pad), dtype=np.int8)
        patches_padded[:M, :N] = patches_int8
    else:
        patches_padded = patches_int8

    # Write row by row to ABUF starting at byte offset = 1 row * N_pad bytes
    abuf_offset = N_pad  # skip row 0 (CLS token)
    patch_bytes = patches_padded.tobytes()
    state.abuf[abuf_offset:abuf_offset + len(patch_bytes)] = patch_bytes

    # Run simulation
    count = sim.run()

    # Extract logits from ACCUM buffer (INT32)
    logits_int32 = state.accum[:num_classes].copy()
    return logits_int32.astype(np.float32), count, state.cycle_count


def calibrate_softmax_scales(model, sample_inputs: list) -> dict:
    """Capture max attention probability per (layer, head) for softmax scale calibration.

    Uses model's built-in output_attentions=True to get attention weights directly.
    Returns {(layer_idx, head_idx): max_prob} across all sample inputs.
    """
    max_attn = {}  # {(layer_idx, head_idx): max_prob}

    # Build a temporary copy with eager attn + output_attentions enabled.
    # output_attentions requires attn_implementation='eager' (SDPA doesn't support it).
    from transformers import AutoConfig
    cfg = model.config.__class__.from_dict(model.config.to_dict())
    cfg.output_attentions = True
    attn_model = type(model)(cfg)
    attn_model.load_state_dict(model.state_dict())
    attn_model.eval()

    with torch.no_grad():
        for inp in sample_inputs:
            inp_dict = dict(inp.items()) if hasattr(inp, 'items') else {'pixel_values': inp}
            outputs = attn_model(**inp_dict)
            # outputs.attentions: tuple of DEPTH tensors, each [batch, num_heads, seq, seq]
            if outputs.attentions:
                for layer_idx, attn_weights in enumerate(outputs.attentions):
                    for h in range(attn_weights.shape[1]):
                        max_prob = float(attn_weights[0, h].max().item())
                        key = (layer_idx, h)
                        max_attn[key] = max(max_attn.get(key, 0.0), max_prob)

    del attn_model
    return max_attn


def build_calibration_scales(calibration, softmax_max_probs: dict = None) -> dict:
    """Map calibration module output scales to IR node names used by the codegen."""
    cal = calibration.scales  # module_name → float scale

    def get(module_name, default=6.0 / 127.0):
        return cal.get(module_name, default)

    scales = {}
    # Positional embedding add output: similar to the embedding dropout output
    emb_scale = get("vit.embeddings.dropout", get("vit.embeddings", 6.0 / 127.0))
    scales["pos_embed_add"] = emb_scale

    # Propagate the block-input scale through all 12 encoder blocks.
    # The INT8 residual VADD (x + branch_output) requires BOTH operands to be at
    # the SAME scale, otherwise each INT8 unit has different weight and the sum is
    # wrong.  We enforce this by forcing out_proj and fc2 to REQUANT their output
    # to the current block_input_scale, so the VADD operands are always compatible.
    # block_input_scale starts as the pos_embed_add output scale and stays constant
    # because every residual output is forced back to the same scale.
    block_input_scale = emb_scale

    for i in range(DEPTH):
        b = f"block{i}"
        p = f"vit.encoder.layer.{i}"

        ln1_scale = get(f"{p}.layernorm_before")
        scales[f"{b}_ln1"] = ln1_scale

        q_scale = get(f"{p}.attention.attention.query")
        k_scale = get(f"{p}.attention.attention.key")
        v_scale = get(f"{p}.attention.attention.value")

        for h in range(NUM_HEADS):
            scales[f"{b}_head{h}_query"] = q_scale
            scales[f"{b}_head{h}_key"] = k_scale
            scales[f"{b}_head{h}_value"] = v_scale
            scales[f"{b}_head{h}_qkt"] = 6.0 / 127.0
            # scale_mul is a no-op in codegen (0.125 is baked into QKT REQUANT),
            # so the WBUF data has the same scale as the QKT output.
            scales[f"{b}_head{h}_scale"] = 6.0 / 127.0
            # Per-head softmax scale: calibrated to each head's max attention probability.
            # Heads with sharp CLS self-attention (≈99%) get a coarse scale;
            # heads with diffuse attention (≈10%) get a fine scale preserving variation.
            if softmax_max_probs and (i, h) in softmax_max_probs:
                max_prob = softmax_max_probs[(i, h)]
            else:
                max_prob = 0.20
            scales[f"{b}_head{h}_softmax"] = max(max_prob, 1e-4) / 127.0
            scales[f"{b}_head{h}_attn_v"] = v_scale

        # Concatenated head outputs feed into out_proj
        scales[f"{b}_concat"] = v_scale

        # Force out_proj REQUANT to block_input_scale so that the residual1 VADD
        # (block_input + out_proj_output) has both operands at the same scale.
        scales[f"{b}_out_proj"] = block_input_scale
        scales[f"{b}_residual1"] = block_input_scale

        ln2_scale = get(f"{p}.layernorm_after")
        scales[f"{b}_ln2"] = ln2_scale

        fc1_scale = get(f"{p}.intermediate.dense")
        scales[f"{b}_fc1"] = fc1_scale
        scales[f"{b}_gelu"] = fc1_scale

        # Force fc2 REQUANT to block_input_scale so that the residual2 VADD
        # (residual1_output + fc2_output) has both operands at the same scale.
        # residual1 output is already at block_input_scale from the fix above.
        scales[f"{b}_fc2"] = block_input_scale
        scales[f"{b}_residual2"] = block_input_scale
        # block_input_scale is preserved: next block's input is this block's
        # residual2 output, which is forced to block_input_scale.

    final_ln_scale = get("vit.layernorm")
    scales["final_ln"] = final_ln_scale
    scales["cls_extract"] = final_ln_scale
    scales["classifier"] = get("classifier", 1.0)
    return scales


def compile_model(model, state_dict, sample_images, processor):
    """Compile the model to a ProgramBinary using calibration from sample images.

    Returns (program, cal_scales) so callers can pass the embedding scale to
    patch_embed_int8() for consistent INT8 quantization.
    """
    print("  Calibrating activation scales...")
    sample_inputs = [processor(images=img, return_tensors="pt") for img in sample_images]
    calibration = calibrate_model(model, sample_inputs)
    softmax_max_probs = calibrate_softmax_scales(model, sample_inputs)
    cal_scales = build_calibration_scales(calibration, softmax_max_probs)
    embed_scale = cal_scales.get("pos_embed_add", 14.0 / 127.0)
    print(f"  Got {len(cal_scales)} calibration entries; "
          f"block0_ln1={cal_scales.get('block0_ln1', 'N/A'):.5f}  "
          f"pos_embed_add={embed_scale:.5f}")
    # Show per-head max probs for block 0
    b0_probs = [softmax_max_probs.get((0, h), 0.20) for h in range(NUM_HEADS)]
    print(f"  Softmax max_prob block0: " + " ".join(f"h{h}={p:.3f}" for h, p in enumerate(b0_probs)))

    compiler = Compiler()
    program = compiler.compile(state_dict, calibration=type('C', (), {'scales': cal_scales})())
    return program, cal_scales


def main():
    parser = argparse.ArgumentParser(description="Compare FP32 vs golden model inference")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    width = 72
    def header(title):
        print("\n" + "=" * width)
        print(f"  {title}")
        print("=" * width)

    # Load model
    header("Loading DeiT-tiny model")
    model, state_dict = load_model()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    id2label = model.config.id2label
    print(f"  Loaded {sum(p.numel() for p in model.parameters()):,} parameters")

    # Download images
    header(f"Collecting {args.max_images} COCO val2017 images")
    images = collect_images(args.max_images)
    if not images:
        print("  No images downloaded. Check network.")
        sys.exit(1)
    print(f"  Got {len(images)} images")

    # Compile model (now using calibration from actual images)
    header("Compiling model to INT8 program")
    t0 = time.time()
    program, cal_scales = compile_model(model, state_dict, [img for _, img in images[:3]], processor)
    embed_scale = cal_scales.get("pos_embed_add", 14.0 / 127.0)
    dt = time.time() - t0
    print(f"  {program.insn_count} instructions, {len(program.data):,} bytes data")
    print(f"  Compiled in {dt:.1f}s")

    # Run comparison
    header(f"Running FP32 vs Golden Model on {len(images)} images")
    top1_matches = 0
    top5_matches = 0
    results = []

    for idx, (img_id, img) in enumerate(images, 1):
        print(f"\n  --- Image {idx}/{len(images)}: COCO id={img_id} ---")

        # FP32 reference
        logits_fp32 = fp32_inference(model, processor, img)
        fp32_top = np.argsort(logits_fp32)[::-1][:args.top_k]

        # Patch embed + golden model
        patches_int8, act_scale = patch_embed_int8(model, processor, img, embed_scale)
        logits_golden, insn_count, cycles = golden_inference(program, patches_int8)
        golden_top = np.argsort(logits_golden)[::-1][:args.top_k]

        # Compare
        top1_match = fp32_top[0] == golden_top[0]
        top5_set_fp32 = set(fp32_top[:5])
        top5_set_golden = set(golden_top[:5])
        top5_overlap = len(top5_set_fp32 & top5_set_golden) / 5.0

        if top1_match:
            top1_matches += 1
        top5_matches += top5_overlap

        # Normalize golden logits for cosine similarity
        # (golden logits are INT32 accumulator values, different scale from FP32)
        norm_fp32 = np.linalg.norm(logits_fp32)
        norm_golden = np.linalg.norm(logits_golden)
        if norm_fp32 > 0 and norm_golden > 0:
            cosine_sim = np.dot(logits_fp32, logits_golden) / (norm_fp32 * norm_golden)
        else:
            cosine_sim = 0.0

        # Rank correlation (Spearman-like on top predictions)
        fp32_ranks = np.argsort(np.argsort(logits_fp32)[::-1])
        golden_ranks = np.argsort(np.argsort(logits_golden)[::-1])

        sym = "MATCH" if top1_match else "MISMATCH"

        print(f"  FP32   top-1: class {fp32_top[0]:>4} = {id2label.get(int(fp32_top[0]), '?')}")
        print(f"  Golden top-1: class {golden_top[0]:>4} = {id2label.get(int(golden_top[0]), '?')}")
        print(f"  Result: {sym}  |  cosine_sim={cosine_sim:.4f}  |  top5_overlap={top5_overlap*100:.0f}%")
        print(f"  Golden: {insn_count} insns, {cycles:,} cycles")

        print(f"\n  {'Rank':<6} {'FP32':>6} {'FP32 label':<30} {'Golden':>6} {'Golden label':<30}")
        print(f"  {'-'*6} {'-'*6} {'-'*30} {'-'*6} {'-'*30}")
        for r in range(args.top_k):
            fp32_cls = int(fp32_top[r])
            golden_cls = int(golden_top[r])
            fp32_label = id2label.get(fp32_cls, str(fp32_cls))[:30]
            golden_label = id2label.get(golden_cls, str(golden_cls))[:30]
            match_marker = " <=" if fp32_cls == golden_cls else ""
            print(f"  {r+1:<6} {fp32_cls:>6} {fp32_label:<30} {golden_cls:>6} {golden_label:<30}{match_marker}")

        results.append({
            "img_id": img_id,
            "top1_match": bool(top1_match),
            "top5_overlap": float(top5_overlap),
            "cosine_sim": float(cosine_sim),
            "fp32_top5": [int(x) for x in fp32_top[:5]],
            "golden_top5": [int(x) for x in golden_top[:5]],
            "cycles": int(cycles),
        })

    # Summary
    header("Summary")
    n = len(results)
    print(f"  Images tested     : {n}")
    print(f"  Top-1 agreement   : {top1_matches}/{n} ({top1_matches/n*100:.1f}%)")
    print(f"  Top-5 overlap     : {top5_matches/n*100:.1f}% average")
    avg_cos = np.mean([r["cosine_sim"] for r in results])
    print(f"  Cosine similarity : {avg_cos:.4f} average")
    avg_cycles = np.mean([r["cycles"] for r in results])
    print(f"  Avg sim cycles    : {avg_cycles:,.0f}")

    if top1_matches < n:
        print(f"\n  Mismatched images:")
        for r in results:
            if not r["top1_match"]:
                fp32_cls = r["fp32_top5"][0]
                golden_cls = r["golden_top5"][0]
                print(f"    id={r['img_id']:<6}  fp32={fp32_cls} {id2label.get(fp32_cls, '?')}"
                      f"  ->  golden={golden_cls} {id2label.get(golden_cls, '?')}")

    # Save results
    out = "golden_comparison.json"
    with open(out, "w") as f:
        json.dump({"summary": {
            "n_images": n,
            "top1_agreement": top1_matches / n,
            "top5_overlap_avg": top5_matches / n,
            "cosine_sim_avg": float(avg_cos),
        }, "per_image": results}, f, indent=2)
    print(f"\n  Results saved to {out}\n")


if __name__ == "__main__":
    main()
