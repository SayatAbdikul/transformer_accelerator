#!/usr/bin/env python3
"""Compare FP32 vs INT8-quantized DeiT-tiny inference accuracy.

Measures the accuracy drop introduced by our per-channel INT8 weight
quantization scheme.  We use "fake quantization": weights are quantized
to INT8 and immediately dequantized back to FP32, then run through the
normal PyTorch forward pass.  This isolates the rounding error without
needing the full hardware simulator.

Two experiments:
  1. Weight-only quantization  – activations stay FP32
  2. Weight + activation quantization – activations also fake-quantized

Usage:
    python compare_accuracy.py [--max-images 100] [--no-act-quant]
"""
import argparse
import sys
import os
import json
import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoImageProcessor, AutoConfig, AutoModelForImageClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taccel.quantizer.fake_quant import (
    apply_weight_quantization,
    calibrate_activation_scales,
    ActivationQuantizer,
    compute_metrics,
)

MODEL_NAME  = "facebook/deit-tiny-patch16-224"
WEIGHTS_PATH = "pytorch_model.bin"

# ── 150 candidate COCO val2017 image IDs ──────────────────────────────────────
# Drawn from the official COCO val2017 annotation set.  Not all IDs exist;
# fetch_image() silently skips 404s so we over-provision and stop at --max-images.
COCO_VAL_IDS = [
    # fmt: off
      139,   285,   632,   724,   776,   785,   802,   872,   885,  1000,
     1268,  1296,  1353,  1425,  1503,  1532,  1584,  1761,  1818,  1993,
     2006,  2052,  2153,  2261,  2473,  2478,  2532,  2685,  3014,  3501,
     3717,  3845,  4024,  4519,  5037,  5190,  5586,  5802,  6040,  6444,
     6587,  6723,  7386,  7513,  7816,  8021,  8202,  8532,  8645,  9400,
     9483,  9590,  9769, 10707, 11051, 11765, 12062, 13004, 14091, 15335,
    15745, 16977, 17436, 18150, 18519, 20247, 21903, 22091, 22992, 23226,
    24021, 25057, 26042, 27803, 28388, 29187, 30504, 31153, 32008, 33397,
    34120, 34870, 36132, 37615, 38338, 39769, 40083, 40671, 41473, 42070,
    44699, 45596, 46750, 48153, 49091, 50007, 51191, 52413, 54171, 55440,
    56127, 57364, 58636, 59765, 60342, 61418, 62808, 63257, 65862, 67406,
    68446, 70141, 71286, 73269, 74717, 75287, 76791, 78823, 80010, 82247,
    83612, 85328, 87038, 89086, 91185, 93112, 95060, 97438, 99432,101022,
   103189,105128,107263,109523,111579,113403,115728,117465,119516,121373,
   123266,125018,127137,128618,130199,132032,133944,135722,137298,139099,
    # fmt: on
]
COCO_BASE = "http://images.cocodataset.org/val2017/{:012d}.jpg"


# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    print(f"  Loading {MODEL_NAME} from {WEIGHTS_PATH} ...")
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_config(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_processor():
    return AutoImageProcessor.from_pretrained(MODEL_NAME)


def fetch_image(img_id: int):
    """Download one COCO val image. Returns (img_id, PIL.Image) or (img_id, None)."""
    url = COCO_BASE.format(img_id)
    try:
        r = requests.get(url, timeout=15, stream=True)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return img_id, img
    except Exception:
        pass
    return img_id, None


def collect_images(max_images: int, workers: int = 12) -> list:
    """Download up to max_images from COCO_VAL_IDS using parallel HTTP requests.

    Returns list of (img_id, PIL.Image) sorted by img_id.
    """
    print(f"  Downloading up to {max_images} COCO val2017 images "
          f"({len(COCO_VAL_IDS)} candidates, {workers} parallel workers) ...")

    collected = []
    remaining = list(COCO_VAL_IDS)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_image, img_id): img_id for img_id in remaining}
        done = 0
        for future in as_completed(futures):
            img_id, img = future.result()
            if img is not None:
                collected.append((img_id, img))
                done += 1
                print(f"    [{done:>3}/{max_images}] id={img_id:6d}  ✓", flush=True)
                if done >= max_images:
                    # Cancel pending futures
                    for f in futures:
                        f.cancel()
                    break
            else:
                print(f"           id={img_id:6d}  ✗ (skip)", flush=True)

    collected.sort(key=lambda x: x[0])
    return collected[:max_images]


def run_inference(model, processor, image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.squeeze(0).cpu().numpy()


def print_header(title: str):
    width = 72
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_metrics_row(idx: int, img_id: int, m: dict, id2label: dict):
    label_fp32 = id2label.get(m["top1_fp32"], str(m["top1_fp32"]))[:28]
    label_q    = id2label.get(m["top1_q"],    str(m["top1_q"]))[:28]
    sym = "✓" if m["top1_match"] else "✗"
    print(f"  {idx:>3}  id={img_id:<6}  "
          f"fp32={m['top1_fp32']:>4} {label_fp32:<28}  "
          f"int8={m['top1_q']:>4} {label_q:<28}  {sym}  "
          f"cos={m['cosine_sim']:.4f}  snr={m['logit_snr_db']:>6.1f}dB  "
          f"kl={m['softmax_kl_div']:.5f}")


def aggregate(all_metrics: list) -> dict:
    keys = ["top1_match", "top5_match", "cosine_sim", "logit_mse",
            "logit_mae", "logit_snr_db", "softmax_kl_div"]
    agg = {}
    for k in keys:
        vals = [float(m[k]) for m in all_metrics if k in m]
        if vals:
            agg[k + "_mean"] = float(np.mean(vals))
            agg[k + "_std"]  = float(np.std(vals))
            agg[k + "_min"]  = float(np.min(vals))
            agg[k + "_max"]  = float(np.max(vals))
    return agg


def print_aggregate(agg: dict, n: int, label: str):
    print(f"\n  ── {label} aggregate ({n} images) ──────────────────────────────")
    print(f"    top-1 preserved  : {agg.get('top1_match_mean',0)*100:>6.1f}%")
    print(f"    top-5 overlap    : {agg.get('top5_match_mean',0)*100:>6.1f}%")
    print(f"    cosine sim       : {agg.get('cosine_sim_mean',0):.6f}  "
          f"± {agg.get('cosine_sim_std',0):.6f}")
    print(f"    logit MSE        : {agg.get('logit_mse_mean',0):.5f}  "
          f"± {agg.get('logit_mse_std',0):.5f}")
    print(f"    logit SNR        : {agg.get('logit_snr_db_mean',0):>6.2f} dB  "
          f"± {agg.get('logit_snr_db_std',0):.2f}")
    print(f"    softmax KL div   : {agg.get('softmax_kl_div_mean',0):.6f}  "
          f"± {agg.get('softmax_kl_div_std',0):.6f}")


def run_experiment(name, model_fp32, model_q, processor, images, id2label):
    """Run both models on every image and return metrics list."""
    print_header(name)
    metrics = []
    for idx, (img_id, img) in enumerate(images, 1):
        logits_fp32 = run_inference(model_fp32, processor, img)
        logits_q    = run_inference(model_q,    processor, img)
        m = compute_metrics(logits_fp32, logits_q)
        m["img_id"] = img_id
        print_metrics_row(idx, img_id, m, id2label)
        metrics.append(m)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=100,
                        help="Number of COCO val images to test (default: 100)")
    parser.add_argument("--workers", type=int, default=12,
                        help="Parallel download workers (default: 12)")
    parser.add_argument("--no-act-quant", action="store_true",
                        help="Skip W8A8 experiment (only run W8)")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────
    print_header("Loading model")
    model_fp32 = load_model()
    processor  = load_processor()
    id2label   = model_fp32.config.id2label
    param_count = sum(p.numel() for p in model_fp32.parameters())
    n_linear    = sum(1 for m in model_fp32.modules()
                      if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)))
    print(f"  Parameters              : {param_count:,}")
    print(f"  Linear/Conv layers      : {n_linear}")

    # ── Weight quantization ───────────────────────────────────────────────
    print_header("Applying per-channel INT8 weight quantization")
    model_wq, n_quantized = apply_weight_quantization(model_fp32)
    print(f"  Quantized {n_quantized} weight tensors")

    total_sq_err = total_sq_sig = total_n = 0.0
    for (_, p1), (_, p2) in zip(model_fp32.named_parameters(),
                                  model_wq.named_parameters()):
        d = (p1.data - p2.data).float()
        total_sq_err += d.pow(2).sum().item()
        total_sq_sig += p1.data.float().pow(2).sum().item()
        total_n      += p1.numel()
    weight_mse = total_sq_err / total_n
    weight_snr = 10.0 * np.log10(total_sq_sig / max(total_sq_err, 1e-30))
    print(f"  Weight MSE (FP32 vs INT8-dequant) : {weight_mse:.2e}")
    print(f"  Weight SNR                        : {weight_snr:.1f} dB")

    # ── Download images ───────────────────────────────────────────────────
    print_header(f"Collecting {args.max_images} COCO val2017 images")
    images = collect_images(args.max_images, workers=args.workers)
    if not images:
        print("  No images downloaded. Check network connectivity.")
        sys.exit(1)
    print(f"\n  Collected {len(images)} images successfully.")

    # ── Experiment 1: W8 (weight-only) ───────────────────────────────────
    wq_metrics = run_experiment(
        f"Experiment 1 — W8: weight-only INT8 ({len(images)} images)",
        model_fp32, model_wq, processor, images, id2label,
    )
    agg_wq = aggregate(wq_metrics)
    print_aggregate(agg_wq, len(wq_metrics), "W8")

    # ── Experiment 2: W8A8 (weight + activation) ──────────────────────────
    agg_waq = {}
    waq_metrics = []
    if not args.no_act_quant:
        print_header("Calibrating activation scales (using all test images)")
        sample_inputs = [processor(images=img, return_tensors="pt")
                         for _, img in images]
        act_scales = calibrate_activation_scales(model_fp32, sample_inputs)
        print(f"  Calibrated {len(act_scales)} activation tensors")

        # Show 5 representative scales
        dense_scales = {k: v for k, v in act_scales.items() if "dense" in k}
        for name, scale in list(dense_scales.items())[:5]:
            print(f"    {name:<55}: scale={scale:.5f}  (max_abs≈{scale*127:.3f})")

        act_qtz = ActivationQuantizer(act_scales)
        act_qtz.attach(model_wq)

        waq_metrics = run_experiment(
            f"Experiment 2 — W8A8: weight + activation INT8 ({len(images)} images)",
            model_fp32, model_wq, processor, images, id2label,
        )
        act_qtz.remove()
        agg_waq = aggregate(waq_metrics)
        print_aggregate(agg_waq, len(waq_metrics), "W8A8")

    # ── Final summary table ───────────────────────────────────────────────
    print_header("Summary — FP32 → INT8 quantization impact")

    def _f(agg, k, fmt=".1f", pct=False):
        v = agg.get(k + "_mean", float("nan"))
        if pct:
            return f"{v*100:{fmt}}%"
        return f"{v:{fmt}}"

    w8a8_col = "  W8A8  " if not args.no_act_quant else "  n/a   "

    print(f"""
  Model                  : {MODEL_NAME}
  Images tested          : {len(images)} COCO val2017
  Quantized layers       : {n_quantized} Linear/Conv2d
  Weight quantization    : per-channel symmetric INT8
  Weight SNR             : {weight_snr:.1f} dB

  ┌─────────────────────────────────┬──────────────────────┬──────────────────────┐
  │ Metric                          │    W8 (weight-only)  │{w8a8_col}(w+act)   │
  ├─────────────────────────────────┼──────────────────────┼──────────────────────┤
  │ top-1 class preserved           │{_f(agg_wq,'top1_match','.1f',True):>21} │{_f(agg_waq,'top1_match','.1f',True) if agg_waq else '  n/a':>21} │
  │ top-5 overlap                   │{_f(agg_wq,'top5_match','.1f',True):>21} │{_f(agg_waq,'top5_match','.1f',True) if agg_waq else '  n/a':>21} │
  │ logit cosine similarity (mean)  │{_f(agg_wq,'cosine_sim','8.6f'):>21} │{_f(agg_waq,'cosine_sim','8.6f') if agg_waq else '  n/a':>21} │
  │ logit cosine similarity (min)   │{agg_wq.get('cosine_sim_min',float('nan')):>21.6f} │{agg_waq.get('cosine_sim_min',float('nan')):>21.6f} │
  │ logit MSE (mean)                │{_f(agg_wq,'logit_mse','8.5f'):>21} │{_f(agg_waq,'logit_mse','8.5f') if agg_waq else '  n/a':>21} │
  │ logit SNR (mean, dB)            │{_f(agg_wq,'logit_snr_db','8.2f'):>21} │{_f(agg_waq,'logit_snr_db','8.2f') if agg_waq else '  n/a':>21} │
  │ logit SNR (min, dB)             │{agg_wq.get('logit_snr_db_min',float('nan')):>21.2f} │{agg_waq.get('logit_snr_db_min',float('nan')):>21.2f} │
  │ softmax KL divergence (mean)    │{_f(agg_wq,'softmax_kl_div','8.6f'):>21} │{_f(agg_waq,'softmax_kl_div','8.6f') if agg_waq else '  n/a':>21} │
  └─────────────────────────────────┴──────────────────────┴──────────────────────┘

  top-1 mismatch images (W8):""")

    mismatches_w8 = [(m["img_id"], m["top1_fp32"], m["top1_q"])
                     for m in wq_metrics if not m["top1_match"]]
    if mismatches_w8:
        for img_id, c_fp32, c_q in mismatches_w8:
            lf = id2label.get(c_fp32, str(c_fp32))
            lq = id2label.get(c_q,    str(c_q))
            print(f"    id={img_id:<6}  fp32={c_fp32} {lf}  →  int8={c_q} {lq}")
    else:
        print("    none — perfect top-1 agreement on all images")

    if not args.no_act_quant and waq_metrics:
        print(f"\n  top-1 mismatch images (W8A8):")
        mismatches_waq = [(m["img_id"], m["top1_fp32"], m["top1_q"])
                          for m in waq_metrics if not m["top1_match"]]
        if mismatches_waq:
            for img_id, c_fp32, c_q in mismatches_waq[:20]:
                lf = id2label.get(c_fp32, str(c_fp32))
                lq = id2label.get(c_q,    str(c_q))
                print(f"    id={img_id:<6}  fp32={c_fp32} {lf}  →  int8={c_q} {lq}")
            if len(mismatches_waq) > 20:
                print(f"    ... and {len(mismatches_waq)-20} more")
        else:
            print("    none")

    print()

    # ── Save JSON ─────────────────────────────────────────────────────────
    results = {
        "config": {
            "model": MODEL_NAME,
            "n_images": len(images),
            "n_quantized_layers": n_quantized,
            "weight_mse": weight_mse,
            "weight_snr_db": weight_snr,
        },
        "W8": {"aggregate": agg_wq, "per_image": wq_metrics},
    }
    if not args.no_act_quant:
        results["W8A8"] = {"aggregate": agg_waq, "per_image": waq_metrics}

    out = "quantization_comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Full results saved to {out}\n")


if __name__ == "__main__":
    main()
