#!/usr/bin/env python3
"""Compare RTL execution against the software golden model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taccel.assembler.assembler import ProgramBinary
from taccel.golden_model import MachineState, Simulator
from tools.run_golden import load_input_array, write_runtime_inputs


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = REPO_ROOT / "rtl" / "verilator" / "build" / "run_program" / "Vtaccel_top"


class _LocalDeiTProcessor:
    """Offline fallback for DeiT image preprocessing.

    The compiler and runtime compare flow only need a callable object that
    converts PIL images into `pixel_values`. Keeping that logic local lets the
    Phase F sign-off harness run without a network fetch for the processor
    metadata.
    """

    def __init__(self) -> None:
        self.resize_short = 256
        self.crop_size = 224
        self.image_mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def _prepare_one(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB")
        width, height = img.size
        if width == 0 or height == 0:
            raise ValueError("Input image must have nonzero dimensions")

        scale = self.resize_short / min(width, height)
        resized_w = int(round(width * scale))
        resized_h = int(round(height * scale))
        img = img.resize((resized_w, resized_h), resample=Image.Resampling.BILINEAR)

        left = max((resized_w - self.crop_size) // 2, 0)
        top = max((resized_h - self.crop_size) // 2, 0)
        img = img.crop((left, top, left + self.crop_size, top + self.crop_size))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.image_mean) / self.image_std
        return np.transpose(arr, (2, 0, 1))

    def __call__(self, *, images, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError(f"Unsupported return_tensors={return_tensors!r}; only 'pt' is supported")

        if isinstance(images, (list, tuple)):
            batch = [self._prepare_one(image) for image in images]
        else:
            batch = [self._prepare_one(images)]
        pixel_values = torch.from_numpy(np.stack(batch, axis=0))
        return {"pixel_values": pixel_values}


def _load_processor(model_name: str):
    from transformers import AutoImageProcessor

    try:
        return AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
    except OSError as exc:
        print(
            f"[compare_rtl_golden] Falling back to local DeiT processor for {model_name}: {exc}",
            file=sys.stderr,
        )
        return _LocalDeiTProcessor()


def _parse_csv_ints(text: str | None) -> set[int] | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    return {int(chunk.strip()) for chunk in text.split(",") if chunk.strip()}


def _scenario_overrides(name: str) -> dict[str, Any]:
    scenarios: dict[str, dict[str, Any]] = {
        "baseline_default": {},
        "experimental_requant_pc": {
            "requant_pc_out_proj": True,
            "requant_pc_out_proj_blocks": {11},
        },
        "experimental_dequant_add": {
            "dequant_add_residual1_blocks": {11},
        },
        "experimental_softmax_attnv": {
            "fused_softmax_attnv_blocks": {11},
        },
        "experimental_fused_out_proj": {
            "fused_softmax_attnv_blocks": {11},
            "fused_softmax_attnv_accum_out_proj": True,
            "requant_pc_out_proj": True,
            "requant_pc_out_proj_blocks": {11},
        },
    }
    if name not in scenarios:
        raise ValueError(f"Unknown compile scenario: {name}")
    return dict(scenarios[name])


def _ensure_runner_built(runner_path: Path, rebuild: bool) -> None:
    if rebuild or not runner_path.exists():
        subprocess.run(
            ["make", "-C", str(REPO_ROOT / "rtl" / "verilator"), "run_program"],
            check=True,
        )


def _load_program(path: Path) -> ProgramBinary:
    return ProgramBinary.from_bytes(path.read_bytes())


def _load_raw_or_npy_int8(path: Path, rows: int | None = None, cols: int | None = None) -> np.ndarray:
    arr = load_input_array(str(path))
    arr = np.asarray(arr, dtype=np.int8)
    if rows is not None and cols is not None:
        arr = arr.reshape(rows, cols)
    return arr


def _write_raw_int8(path: Path, arr: np.ndarray) -> None:
    np.asarray(arr, dtype=np.int8).reshape(-1).tofile(path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _run_golden_program(
    program: ProgramBinary,
    patches_int8: np.ndarray | None,
    cls_int8: np.ndarray | None,
    folded_pos_embed: bool,
    num_classes: int,
    trace_nodes: list[str] | None = None,
) -> dict[str, Any]:
    state = MachineState(dram_data=program.data)
    sim = Simulator(state)
    sim.load_program(program)
    if trace_nodes:
        sim.enable_trace(trace_nodes)

    if patches_int8 is not None:
        write_runtime_inputs(
            state,
            program,
            patches_int8,
            cls_input=cls_int8,
            folded_pos_embed=folded_pos_embed,
        )

    insn_count = sim.run()
    logits = state.accum[:num_classes].astype(np.int32).tolist()
    return {
        "fault": False,
        "fault_code": 0,
        "instruction_count": int(insn_count),
        "cycle_count": int(state.cycle_count),
        "logits": logits,
        "trace": sim.get_trace_payload() if trace_nodes else None,
    }


def _compile_mode_cycle_budget(min_budget: int, golden_cycles: int) -> int:
    # Full compiler-generated DeiT programs are much longer than the focused
    # RTL block regressions, and the RTL usually needs materially more cycles
    # than the software golden model because it models transfer and engine
    # sequencing explicitly. Use the golden cycle count to pick a realistic
    # runner budget while still keeping a hard timeout.
    derived_budget = int(golden_cycles) * 6 + 1_000_000
    return max(int(min_budget), derived_budget)


def _invoke_runner(
    runner_path: Path,
    program_path: Path,
    summary_path: Path,
    patches_raw_path: Path | None,
    patch_rows: int | None,
    patch_cols: int | None,
    cls_raw_path: Path | None,
    folded_pos_embed: bool,
    num_classes: int,
    max_cycles: int,
    trace_json_out: Path | None = None,
    snapshot_request_path: Path | None = None,
    snapshot_manifest_out: Path | None = None,
    snapshot_data_out: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    cmd = [
        str(runner_path),
        "--program",
        str(program_path),
        "--json-out",
        str(summary_path),
        "--num-classes",
        str(num_classes),
        "--max-cycles",
        str(max_cycles),
    ]
    if patches_raw_path is not None:
        cmd.extend(
            [
                "--patches-raw",
                str(patches_raw_path),
                "--patch-rows",
                str(patch_rows),
                "--patch-cols",
                str(patch_cols),
            ]
        )
    if cls_raw_path is not None:
        cmd.extend(["--cls-raw", str(cls_raw_path)])
    if folded_pos_embed:
        cmd.append("--folded-pos-embed")
    if trace_json_out is not None:
        cmd.extend(["--trace-json-out", str(trace_json_out)])
    snapshot_args = [
        snapshot_request_path,
        snapshot_manifest_out,
        snapshot_data_out,
    ]
    if any(path is not None for path in snapshot_args):
        if not all(path is not None for path in snapshot_args):
            raise ValueError("Snapshot runner arguments must be provided together")
        cmd.extend(
            [
                "--snapshot-request",
                str(snapshot_request_path),
                "--snapshot-manifest-out",
                str(snapshot_manifest_out),
                "--snapshot-data-out",
                str(snapshot_data_out),
            ]
        )

    proc = subprocess.run(cmd, check=False)
    return proc.returncode, json.loads(summary_path.read_text())


def _first_mismatch(lhs: list[int], rhs: list[int]) -> dict[str, int] | None:
    for idx, (left, right) in enumerate(zip(lhs, rhs)):
        if left != right:
            return {"index": idx, "golden": left, "rtl": right}
    if len(lhs) != len(rhs):
        return {"index": min(len(lhs), len(rhs)), "golden": len(lhs), "rtl": len(rhs)}
    return None


def _iter_trace_events(program: ProgramBinary) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for pc in sorted(program.trace_manifest):
        for event_index, event in enumerate(program.trace_manifest[pc]):
            record = dict(event)
            record["pc"] = int(pc)
            record["event_index"] = int(event_index)
            record.setdefault("source", "architectural")
            events.append(record)
    return events


def _trace_node_order_from_program(program: ProgramBinary) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for event in _iter_trace_events(program):
        node_name = str(event["node_name"])
        if node_name not in seen:
            seen.add(node_name)
            order.append(node_name)
    return order


def _format_snapshot_field(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _write_snapshot_request(path: Path, trace_events: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for event in trace_events:
        if event.get("source") == "virtual":
            continue
        fields = [
            int(event["pc"]),
            int(event["event_index"]),
            str(event["node_name"]),
            int(event["buf_id"]),
            int(event["offset_units"]),
            int(event["mem_rows"]),
            int(event["mem_cols"]),
            int(event["logical_rows"]),
            int(event["logical_cols"]),
            int(event["full_rows"]),
            int(event["full_cols"]),
            int(event["row_start"]),
            str(event["dtype"]),
            float(event["scale"]),
            str(event.get("source", "architectural")),
        ]
        for field in fields:
            if isinstance(field, str) and ("," in field or "\n" in field or "\r" in field):
                raise ValueError(f"Snapshot request field contains unsupported CSV characters: {field!r}")
        lines.append(",".join(_format_snapshot_field(field) for field in fields))
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text)


def _event_key(event: dict[str, Any]) -> tuple[int, int, str]:
    return (int(event["pc"]), int(event["event_index"]), str(event["node_name"]))


def _decode_raw_payload(dtype: str, raw: bytes, logical_rows: int, logical_cols: int) -> np.ndarray:
    expected_elems = int(logical_rows) * int(logical_cols)
    if dtype == "int8":
        arr = np.frombuffer(raw, dtype=np.int8)
    elif dtype == "int32":
        arr = np.frombuffer(raw, dtype="<i4")
    else:
        raise ValueError(f"Unsupported traced dtype: {dtype}")

    if arr.size != expected_elems:
        raise ValueError(
            f"Snapshot payload size mismatch for {dtype}: expected {expected_elems} elems, got {arr.size}"
        )
    return arr.reshape(int(logical_rows), int(logical_cols)).copy()


def _load_snapshot_bundle(
    manifest_path: Path,
    data_path: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, str], dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    entries = list(manifest.get("entries", []))
    blob = data_path.read_bytes()
    capture_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    for entry in entries:
        key = _event_key(entry)
        record = dict(entry)
        status = str(record.get("status", "missing_snapshot"))
        if status == "captured":
            byte_offset = int(record["byte_offset"])
            byte_size = int(record["byte_size"])
            raw = blob[byte_offset: byte_offset + byte_size]
            if len(raw) != byte_size:
                raise ValueError(
                    f"Snapshot data truncated for {record['node_name']} event {record['event_index']}"
                )
            record["raw"] = _decode_raw_payload(
                dtype=str(record["dtype"]),
                raw=raw,
                logical_rows=int(record["logical_rows"]),
                logical_cols=int(record["logical_cols"]),
            )
        capture_map[key] = record
    return entries, capture_map


def _load_golden_raw_event_map(trace_payload: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    raw_events = list((trace_payload or {}).get("raw_events", []))
    event_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    for event in raw_events:
        record = dict(event)
        if record.get("raw_available"):
            raw = np.asarray(record.get("raw", []))
            dtype = str(record["dtype"])
            if dtype == "int8":
                raw = raw.astype(np.int8, copy=False)
            elif dtype == "int32":
                raw = raw.astype(np.int32, copy=False)
            else:
                raise ValueError(f"Unsupported golden trace dtype: {dtype}")
            record["raw"] = raw.reshape(int(record["logical_rows"]), int(record["logical_cols"])).copy()
        event_map[_event_key(record)] = record
    return event_map


def _build_node_specs(
    program: ProgramBinary,
) -> tuple[list[str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    node_order: list[str] = []
    node_specs: dict[str, dict[str, Any]] = {}
    skipped_virtual: list[dict[str, Any]] = []
    for event in _iter_trace_events(program):
        source = str(event.get("source", "architectural"))
        node_name = str(event["node_name"])
        if source == "virtual":
            skipped_virtual.append(
                {
                    "node_name": node_name,
                    "pc": int(event["pc"]),
                    "event_index": int(event["event_index"]),
                    "dtype": str(event["dtype"]),
                    "scale": float(event["scale"]),
                    "full_rows": int(event["full_rows"]),
                    "full_cols": int(event["full_cols"]),
                    "reason": "virtual_skipped",
                }
            )
            continue
        if node_name not in node_specs:
            node_order.append(node_name)
            node_specs[node_name] = {
                "dtype": str(event["dtype"]),
                "scale": float(event["scale"]),
                "full_rows": int(event["full_rows"]),
                "full_cols": int(event["full_cols"]),
                "events": [],
            }
        node_specs[node_name]["events"].append(event)
    return node_order, node_specs, skipped_virtual


def _assemble_node_tensor(
    node_spec: dict[str, Any],
    actual_event_map: dict[tuple[int, int, str], dict[str, Any]],
    *,
    require_raw: bool,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    dtype_name = str(node_spec["dtype"])
    if dtype_name == "int8":
        dtype = np.int8
    elif dtype_name == "int32":
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported node dtype: {dtype_name}")

    tensor = np.zeros(
        (int(node_spec["full_rows"]), int(node_spec["full_cols"])),
        dtype=dtype,
    )
    for expected in node_spec["events"]:
        actual = actual_event_map.get(_event_key(expected))
        if actual is None:
            return None, {
                "kind": "missing_snapshot",
                "event": expected,
                "status": "missing_entry",
            }
        if require_raw:
            status = str(actual.get("status", "missing_snapshot"))
            if status != "captured":
                return None, {
                    "kind": "missing_snapshot",
                    "event": expected,
                    "status": status,
                }
        else:
            if not actual.get("raw_available", False):
                return None, {
                    "kind": "missing_snapshot",
                    "event": expected,
                    "status": "raw_unavailable",
                }

        raw = actual.get("raw")
        if raw is None:
            return None, {
                "kind": "missing_snapshot",
                "event": expected,
                "status": "raw_missing",
            }
        logical_rows = int(expected["logical_rows"])
        logical_cols = int(expected["logical_cols"])
        row_start = int(expected["row_start"])
        tensor[row_start: row_start + logical_rows, :logical_cols] = raw
    return tensor, None


def _find_event_for_flat_index(node_spec: dict[str, Any], flat_index: int) -> dict[str, Any]:
    full_cols = int(node_spec["full_cols"])
    row_idx = flat_index // full_cols
    col_idx = flat_index % full_cols
    for event in node_spec["events"]:
        row_start = int(event["row_start"])
        logical_rows = int(event["logical_rows"])
        logical_cols = int(event["logical_cols"])
        if row_start <= row_idx < row_start + logical_rows and col_idx < logical_cols:
            return event
    return node_spec["events"][0]


def _dequantized_metrics(golden_raw: np.ndarray, rtl_raw: np.ndarray, scale: float) -> dict[str, float]:
    golden_f = golden_raw.astype(np.float32) * np.float32(scale)
    rtl_f = rtl_raw.astype(np.float32) * np.float32(scale)
    diff = np.abs(golden_f - rtl_f)
    golden_flat = golden_f.reshape(-1)
    rtl_flat = rtl_f.reshape(-1)
    if golden_flat.size == 0:
        cosine = 1.0
    else:
        golden_norm = float(np.linalg.norm(golden_flat))
        rtl_norm = float(np.linalg.norm(rtl_flat))
        if golden_norm == 0.0 and rtl_norm == 0.0:
            cosine = 1.0
        elif golden_norm == 0.0 or rtl_norm == 0.0:
            cosine = 0.0
        else:
            cosine = float(np.dot(golden_flat, rtl_flat) / (golden_norm * rtl_norm))
    return {
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "cosine_similarity": cosine,
    }


def _compute_first_divergence(
    program: ProgramBinary,
    golden_trace: dict[str, Any],
    snapshot_manifest_path: Path,
    snapshot_data_path: Path,
    *,
    artifact_paths: dict[str, Any],
) -> dict[str, Any] | None:
    node_order, node_specs, skipped_virtual = _build_node_specs(program)
    _, snapshot_map = _load_snapshot_bundle(snapshot_manifest_path, snapshot_data_path)
    golden_map = _load_golden_raw_event_map(golden_trace)

    for node_name in node_order:
        node_spec = node_specs[node_name]
        golden_tensor, golden_error = _assemble_node_tensor(node_spec, golden_map, require_raw=False)
        if golden_error is not None:
            raise RuntimeError(
                f"Golden trace is missing architectural raw data for {node_name}: {golden_error['status']}"
            )

        rtl_tensor, rtl_error = _assemble_node_tensor(node_spec, snapshot_map, require_raw=True)
        if rtl_error is not None:
            event = dict(rtl_error["event"])
            return {
                "node_name": node_name,
                "trace_pc": int(event["pc"]),
                "event_index": int(event["event_index"]),
                "node_metadata": {
                    "dtype": str(node_spec["dtype"]),
                    "scale": float(node_spec["scale"]),
                    "full_rows": int(node_spec["full_rows"]),
                    "full_cols": int(node_spec["full_cols"]),
                    "fragment_count": int(len(node_spec["events"])),
                },
                "mismatch_kind": str(rtl_error["kind"]),
                "missing_status": str(rtl_error["status"]),
                "first_differing_element_index": None,
                "raw_values": {"golden": None, "rtl": None},
                "dequantized_summary": None,
                "artifacts": dict(artifact_paths),
                "skipped_virtual_nodes": skipped_virtual,
            }

        if golden_tensor.shape != rtl_tensor.shape:
            event = dict(node_spec["events"][0])
            return {
                "node_name": node_name,
                "trace_pc": int(event["pc"]),
                "event_index": int(event["event_index"]),
                "node_metadata": {
                    "dtype": str(node_spec["dtype"]),
                    "scale": float(node_spec["scale"]),
                    "full_rows": int(node_spec["full_rows"]),
                    "full_cols": int(node_spec["full_cols"]),
                    "fragment_count": int(len(node_spec["events"])),
                    "golden_shape": list(golden_tensor.shape),
                    "rtl_shape": list(rtl_tensor.shape),
                },
                "mismatch_kind": "shape_mismatch",
                "first_differing_element_index": None,
                "raw_values": {"golden": None, "rtl": None},
                "dequantized_summary": None,
                "artifacts": dict(artifact_paths),
                "skipped_virtual_nodes": skipped_virtual,
            }

        flat_golden = golden_tensor.reshape(-1)
        flat_rtl = rtl_tensor.reshape(-1)
        mismatch_idx = np.flatnonzero(flat_golden != flat_rtl)
        if mismatch_idx.size:
            first_idx = int(mismatch_idx[0])
            event = _find_event_for_flat_index(node_spec, first_idx)
            full_cols = int(node_spec["full_cols"])
            return {
                "node_name": node_name,
                "trace_pc": int(event["pc"]),
                "event_index": int(event["event_index"]),
                "node_metadata": {
                    "dtype": str(node_spec["dtype"]),
                    "scale": float(node_spec["scale"]),
                    "full_rows": int(node_spec["full_rows"]),
                    "full_cols": int(node_spec["full_cols"]),
                    "fragment_count": int(len(node_spec["events"])),
                },
                "mismatch_kind": "raw_value_mismatch",
                "first_differing_element_index": first_idx,
                "first_differing_row": int(first_idx // full_cols),
                "first_differing_col": int(first_idx % full_cols),
                "raw_values": {
                    "golden": int(flat_golden[first_idx]),
                    "rtl": int(flat_rtl[first_idx]),
                },
                "dequantized_summary": _dequantized_metrics(
                    golden_tensor,
                    rtl_tensor,
                    float(node_spec["scale"]),
                ),
                "artifacts": dict(artifact_paths),
                "skipped_virtual_nodes": skipped_virtual,
            }

    return None


def _maybe_emit_first_divergence(
    *,
    program: ProgramBinary,
    golden_trace: dict[str, Any] | None,
    runner_path: Path,
    program_path: Path,
    rtl_summary_path: Path,
    rtl_trace_path: Path,
    patches_raw_path: Path | None,
    patch_rows: int | None,
    patch_cols: int | None,
    cls_raw_path: Path | None,
    folded_pos_embed: bool,
    num_classes: int,
    max_cycles: int,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
    trace_events = _iter_trace_events(program)
    if golden_trace is None or not trace_events:
        runner_rc, rtl = _invoke_runner(
            runner_path=runner_path,
            program_path=program_path,
            summary_path=rtl_summary_path,
            patches_raw_path=patches_raw_path,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            cls_raw_path=cls_raw_path,
            folded_pos_embed=folded_pos_embed,
            num_classes=num_classes,
            max_cycles=max_cycles,
            trace_json_out=rtl_trace_path,
        )
        artifacts["rtl_trace"] = str(rtl_trace_path)
        return runner_rc, rtl, None

    snapshot_request_path = work_dir / "snapshot_request.csv"
    snapshot_manifest_path = work_dir / "rtl_snapshot_manifest.json"
    snapshot_data_path = work_dir / "rtl_snapshot_data.bin"
    first_divergence_path = work_dir / "first_divergence.json"
    _write_snapshot_request(snapshot_request_path, trace_events)

    runner_rc, rtl = _invoke_runner(
        runner_path=runner_path,
        program_path=program_path,
        summary_path=rtl_summary_path,
        patches_raw_path=patches_raw_path,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        cls_raw_path=cls_raw_path,
        folded_pos_embed=folded_pos_embed,
        num_classes=num_classes,
        max_cycles=max_cycles,
        trace_json_out=rtl_trace_path,
        snapshot_request_path=snapshot_request_path,
        snapshot_manifest_out=snapshot_manifest_path,
        snapshot_data_out=snapshot_data_path,
    )

    artifacts["rtl_trace"] = str(rtl_trace_path)
    artifacts["snapshot_request"] = str(snapshot_request_path)
    artifacts["snapshot_manifest"] = str(snapshot_manifest_path)
    artifacts["snapshot_data"] = str(snapshot_data_path)

    divergence = _compute_first_divergence(
        program=program,
        golden_trace=golden_trace,
        snapshot_manifest_path=snapshot_manifest_path,
        snapshot_data_path=snapshot_data_path,
        artifact_paths={
            **artifacts,
            "golden_trace": artifacts.get("golden_trace"),
            "rtl_trace": str(rtl_trace_path),
            "snapshot_manifest": str(snapshot_manifest_path),
            "snapshot_data": str(snapshot_data_path),
        },
    )
    if divergence is not None:
        first_divergence_path.write_text(json.dumps(divergence, indent=2, sort_keys=True, default=_json_default))
        artifacts["first_divergence"] = str(first_divergence_path)
    return runner_rc, rtl, divergence


def compare_program_mode(args: argparse.Namespace) -> dict[str, Any]:
    runner_path = Path(args.runner).resolve()
    _ensure_runner_built(runner_path, rebuild=args.rebuild_runner)

    work_dir = Path(args.work_dir or tempfile.mkdtemp(prefix="compare_rtl_golden_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    program_path = Path(args.program).resolve()
    program = _load_program(program_path)

    patches = None
    cls_int8 = None
    patches_raw_path = None
    cls_raw_path = None

    if args.input is not None:
        patches = _load_raw_or_npy_int8(
            Path(args.input),
            rows=args.patch_rows,
            cols=args.patch_cols,
        )
        if patches.ndim != 2:
            raise ValueError("Program-mode input must be a 2D INT8 tensor")
        patches_raw_path = work_dir / "patches.raw"
        _write_raw_int8(patches_raw_path, patches)

    if args.cls_input is not None:
        cls_int8 = _load_raw_or_npy_int8(Path(args.cls_input))
        cls_raw_path = work_dir / "cls.raw"
        _write_raw_int8(cls_raw_path, cls_int8.reshape(-1))

    golden = _run_golden_program(
        program=program,
        patches_int8=patches,
        cls_int8=cls_int8,
        folded_pos_embed=args.folded_pos_embed,
        num_classes=args.num_classes,
    )

    rtl_summary_path = work_dir / "rtl_summary.json"
    rtl_trace_path = work_dir / "rtl_trace.json"
    golden_trace_path = work_dir / "golden_trace.json"

    runner_rc, rtl = _invoke_runner(
        runner_path=runner_path,
        program_path=program_path,
        summary_path=rtl_summary_path,
        patches_raw_path=patches_raw_path,
        patch_rows=(patches.shape[0] if patches is not None else None),
        patch_cols=(patches.shape[1] if patches is not None else None),
        cls_raw_path=cls_raw_path,
        folded_pos_embed=args.folded_pos_embed,
        num_classes=args.num_classes,
        max_cycles=args.max_cycles,
    )

    first_mismatch = _first_mismatch(golden["logits"], rtl["logits"])
    passed = (
        runner_rc == 0
        and rtl["status"] == "halted"
        and not rtl["fault"]
        and first_mismatch is None
    )

    artifacts = {
        "work_dir": str(work_dir),
        "rtl_summary": str(rtl_summary_path),
    }
    first_divergence = None

    if not passed:
        trace_nodes = _trace_node_order_from_program(program)
        rerun_golden = _run_golden_program(
            program=program,
            patches_int8=patches,
            cls_int8=cls_int8,
            folded_pos_embed=args.folded_pos_embed,
            num_classes=args.num_classes,
            trace_nodes=trace_nodes,
        )
        golden_trace_path.write_text(
            json.dumps(rerun_golden["trace"], indent=2, sort_keys=True, default=_json_default)
        )
        artifacts["golden_trace"] = str(golden_trace_path)
        runner_rc, rtl, first_divergence = _maybe_emit_first_divergence(
            program=program,
            golden_trace=rerun_golden["trace"],
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            rtl_trace_path=rtl_trace_path,
            patches_raw_path=patches_raw_path,
            patch_rows=patches.shape[0] if patches is not None else None,
            patch_cols=patches.shape[1] if patches is not None else None,
            cls_raw_path=cls_raw_path,
            folded_pos_embed=args.folded_pos_embed,
            num_classes=args.num_classes,
            max_cycles=args.max_cycles,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        first_mismatch = _first_mismatch(golden["logits"], rtl["logits"])
        passed = (
            runner_rc == 0
            and rtl["status"] == "halted"
            and not rtl["fault"]
            and first_mismatch is None
        )

    return {
        "mode": "program",
        "pass": passed,
        "runner_exit_code": runner_rc,
        "golden": golden,
        "rtl": rtl,
        "mismatch_count": 0 if first_mismatch is None else 1,
        "first_mismatch": first_mismatch,
        "program_manifest": program.compiler_manifest,
        "artifacts": artifacts,
        "first_divergence": first_divergence,
    }


def compare_compile_mode(args: argparse.Namespace) -> dict[str, Any]:
    import compare_golden as cg

    runner_path = Path(args.runner).resolve()
    _ensure_runner_built(runner_path, rebuild=args.rebuild_runner)

    work_dir = Path(args.work_dir or tempfile.mkdtemp(prefix="compare_rtl_compile_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    cg.WEIGHTS_PATH = args.weights
    model, state_dict = cg.load_model()
    processor = _load_processor(cg.MODEL_NAME)

    calibration_paths = args.calibration_image or [args.image]
    sample_images = []
    for img_path in calibration_paths:
        with Image.open(img_path) as img:
            sample_images.append(img.convert("RGB"))

    compile_kwargs = _scenario_overrides(args.scenario)
    compile_kwargs["gelu_from_accum"] = args.gelu_from_accum
    if args.requant_pc_qkv_blocks:
        compile_kwargs["requant_pc_qkv"] = True
        compile_kwargs["requant_pc_qkv_selection"] = cg.build_requant_pc_qkv_selection(
            blocks_text=args.requant_pc_qkv_blocks,
            heads_text=args.requant_pc_qkv_heads,
            projections_text=args.requant_pc_qkv_projections,
            exclude_text=args.requant_pc_qkv_exclude,
        )

    program, cal_scales = cg.compile_model(
        model,
        state_dict,
        sample_images,
        processor,
        **compile_kwargs,
    )

    embed_scale = cal_scales.get("pos_embed_add", 14.0 / 127.0)
    with Image.open(args.image) as runtime_img:
        runtime_img = runtime_img.convert("RGB")
        patches_int8, cls_int8, _ = cg.patch_embed_int8(
            model,
            processor,
            runtime_img,
            embed_scale,
            fold_cls_pos_embed=args.fold_cls_pos_embed,
        )

    program_path = work_dir / "program.bin"
    program_path.write_bytes(program.to_bytes())
    patches_raw_path = work_dir / "patches.raw"
    _write_raw_int8(patches_raw_path, patches_int8)
    cls_raw_path = None
    if cls_int8 is not None:
        cls_raw_path = work_dir / "cls.raw"
        _write_raw_int8(cls_raw_path, cls_int8.reshape(-1))

    golden_logits, insn_count, cycle_count, _ = cg.golden_inference(
        program,
        patches_int8,
        cls_int8=cls_int8,
        num_classes=args.num_classes,
    )
    golden = {
        "fault": False,
        "fault_code": 0,
        "instruction_count": int(insn_count),
        "cycle_count": int(cycle_count),
        "logits": np.asarray(golden_logits, dtype=np.int32).tolist(),
    }
    rtl_cycle_budget = _compile_mode_cycle_budget(args.max_cycles, golden["cycle_count"])

    rtl_summary_path = work_dir / "rtl_summary.json"
    rtl_trace_path = work_dir / "rtl_trace.json"
    golden_trace_path = work_dir / "golden_trace.json"
    # patch_embed_int8() always folds patch position embeddings (B3 opt);
    # pass folded_pos_embed=True so the RTL runner zeroes those DRAM rows,
    # matching golden_inference which always zeroes pos_embed_patch_dram_offset.
    runner_rc, rtl = _invoke_runner(
        runner_path=runner_path,
        program_path=program_path,
        summary_path=rtl_summary_path,
        patches_raw_path=patches_raw_path,
        patch_rows=int(patches_int8.shape[0]),
        patch_cols=int(patches_int8.shape[1]),
        cls_raw_path=cls_raw_path,
        folded_pos_embed=True,
        num_classes=args.num_classes,
        max_cycles=rtl_cycle_budget,
    )

    first_mismatch = _first_mismatch(golden["logits"], rtl["logits"])
    passed = (
        runner_rc == 0
        and rtl["status"] == "halted"
        and not rtl["fault"]
        and first_mismatch is None
    )

    artifacts = {
        "work_dir": str(work_dir),
        "program": str(program_path),
        "rtl_summary": str(rtl_summary_path),
    }
    first_divergence = None

    if not passed:
        trace_nodes = _trace_node_order_from_program(program)
        _, _, _, golden_trace = cg.golden_inference(
            program,
            patches_int8,
            cls_int8=cls_int8,
            num_classes=args.num_classes,
            trace_nodes=trace_nodes,
        )
        golden_trace_path.write_text(
            json.dumps(golden_trace, indent=2, sort_keys=True, default=_json_default)
        )
        artifacts["golden_trace"] = str(golden_trace_path)
        runner_rc, rtl, first_divergence = _maybe_emit_first_divergence(
            program=program,
            golden_trace=golden_trace,
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            rtl_trace_path=rtl_trace_path,
            patches_raw_path=patches_raw_path,
            patch_rows=int(patches_int8.shape[0]),
            patch_cols=int(patches_int8.shape[1]),
            cls_raw_path=cls_raw_path,
            folded_pos_embed=True,
            num_classes=args.num_classes,
            max_cycles=rtl_cycle_budget,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        first_mismatch = _first_mismatch(golden["logits"], rtl["logits"])
        passed = (
            runner_rc == 0
            and rtl["status"] == "halted"
            and not rtl["fault"]
            and first_mismatch is None
        )

    return {
        "mode": "compile",
        "scenario": args.scenario,
        "pass": passed,
        "runner_exit_code": runner_rc,
        "golden": golden,
        "rtl": rtl,
        "rtl_cycle_budget": rtl_cycle_budget,
        "mismatch_count": 0 if first_mismatch is None else 1,
        "first_mismatch": first_mismatch,
        "program_manifest": program.compiler_manifest,
        "artifacts": artifacts,
        "first_divergence": first_divergence,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare RTL execution against the software golden model")
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER), help="Path to the Verilator program runner")
    parser.add_argument("--work-dir", help="Directory for intermediate artifacts and summaries")
    parser.add_argument("--summary-out", required=True, help="Output JSON summary path")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of logits to compare from ACCUM")
    parser.add_argument("--max-cycles", type=int, default=500000, help="Cycle budget for the RTL runner")
    parser.add_argument("--rebuild-runner", action="store_true", help="Force a fresh Verilator runner build")

    sub = parser.add_subparsers(dest="mode", required=True)

    p_program = sub.add_parser("program", help="Compare a precompiled ProgramBinary")
    p_program.add_argument("--program", required=True, help="Path to program.bin")
    p_program.add_argument("--input", help="Program input tensor (.npy or raw .bin)")
    p_program.add_argument("--patch-rows", type=int, help="Rows for raw input .bin")
    p_program.add_argument("--patch-cols", type=int, help="Columns for raw input .bin")
    p_program.add_argument("--cls-input", help="Optional folded CLS row (.npy or raw .bin)")
    p_program.add_argument("--folded-pos-embed", action="store_true", help="Zero folded position-embedding regions")

    p_compile = sub.add_parser("compile", help="Compile a DeiT model variant, then compare RTL vs golden")
    p_compile.add_argument("--scenario", required=True, choices=[
        "baseline_default",
        "experimental_requant_pc",
        "experimental_dequant_add",
        "experimental_softmax_attnv",
        "experimental_fused_out_proj",
    ])
    p_compile.add_argument("--weights", required=True, help="Path to pytorch_model.bin")
    p_compile.add_argument("--image", required=True, help="Runtime image used for patch embedding and compare")
    p_compile.add_argument("--calibration-image", action="append", help="Optional calibration image(s)")
    p_compile.add_argument("--fold-cls-pos-embed", action="store_true", help="Fold CLS position embedding on the host")
    p_compile.add_argument("--gelu-from-accum", action="store_true", help="Enable GELU-from-ACCUM during compile")
    p_compile.add_argument("--requant-pc-qkv-blocks", help="Optional block CSV for QKV REQUANT_PC selection")
    p_compile.add_argument("--requant-pc-qkv-heads", default="", help="Optional head CSV for QKV REQUANT_PC selection")
    p_compile.add_argument("--requant-pc-qkv-projections", default="all", help="Projection filter for QKV REQUANT_PC selection")
    p_compile.add_argument("--requant-pc-qkv-exclude", default="", help="Exclude triplets for QKV REQUANT_PC selection")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "program":
        summary = compare_program_mode(args)
    else:
        summary = compare_compile_mode(args)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))

    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
