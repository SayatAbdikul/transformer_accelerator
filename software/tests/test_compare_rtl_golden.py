import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from taccel.assembler.assembler import Assembler
from taccel.isa.opcodes import BUF_ABUF, BUF_ACCUM


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "rtl" / "verilator" / "build" / "run_program" / "Vtaccel_top"
COMPARE_SCRIPT = REPO_ROOT / "software" / "tools" / "compare_rtl_golden.py"
COMPARE_SPEC = importlib.util.spec_from_file_location("compare_rtl_golden", COMPARE_SCRIPT)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
compare_rtl_golden = importlib.util.module_from_spec(COMPARE_SPEC)
COMPARE_SPEC.loader.exec_module(compare_rtl_golden)


@pytest.fixture(scope="module", autouse=True)
def build_runner():
    subprocess.run(
        ["make", "-C", str(REPO_ROOT / "rtl" / "verilator"), "run_program"],
        check=True,
    )


def _write_program(tmp_path: Path, source: str) -> Path:
    program = Assembler().assemble(source)
    path = tmp_path / "program.bin"
    path.write_bytes(program.to_bytes())
    return path


def _run_runner(program_path: Path, summary_path: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [
        str(RUNNER),
        "--program",
        str(program_path),
        "--json-out",
        str(summary_path),
        "--num-classes",
        "8",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, check=False)


def test_runner_halt_summary(tmp_path: Path):
    program_path = _write_program(tmp_path, "NOP\nHALT\n")
    summary_path = tmp_path / "summary.json"

    proc = _run_runner(program_path, summary_path)
    summary = json.loads(summary_path.read_text())

    assert proc.returncode == 0
    assert summary["status"] == "halted"
    assert summary["done"] is True
    assert summary["fault"] is False
    assert summary["retired_instructions"] == 2
    assert summary["violations"] == []


def test_runner_truncated_program_reports_parse_error(tmp_path: Path):
    program_path = tmp_path / "truncated.bin"
    program_path.write_bytes(b"TAC")
    summary_path = tmp_path / "summary.json"

    proc = _run_runner(program_path, summary_path)
    summary = json.loads(summary_path.read_text())

    assert proc.returncode == 2
    assert summary["status"] == "parse_error"
    assert summary["parse_error"] is not None


def test_runner_timeout_is_reported(tmp_path: Path):
    program_path = _write_program(tmp_path, "NOP\nHALT\n")
    summary_path = tmp_path / "summary.json"

    proc = _run_runner(program_path, summary_path, ["--max-cycles", "0"])
    summary = json.loads(summary_path.read_text())

    assert proc.returncode == 3
    assert summary["status"] == "timeout"
    assert summary["timeout"] is True
    assert "cycle_budget_exhausted" in summary["violations"]


def test_runner_dma_fault_captures_fault_context(tmp_path: Path):
    program_path = _write_program(
        tmp_path,
        "\n".join(
            [
                "SET_ADDR_LO R0, 0x0001000",
                "SET_ADDR_HI R0, 0x0000000",
                "LOAD buf_id=ABUF, sram_off=0, xfer_len=1, addr_reg=0, dram_off=0",
                "SYNC 0b001",
                "HALT",
            ]
        ) + "\n",
    )
    summary_path = tmp_path / "summary.json"

    proc = _run_runner(program_path, summary_path, ["--inject-rresp-addr", "0x1000:2"])
    summary = json.loads(summary_path.read_text())

    assert proc.returncode == 0
    assert summary["status"] == "fault"
    assert summary["fault"] is True
    assert summary["fault_code"] == 2
    assert summary["fault_context"]["valid"] is True
    assert summary["fault_context"]["source_name"] == "dma"
    assert summary["fault_context"]["opcode"] == 7


def test_compare_program_mode_smoke_passes(tmp_path: Path):
    program_path = _write_program(tmp_path, "NOP\nHALT\n")
    summary_path = tmp_path / "compare_summary.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--summary-out",
            str(summary_path),
            "--runner",
            str(RUNNER),
            "--num-classes",
            "8",
            "program",
            "--program",
            str(program_path),
        ],
        check=False,
    )
    summary = json.loads(summary_path.read_text())

    assert proc.returncode == 0
    assert summary["pass"] is True
    assert summary["runner_exit_code"] == 0
    assert summary["mismatch_count"] == 0


def test_runner_snapshot_captures_int8_rows(tmp_path: Path):
    program = Assembler().assemble(
        "\n".join(
                [
                "SET_ADDR_LO R0, 0x0000030",
                "SET_ADDR_HI R0, 0x0000000",
                "LOAD buf_id=ABUF, sram_off=0, xfer_len=1, addr_reg=0, dram_off=0",
                "SYNC 0b001",
                "HALT",
            ]
        ) + "\n",
        data=bytes(range(16)),
    )
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())
    summary_path = tmp_path / "summary.json"
    snapshot_request = tmp_path / "snapshot_request.csv"
    snapshot_manifest = tmp_path / "snapshot_manifest.json"
    snapshot_data = tmp_path / "snapshot_data.bin"
    snapshot_request.write_text(
        "3,0,trace_abuf,0,0,1,1,1,16,1,16,0,int8,1,architectural\n"
    )

    proc = _run_runner(
        program_path,
        summary_path,
        [
            "--snapshot-request",
            str(snapshot_request),
            "--snapshot-manifest-out",
            str(snapshot_manifest),
            "--snapshot-data-out",
            str(snapshot_data),
        ],
    )
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(snapshot_manifest.read_text())
    entry = manifest["entries"][0]
    payload = snapshot_data.read_bytes()[entry["byte_offset"]: entry["byte_offset"] + entry["byte_size"]]

    assert proc.returncode == 0
    assert summary["status"] == "halted"
    assert entry["status"] == "captured"
    assert payload == bytes(range(16))


def test_runner_snapshot_captures_int32_rows(tmp_path: Path):
    accum_values = np.array([10, -20, 30, -40], dtype=np.int32)
    program = Assembler().assemble(
        "\n".join(
                [
                "SET_ADDR_LO R0, 0x0000030",
                "SET_ADDR_HI R0, 0x0000000",
                "LOAD buf_id=ACCUM, sram_off=0, xfer_len=1, addr_reg=0, dram_off=0",
                "SYNC 0b001",
                "HALT",
            ]
        ) + "\n",
        data=accum_values.astype("<i4").tobytes(),
    )
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())
    summary_path = tmp_path / "summary.json"
    snapshot_request = tmp_path / "snapshot_request.csv"
    snapshot_manifest = tmp_path / "snapshot_manifest.json"
    snapshot_data = tmp_path / "snapshot_data.bin"
    snapshot_request.write_text(
        "3,0,trace_accum,2,0,1,1,1,4,1,4,0,int32,1,architectural\n"
    )

    proc = _run_runner(
        program_path,
        summary_path,
        [
            "--snapshot-request",
            str(snapshot_request),
            "--snapshot-manifest-out",
            str(snapshot_manifest),
            "--snapshot-data-out",
            str(snapshot_data),
        ],
    )
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(snapshot_manifest.read_text())
    entry = manifest["entries"][0]
    payload = snapshot_data.read_bytes()[entry["byte_offset"]: entry["byte_offset"] + entry["byte_size"]]
    observed = np.frombuffer(payload, dtype="<i4").astype(np.int32)

    assert proc.returncode == 0
    assert summary["status"] == "halted"
    assert entry["status"] == "captured"
    np.testing.assert_array_equal(observed, accum_values)


def test_runner_snapshot_virtual_entries_are_skipped(tmp_path: Path):
    program_path = _write_program(tmp_path, "NOP\nHALT\n")
    summary_path = tmp_path / "summary.json"
    snapshot_request = tmp_path / "snapshot_request.csv"
    snapshot_manifest = tmp_path / "snapshot_manifest.json"
    snapshot_data = tmp_path / "snapshot_data.bin"
    snapshot_request.write_text(
        "0,0,virtual_node,0,0,1,1,1,4,1,4,0,int8,1,virtual\n"
    )

    proc = _run_runner(
        program_path,
        summary_path,
        [
            "--snapshot-request",
            str(snapshot_request),
            "--snapshot-manifest-out",
            str(snapshot_manifest),
            "--snapshot-data-out",
            str(snapshot_data),
        ],
    )
    manifest = json.loads(snapshot_manifest.read_text())
    entry = manifest["entries"][0]

    assert proc.returncode == 0
    assert entry["status"] == "skipped_virtual"
    assert snapshot_data.read_bytes() == b""


def _make_trace_program() -> object:
    program = Assembler().assemble("NOP\nHALT\n")
    program.trace_manifest = {
        0: [
            {
                "node_name": "node_a",
                "buf_id": BUF_ABUF,
                "offset_units": 0,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 2,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int8",
                "scale": 0.5,
                "when": "after",
            },
            {
                "node_name": "node_a",
                "buf_id": BUF_ABUF,
                "offset_units": 1,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 2,
                "full_cols": 4,
                "row_start": 1,
                "dtype": "int8",
                "scale": 0.5,
                "when": "after",
            },
            {
                "node_name": "virtual_node",
                "buf_id": BUF_ABUF,
                "offset_units": 0,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int8",
                "scale": 1.0,
                "when": "after",
                "source": "virtual",
            },
        ]
    }
    return program


def _make_golden_trace() -> dict:
    return {
        "raw_events": [
            {
                "pc": 0,
                "event_index": 0,
                "node_name": "node_a",
                "dtype": "int8",
                "scale": 0.5,
                "source": "architectural",
                "row_start": 0,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 2,
                "full_cols": 4,
                "raw_available": True,
                "raw": [1, 2, 3, 4],
            },
            {
                "pc": 0,
                "event_index": 1,
                "node_name": "node_a",
                "dtype": "int8",
                "scale": 0.5,
                "source": "architectural",
                "row_start": 1,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 2,
                "full_cols": 4,
                "raw_available": True,
                "raw": [5, 6, 7, 8],
            },
            {
                "pc": 0,
                "event_index": 2,
                "node_name": "virtual_node",
                "dtype": "int8",
                "scale": 1.0,
                "source": "virtual",
                "row_start": 0,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "raw_available": False,
            },
        ]
    }


def _write_snapshot_bundle(tmp_path: Path, *, second_fragment: bytes) -> tuple[Path, Path]:
    manifest_path = tmp_path / "snapshot_manifest.json"
    data_path = tmp_path / "snapshot_data.bin"
    first = bytes([1, 2, 3, 4])
    data_path.write_bytes(first + second_fragment)
    manifest = {
        "entries": [
            {
                "pc": 0,
                "event_index": 0,
                "node_name": "node_a",
                "buf_id": BUF_ABUF,
                "offset_units": 0,
                "mem_rows": 1,
                "mem_cols": 1,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 2,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int8",
                "scale": 0.5,
                "source": "architectural",
                "status": "captured",
                "cycle": 1,
                "byte_offset": 0,
                "byte_size": 4,
            },
            {
                "pc": 0,
                "event_index": 1,
                "node_name": "node_a",
                "buf_id": BUF_ABUF,
                "offset_units": 1,
                "mem_rows": 1,
                "mem_cols": 1,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 2,
                "full_cols": 4,
                "row_start": 1,
                "dtype": "int8",
                "scale": 0.5,
                "source": "architectural",
                "status": "captured",
                "cycle": 1,
                "byte_offset": 4,
                "byte_size": 4,
            },
        ]
    }
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, data_path


def test_compute_first_divergence_returns_none_for_exact_match(tmp_path: Path):
    program = _make_trace_program()
    golden_trace = _make_golden_trace()
    manifest_path, data_path = _write_snapshot_bundle(tmp_path, second_fragment=bytes([5, 6, 7, 8]))

    divergence = compare_rtl_golden._compute_first_divergence(
        program=program,
        golden_trace=golden_trace,
        snapshot_manifest_path=manifest_path,
        snapshot_data_path=data_path,
        artifact_paths={"golden_trace": "golden.json", "rtl_trace": "rtl.json"},
    )

    assert divergence is None


def test_compute_first_divergence_reports_first_mismatching_fragment(tmp_path: Path):
    program = _make_trace_program()
    golden_trace = _make_golden_trace()
    manifest_path, data_path = _write_snapshot_bundle(tmp_path, second_fragment=bytes([5, 6, 7, 9]))

    divergence = compare_rtl_golden._compute_first_divergence(
        program=program,
        golden_trace=golden_trace,
        snapshot_manifest_path=manifest_path,
        snapshot_data_path=data_path,
        artifact_paths={"golden_trace": "golden.json", "rtl_trace": "rtl.json"},
    )

    assert divergence is not None
    assert divergence["node_name"] == "node_a"
    assert divergence["mismatch_kind"] == "raw_value_mismatch"
    assert divergence["trace_pc"] == 0
    assert divergence["event_index"] == 1
    assert divergence["first_differing_element_index"] == 7
    assert divergence["raw_values"] == {"golden": 8, "rtl": 9}
    assert divergence["skipped_virtual_nodes"][0]["node_name"] == "virtual_node"


def test_compare_program_mode_writes_first_divergence_artifact_on_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    program = Assembler().assemble("NOP\nHALT\n")
    program.trace_manifest = {
        0: [
            {
                    "node_name": "trace_abuf",
                    "buf_id": BUF_ABUF,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int8",
                "scale": 1.0,
                "when": "after",
            }
        ]
    }
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())
    work_dir = tmp_path / "work"
    summary_out = tmp_path / "summary.json"

    monkeypatch.setattr(compare_rtl_golden, "_ensure_runner_built", lambda *args, **kwargs: None)

    def fake_invoke_runner(
        runner_path,
        program_path,
        summary_path,
        patches_raw_path,
        patch_rows,
        patch_cols,
        cls_raw_path,
        folded_pos_embed,
        num_classes,
        max_cycles,
        trace_json_out=None,
        snapshot_request_path=None,
        snapshot_manifest_out=None,
        snapshot_data_out=None,
    ):
        summary = {
            "status": "halted",
            "done": True,
            "fault": False,
            "fault_code": 0,
            "logits": [1, 0, 0, 0, 0, 0, 0, 0],
        }
        Path(summary_path).write_text(json.dumps(summary))
        if trace_json_out is not None:
            Path(trace_json_out).write_text(json.dumps({"events": [{"pc": 0, "opcode": 0}]}))
        if snapshot_manifest_out is not None and snapshot_data_out is not None:
            Path(snapshot_manifest_out).write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "pc": 0,
                                "event_index": 0,
                                "node_name": "trace_abuf",
                                "buf_id": BUF_ABUF,
                                "offset_units": 0,
                                "mem_rows": 1,
                                "mem_cols": 1,
                                "logical_rows": 1,
                                "logical_cols": 4,
                                "full_rows": 1,
                                "full_cols": 4,
                                "row_start": 0,
                                "dtype": "int8",
                                "scale": 1.0,
                                "source": "architectural",
                                "status": "captured",
                                "cycle": 1,
                                "byte_offset": 0,
                                "byte_size": 4,
                            }
                        ]
                    }
                )
            )
            Path(snapshot_data_out).write_bytes(bytes([1, 0, 0, 0]))
        return 0, summary

    monkeypatch.setattr(compare_rtl_golden, "_invoke_runner", fake_invoke_runner)

    args = compare_rtl_golden.build_parser().parse_args(
        [
            "--summary-out",
            str(summary_out),
            "--runner",
            str(RUNNER),
            "--work-dir",
            str(work_dir),
            "--num-classes",
            "8",
            "program",
            "--program",
            str(program_path),
        ]
    )
    summary = compare_rtl_golden.compare_program_mode(args)

    assert summary["pass"] is False
    assert summary["first_divergence"]["node_name"] == "trace_abuf"
    assert Path(summary["artifacts"]["first_divergence"]).exists()
    assert Path(summary["artifacts"]["snapshot_manifest"]).exists()
    assert Path(summary["artifacts"]["snapshot_data"]).exists()
