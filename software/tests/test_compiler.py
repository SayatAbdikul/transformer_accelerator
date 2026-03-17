"""Tests for compiler (single-layer compile + verify)."""
import pytest
import numpy as np
from taccel.compiler.tiler import tile_matmul, pad_dim, TILE

from taccel.compiler.memory_alloc import MemoryAllocator, BufferAllocator
from taccel.isa.opcodes import ABUF_SIZE, WBUF_SIZE, ACCUM_SIZE


class TestMemoryAllocator:
    def test_basic_alloc(self):
        alloc = BufferAllocator(0, 128 * 1024)
        a = alloc.alloc("a", 1024)
        assert a.offset_units == 0
        assert a.size_bytes == 1024

    def test_sequential_alloc(self):
        alloc = BufferAllocator(0, 128 * 1024)
        a = alloc.alloc("a", 256)
        b = alloc.alloc("b", 512)
        assert b.offset_units == a.offset_units + a.size_units

    def test_alloc_exceeds_capacity(self):
        alloc = BufferAllocator(0, 256)
        with pytest.raises(MemoryError):
            alloc.alloc("too_big", 512)

    def test_free_and_reuse(self):
        alloc = BufferAllocator(0, 128 * 1024)
        a = alloc.alloc("a", 256)
        b = alloc.alloc("b", 256)
        a_offset = a.offset_units
        alloc.free("a")
        # b stays at its original offset (no compaction)
        b_check = alloc.get("b")
        assert b_check.offset_units == b.offset_units
        # freed space is reusable
        c = alloc.alloc("c", 256)
        assert c.offset_units == a_offset

    def test_high_water_mark(self):
        alloc = BufferAllocator(0, 128 * 1024)
        alloc.alloc("a", 1024 * 16)
        alloc.alloc("b", 1024 * 16)
        hw = alloc.high_water_units * 16
        assert hw >= 2048 * 16


class TestTiledLinearLayer:
    def test_linear_tile_schedule(self):
        """Single linear layer [197, 192] @ [192, 192] tile schedule."""
        M, N, K = 197, 192, 192
        sched = tile_matmul(M, N, K)

        # Should have 13 * 12 * 12 = 1872 tile ops
        assert sched.m_tiles == 13
        assert sched.n_tiles == 12
        assert sched.k_tiles == 12
        assert len(sched.ops) == 13 * 12 * 12

    def test_linear_output_fits_abuf(self):
        """Standard linear output [208, 192] = 39,936 bytes << 128KB."""
        M_pad = pad_dim(197)
        N_pad = pad_dim(192)
        output_bytes = M_pad * N_pad
        assert output_bytes < ABUF_SIZE, f"{output_bytes} >= {ABUF_SIZE}"

    def test_fc1_output_exceeds_abuf(self):
        """FC1 output [208, 768] = 159,744 bytes > 128KB ABUF, needs strip-mining."""
        M_pad = pad_dim(197)
        N_pad = pad_dim(768)
        output_bytes = M_pad * N_pad
        assert output_bytes > ABUF_SIZE, f"Expected overflow, got {output_bytes}"

    def test_fc1_weights_fit_wbuf(self):
        """FC1 weights [768, 192] = 147,456 bytes < 256KB WBUF."""
        weight_bytes = pad_dim(768) * pad_dim(192)
        assert weight_bytes < WBUF_SIZE

    def test_fc2_weights_fit_wbuf(self):
        """FC2 weights [192, 768] = 147,456 bytes < 256KB WBUF."""
        weight_bytes = pad_dim(192) * pad_dim(768)
        assert weight_bytes < WBUF_SIZE


class TestMemoryconsistency:
    def test_abuf_max_tiles(self):
        """Verify ABUF can hold [208, 192] INT8 activations."""
        M_pad = pad_dim(197)
        N_pad = pad_dim(192)
        bytes_needed = M_pad * N_pad  # 39,936
        assert bytes_needed < ABUF_SIZE

    def test_wbuf_weight_capacity(self):
        """Verify WBUF can hold [192, 192] INT8 weight tile."""
        bytes_needed = 192 * 192  # 36,864
        assert bytes_needed < WBUF_SIZE

    def test_accum_capacity(self):
        """ACCUM is 64KB. Full [208, 192] INT32 doesn't fit — strip mining is required.
        One strip [16, 192] = 12,288 bytes does fit.
        """
        # Full tile doesn't fit
        M_pad = pad_dim(197)
        N_pad = pad_dim(192)
        full_bytes = M_pad * N_pad * 4
        assert full_bytes > ACCUM_SIZE, "Expected full tile to exceed ACCUM (strip-mining needed)"

        # One strip of 16 rows fits
        strip_bytes = TILE * N_pad * 4  # [16, 192] INT32
        assert strip_bytes <= ACCUM_SIZE, f"One strip {strip_bytes}B > ACCUM {ACCUM_SIZE}B"

    def test_attention_kt_wbuf_pressure(self):
        """3 heads' K^T fits in WBUF alongside other data."""
        head_dim = 64
        M_pad = pad_dim(197)
        kt_per_head = head_dim * M_pad  # 64 * 208 = 13,312 bytes
        kt_total_3_heads = kt_per_head * 3  # 39,936 bytes
        assert kt_total_3_heads < WBUF_SIZE, \
            f"3 heads K^T ({kt_total_3_heads}B) doesn't fit in WBUF ({WBUF_SIZE}B)"
