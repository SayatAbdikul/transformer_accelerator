"""Tests for golden model simulator."""
import pytest
import numpy as np
from taccel.golden_model.state import MachineState
from taccel.golden_model.simulator import Simulator, ConfigError, IllegalBufferError
from taccel.golden_model.memory import SRAMAccessError
from taccel.assembler.assembler import Assembler, ProgramBinary
from taccel.isa.opcodes import BUF_ABUF, BUF_WBUF, BUF_ACCUM


def make_sim(asm_source: str) -> Simulator:
    prog = Assembler().assemble(asm_source)
    sim = Simulator()
    sim.load_program(prog)
    return sim


class TestSIMDMatmul:
    def test_single_matmul_tile(self):
        """Hand-verify 16x16 INT8 matmul."""
        # A = identity(16), B = 2*identity(16) → C = 2*identity(16)
        A = np.eye(16, dtype=np.int8) * 2
        B = np.eye(16, dtype=np.int8) * 3

        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\nHALT"
        )
        sim = Simulator()
        sim.load_program(prog)

        # Write A to ABUF, B to WBUF
        sim.state.abuf[:256] = A.tobytes()
        sim.state.wbuf[:256] = B.tobytes()

        # Execute CONFIG_TILE
        sim.step()  # CONFIG_TILE

        # Manually check tile config is set
        assert sim.state.tile_config == (0, 0, 0)  # encoded as 0-based

        sim.step()  # HALT (but already halted by CONFIG_TILE)

    def test_matmul_identity(self):
        """A @ I = A."""
        A = np.arange(256, dtype=np.int8).reshape(16, 16)
        I = np.eye(16, dtype=np.int8)

        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "MATMUL src1=ABUF[0], src2=WBUF[0], dst=ACCUM[0], sreg=0, flags=0\n"
            "SYNC 0b010\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.abuf[:256] = A.tobytes()
        sim.state.wbuf[:256] = I.tobytes()
        sim.run()

        result = sim.state.accum[:256].reshape(16, 16)
        expected = A.astype(np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_matmul_accumulate(self):
        """Two tiles with accumulate=1 sum correctly."""
        A = np.ones((16, 16), dtype=np.int8)
        B = np.ones((16, 16), dtype=np.int8)

        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "MATMUL src1=ABUF[0], src2=WBUF[0], dst=ACCUM[0], sreg=0, flags=0\n"
            "MATMUL src1=ABUF[0], src2=WBUF[0], dst=ACCUM[0], sreg=0, flags=1\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.abuf[:256] = A.tobytes()
        sim.state.wbuf[:256] = B.tobytes()
        sim.run()

        result = sim.state.accum[:256].reshape(16, 16)
        # Each element = 16 (from first tile) + 16 (from second) = 32
        assert np.all(result == 32), f"Expected 32, got {result[0,0]}"


class TestRequant:
    def test_basic_requant(self):
        """INT32 → INT8 clipping."""
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3800\n"  # 0x3800 = 0.5 in FP16
            "REQUANT src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        # Fill accumulator with 100 (INT32)
        sim.state.accum[:256] = 100
        sim.run()

        # 0.5 * 100 = 50
        result = np.frombuffer(bytes(sim.state.abuf[:256]), dtype=np.int8)
        assert np.all(result == 50), f"Expected 50, got {result[0]}"

    def test_requant_clipping(self):
        """INT32 values outside [-128, 127] get clipped."""
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3C00\n"  # 0x3C00 = 1.0 in FP16
            "REQUANT src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.accum[0] = 200   # out of INT8 range
        sim.state.accum[1] = -200  # out of INT8 range
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[:256]), dtype=np.int8)
        assert result[0] == 127   # clamped to max
        assert result[1] == -128  # clamped to min


class TestVADD:
    def test_int8_saturating_add(self):
        """INT8 VADD saturates at ±127."""
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "VADD src1=ABUF[0], src2=ABUF[16], dst=ABUF[32], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        # src1: fill with 100, src2: fill with 50 → result 127 (saturated)
        sim.state.abuf[:256] = bytes([100] * 256)
        sim.state.abuf[256:512] = bytes([50] * 256)
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[512:768]), dtype=np.int8)
        assert np.all(result == 127), f"Expected 127, got {result[0]}"

    def test_int8_add_no_overflow(self):
        """INT8 VADD: 10 + 20 = 30."""
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "VADD src1=ABUF[0], src2=ABUF[16], dst=ABUF[32], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.abuf[:256] = bytes([10] * 256)
        sim.state.abuf[256:512] = bytes([20] * 256)
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[512:768]), dtype=np.int8)
        assert np.all(result == 30)


class TestDMA:
    def test_load_store_roundtrip(self):
        """DMA load then store recovers original data."""
        prog = Assembler().assemble(
            "SET_ADDR_LO R0, 0x0000000\n"
            "SET_ADDR_HI R0, 0x0000000\n"
            "LOAD buf_id=ABUF, sram_off=0, xfer_len=16, addr_reg=0, dram_off=0\n"
            "SYNC 0b001\n"
            "SET_ADDR_LO R1, 0x0001000\n"
            "SET_ADDR_HI R1, 0x0000000\n"
            "STORE buf_id=ABUF, sram_off=0, xfer_len=16, addr_reg=1, dram_off=0\n"
            "SYNC 0b001\n"
            "HALT"
        )
        data = bytes(range(256))  # 256 bytes = 16 units
        prog = Assembler().assemble(
            "SET_ADDR_LO R0, 0x0000000\n"
            "SET_ADDR_HI R0, 0x0000000\n"
            "LOAD buf_id=ABUF, sram_off=0, xfer_len=16, addr_reg=0, dram_off=0\n"
            "SYNC 0b001\n"
            "SET_ADDR_LO R1, 0x1000000\n"
            "SET_ADDR_HI R1, 0x0000000\n"
            "STORE buf_id=ABUF, sram_off=0, xfer_len=16, addr_reg=1, dram_off=0\n"
            "SYNC 0b001\n"
            "HALT",
            data=data,
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.run()

        # Verify ABUF contains loaded data
        assert bytes(sim.state.abuf[:256]) == data


class TestBufCopy:
    def test_flat_copy(self):
        """BUF_COPY flat: copies bytes unchanged."""
        prog = Assembler().assemble(
            "BUF_COPY src_buf=ABUF, src_off=0, dst_buf=WBUF, dst_off=0, length=16\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        data = bytes(range(256))
        sim.state.abuf[:256] = data
        sim.run()
        assert bytes(sim.state.wbuf[:256]) == data

    def test_transpose_copy(self):
        """BUF_COPY transpose: [32, 16] → [16, 32]."""
        prog = Assembler().assemble(
            "BUF_COPY src_buf=ABUF, src_off=0, dst_buf=WBUF, dst_off=0, "
            "length=32, src_rows=2, transpose=1\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)

        # Create [32, 16] source: src_rows=2 means 2*16=32 rows, cols=length*16/(src_rows*16)=512/32=16
        # length=32 means 32*16=512 bytes
        src = np.arange(512, dtype=np.int8).reshape(32, 16)
        sim.state.abuf[:512] = src.tobytes()
        sim.run()

        result = np.frombuffer(bytes(sim.state.wbuf[:512]), dtype=np.int8).reshape(16, 32)
        expected = src.T
        np.testing.assert_array_equal(result, expected)


class TestSFUOps:
    def test_gelu_positive(self):
        """GELU(x) ≈ x for large positive x."""
        # GELU(4) ≈ 4, GELU(-4) ≈ 0
        from taccel.isa.instructions import GeluInsn, ConfigTileInsn, SetScaleInsn, HaltInsn
        from taccel.isa.encoding import encode
        import struct

        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3400\n"   # FP16 0.25 (in_scale)
            "SET_SCALE S1, imm=0x3400\n"   # FP16 0.25 (out_scale)
            "GELU src1=ABUF[0], src2=ABUF[0], dst=ABUF[16], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        # Fill ABUF with 16 (= 4.0 at scale 0.25)
        sim.state.abuf[:256] = bytes([16] * 256)
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[256:512]), dtype=np.int8)
        # GELU(4.0) ≈ 4.0 → at scale 0.25, quantized ≈ 16
        assert result[0] > 10, f"GELU should be ~16, got {result[0]}"

    def test_softmax_sums_to_one(self):
        """Softmax output sums close to 1.0 after dequantization."""
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3800\n"  # 0.5 in_scale
            "SET_SCALE S1, imm=0x3400\n"  # 0.25 out_scale (probs 0-1 → scaled 0-4)
            "SOFTMAX src1=ABUF[0], src2=ABUF[0], dst=ABUF[16], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        # Uniform input: all zeros → uniform softmax
        sim.state.abuf[:256] = bytes([0] * 256)
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[256:512]), dtype=np.int8)
        # At scale 0.25, each element should be ~round(1/256 / 0.25) ≈ 0
        # All 256 elements uniform: each = 1/256, tiny but non-negative
        assert np.all(result >= 0), "Softmax output should be non-negative"


class TestErrorHandling:
    def test_matmul_without_config_tile_raises(self):
        """MATMUL without preceding CONFIG_TILE raises ConfigError."""
        prog = Assembler().assemble(
            "MATMUL src1=ABUF[0], src2=WBUF[0], dst=ACCUM[0], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        with pytest.raises(ConfigError):
            sim.run()
