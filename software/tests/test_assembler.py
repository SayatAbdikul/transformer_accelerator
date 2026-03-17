"""Tests for assembler/disassembler."""
import pytest
from taccel.assembler import Assembler, Disassembler
from taccel.assembler.assembler import ProgramBinary, MAGIC, VERSION
from taccel.isa.encoding import decode


SAMPLE_ASM = """\
; Sample program
NOP
SYNC 0b001
CONFIG_TILE M=13, N=13, K=4
SET_SCALE S0, imm=0x3000
SET_ADDR_LO R0, 0x0000000
LOAD buf_id=WBUF, sram_off=0, xfer_len=9216, addr_reg=0, dram_off=0
MATMUL src1=ABUF[0], src2=WBUF[0], dst=ACCUM[0], sreg=0, flags=0
SYNC 0b010
REQUANT src1=ACCUM[0], src2=ABUF[0], dst=ABUF[512], sreg=0, flags=0
HALT
"""


class TestAssembler:
    def setup_method(self):
        self.asm = Assembler()
        self.disasm = Disassembler()

    def test_basic_assemble(self):
        prog = self.asm.assemble(SAMPLE_ASM)
        assert prog.insn_count == 10
        assert len(prog.instructions) == 80  # 10 * 8 bytes

    def test_program_binary_magic(self):
        prog = self.asm.assemble(SAMPLE_ASM)
        raw = prog.to_bytes()
        import struct
        magic = struct.unpack(">I", raw[:4])[0]
        assert magic == MAGIC

    def test_program_binary_version(self):
        prog = self.asm.assemble(SAMPLE_ASM)
        raw = prog.to_bytes()
        import struct
        version = struct.unpack(">H", raw[4:6])[0]
        assert version == VERSION

    def test_program_binary_roundtrip(self):
        prog = self.asm.assemble(SAMPLE_ASM)
        raw = prog.to_bytes()
        prog2 = ProgramBinary.from_bytes(raw)
        assert prog2.insn_count == prog.insn_count
        assert prog2.instructions == prog.instructions
        assert prog2.entry_point == prog.entry_point

    def test_disassemble(self):
        prog = self.asm.assemble(SAMPLE_ASM)
        text = self.disasm.disassemble(prog)
        lines = text.strip().split('\n')
        assert len(lines) == 10
        assert 'NOP' in lines[0]
        assert 'SYNC' in lines[1]
        assert 'CONFIG_TILE' in lines[2]
        assert 'HALT' in lines[-1]

    def test_assemble_disassemble_assemble_identity(self):
        """Assemble → disassemble → re-assemble produces identical binary."""
        prog1 = self.asm.assemble(SAMPLE_ASM)

        text = self.disasm.disassemble(prog1)

        # Convert disassembled text to assembly that can be re-assembled
        # The disassembler uses [0xXXXX] prefix, strip it
        lines = []
        for line in text.split('\n'):
            if '] ' in line:
                lines.append(line.split('] ', 1)[1])
            else:
                lines.append(line)

        prog2 = self.asm.assemble('\n'.join(lines))
        assert prog1.instructions == prog2.instructions, \
            f"Binary mismatch after round-trip"

    def test_nop_encoding(self):
        prog = self.asm.assemble("NOP")
        raw_insn = prog.get_instruction_bytes(0)
        insn = decode(raw_insn)
        from taccel.isa.instructions import NopInsn
        assert isinstance(insn, NopInsn)

    def test_config_tile_tile_count_convention(self):
        """Assembly uses tile counts (1-based), encoding is 0-based."""
        prog = self.asm.assemble("CONFIG_TILE M=13, N=13, K=4")
        insn = decode(prog.get_instruction_bytes(0))
        # Encoded as M-1=12, N-1=12, K-1=3
        assert insn.M == 12 and insn.N == 12 and insn.K == 3

    def test_data_section(self):
        import os
        data = os.urandom(64)
        prog = self.asm.assemble("NOP\nHALT", data=data)
        raw = prog.to_bytes()
        prog2 = ProgramBinary.from_bytes(raw)
        assert prog2.data == data

    def test_empty_program(self):
        prog = self.asm.assemble("")
        assert prog.insn_count == 0


class TestDisassembler:
    def test_config_tile_display(self):
        """Disassembler shows tile counts (1-based)."""
        asm = Assembler()
        prog = asm.assemble("CONFIG_TILE M=13, N=13, K=4")
        disasm = Disassembler()
        text = disasm.disassemble(prog)
        assert "M=13" in text and "N=13" in text and "K=4" in text

    def test_buf_copy_transpose_shown(self):
        prog = Assembler().assemble(
            "BUF_COPY src_buf=ABUF, src_off=0, dst_buf=WBUF, dst_off=0, "
            "length=832, src_rows=13, transpose=1"
        )
        text = Disassembler().disassemble(prog)
        assert "transpose=1" in text
        assert "src_rows=13" in text
