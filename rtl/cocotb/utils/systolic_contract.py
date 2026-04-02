"""Shared cocotb helpers for logical MATMUL contract preparation."""

from utils.insn_builder import BUF_COPY, LOAD, SET_ADDR_HI, SET_ADDR_LO, SYNC, BUF_ABUF, BUF_WBUF, BUF_ACCUM


SYS_DIM = 16
A_STAGE_OFF = 1024


def set_addr_insns(reg: int, addr: int) -> list[int]:
    return [
        SET_ADDR_LO(reg, addr & 0x0FFFFFFF),
        SET_ADDR_HI(reg, (addr >> 28) & 0x0FFFFFFF),
    ]


def append_load_sync(prog: list[int], reg: int, addr: int, buf_id: int, sram_off: int, xfer_len: int) -> None:
    prog.extend(set_addr_insns(reg, addr))
    prog.append(LOAD(buf_id, sram_off, xfer_len, reg, 0))
    prog.append(SYNC(0b001))


def append_prepare_a_tile(prog: list[int], reg: int, addr: int, abuf_off: int) -> None:
    append_load_sync(prog, reg, addr, BUF_ACCUM, A_STAGE_OFF, SYS_DIM)
    prog.append(BUF_COPY(BUF_ACCUM, A_STAGE_OFF, BUF_ABUF, abuf_off, SYS_DIM, 1, 1))


def append_prepare_b_tile(prog: list[int], reg: int, addr: int, wbuf_off: int) -> None:
    append_load_sync(prog, reg, addr, BUF_WBUF, wbuf_off, SYS_DIM)


def flatten_16x16(mat: list[list[int]]) -> bytes:
    return bytes((row[col] & 0xFF) for row in mat for col in range(16))


def flatten_tile_32x32(mat: list[list[int]], row_base: int, col_base: int) -> bytes:
    return bytes((mat[row_base + r][col_base + c] & 0xFF) for r in range(16) for c in range(16))


def flatten_tile_16x64(mat: list[list[int]], col_base: int) -> bytes:
    return bytes((mat[r][col_base + c] & 0xFF) for r in range(16) for c in range(16))


def flatten_tile_64x16(mat: list[list[int]], row_base: int) -> bytes:
    return bytes((mat[row_base + r][c] & 0xFF) for r in range(16) for c in range(16))


def prepare_logical_16x16(dram, prog: list[int], a: list[list[int]], b: list[list[int]],
                          a_addr: int, b_addr: int, abuf_off: int = 0, wbuf_off: int = 0) -> None:
    dram.write_bytes(a_addr, flatten_16x16(a))
    dram.write_bytes(b_addr, flatten_16x16(b))
    append_prepare_a_tile(prog, 0, a_addr, abuf_off)
    append_prepare_b_tile(prog, 1, b_addr, wbuf_off)


def prepare_logical_32x32(dram, prog: list[int], a: list[list[int]], b: list[list[int]],
                          a_base: int, b_base: int, abuf_off: int = 0, wbuf_off: int = 0) -> None:
    for mt in range(2):
        for kt in range(2):
            tile = mt * 2 + kt
            a_addr = a_base + tile * 0x1000
            dram.write_bytes(a_addr, flatten_tile_32x32(a, mt * 16, kt * 16))
            append_prepare_a_tile(prog, 0, a_addr, abuf_off + tile * 16)

    for kt in range(2):
        for nt in range(2):
            tile = kt * 2 + nt
            b_addr = b_base + tile * 0x1000
            dram.write_bytes(b_addr, flatten_tile_32x32(b, kt * 16, nt * 16))
            append_prepare_b_tile(prog, 1, b_addr, wbuf_off + tile * 16)


def prepare_logical_16x64x16(dram, prog: list[int], a: list[list[int]], b: list[list[int]],
                             a_base: int, b_base: int, abuf_off: int = 0, wbuf_off: int = 0) -> None:
    for kt in range(4):
        a_addr = a_base + kt * 0x1000
        b_addr = b_base + kt * 0x1000
        dram.write_bytes(a_addr, flatten_tile_16x64(a, kt * 16))
        dram.write_bytes(b_addr, flatten_tile_64x16(b, kt * 16))
        append_prepare_a_tile(prog, 0, a_addr, abuf_off + kt * 16)
        append_prepare_b_tile(prog, 1, b_addr, wbuf_off + kt * 16)
