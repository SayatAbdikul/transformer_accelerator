#!/usr/bin/env python3
"""CLI: pytorch_model.bin → program.bin"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="TACCEL compiler: PyTorch model → ProgramBinary")
    parser.add_argument("--weights", required=True, help="PyTorch weights file (pytorch_model.bin)")
    parser.add_argument("--model", default="deit-tiny",
                        choices=["deit-tiny"], help="Model architecture")
    parser.add_argument("-o", "--output", default="program.bin", help="Output .bin file")
    parser.add_argument("--calibration-images", nargs="*",
                        help="Image files for calibration (optional)")
    args = parser.parse_args()

    print(f"Loading weights from {args.weights}...")
    import torch
    state_dict = torch.load(args.weights, map_location="cpu")
    if hasattr(state_dict, 'items') is False:
        state_dict = state_dict.state_dict()

    print(f"Compiling {args.model}...")
    from taccel.compiler import Compiler
    compiler = Compiler()
    prog = compiler.compile(state_dict)

    with open(args.output, 'wb') as f:
        f.write(prog.to_bytes())

    print(f"\nCompilation complete:")
    print(f"  Instructions: {prog.insn_count}")
    print(f"  Instruction bytes: {len(prog.instructions):,}")
    print(f"  Data (weights): {len(prog.data):,} bytes")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
