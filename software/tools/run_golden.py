#!/usr/bin/env python3
"""CLI: simulate program.bin with input image"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="TACCEL golden model simulator")
    parser.add_argument("program", help="Compiled program.bin")
    parser.add_argument("--input", help="Input image file (.npy or .bin)")
    parser.add_argument("--output", help="Output logits file (.npy)")
    parser.add_argument("--top-k", type=int, default=5, help="Show top-K predictions")
    args = parser.parse_args()

    print(f"Loading program {args.program}...")
    from taccel.assembler.assembler import ProgramBinary
    with open(args.program, 'rb') as f:
        raw = f.read()
    prog = ProgramBinary.from_bytes(raw)
    print(f"  {prog.insn_count} instructions, {len(prog.data):,} bytes data")

    from taccel.golden_model import Simulator, MachineState
    state = MachineState(dram_data=prog.data)
    sim = Simulator(state)
    sim.load_program(prog)

    # Load input if provided
    if args.input:
        if args.input.endswith('.npy'):
            inp = np.load(args.input)
        else:
            inp = np.frombuffer(open(args.input, 'rb').read(), dtype=np.int8)
        # Write input to ABUF
        state.abuf[:len(inp)] = inp.tobytes()
        print(f"Loaded input: {inp.shape}")

    print("Running simulation...")
    count = sim.run()
    print(f"  Executed {count} instructions, {state.cycle_count} cycles")

    # Extract output from ABUF (classifier output at start of ABUF)
    # The compiler places classifier output at ABUF[0] as INT32 in ACCUM
    logits_int32 = state.accum[:1000].copy()
    logits_fp32 = logits_int32.astype(np.float32)

    if args.output:
        np.save(args.output, logits_fp32)
        print(f"Saved logits to {args.output}")

    # Show top predictions
    top_k = min(args.top_k, 1000)
    top_indices = np.argsort(logits_fp32)[::-1][:top_k]
    print(f"\nTop-{top_k} predictions:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. class {idx}: {logits_fp32[idx]:.2f}")


if __name__ == "__main__":
    main()
