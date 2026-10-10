#!/usr/bin/env python3
"""Defect 1.1-9: scalar bitcast miscompiles (vector bitcast is fine).

Doc claim: `scalar.to(tl.int32, bitcast=True)` (used for ULP/nextafter-style
steps) gives a wrong result; the same bitcast on a *vector* works. Workaround:
apply the bitcast to a vector (even of size 1 / on a vector of values), or avoid
scalar bitcast.

  A) scalar_bitcast : bitcast a 0-d value via `x.to(tl.int32, bitcast=True)` (suspect)
  B) vector_bitcast : load a 1-element vector and bitcast the vector           (ok)

Reference: the exact IEEE-754 bit pattern of the float (compute on CPU).

Run: python repro_1_1_9_scalar_bitcast.py [--repeat 20]
"""

from __future__ import annotations

import argparse
import os
import struct


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--val", type=float, default=1.5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def scalar_bitcast(x_ptr, out_ptr):
        x = tl.load(x_ptr)                 # 0-d scalar
        b = x.to(tl.int32, bitcast=True)   # <-- suspect scalar bitcast
        tl.store(out_ptr, b)

    @triton.jit
    def vector_bitcast(x_ptr, out_ptr, B: tl.constexpr):
        ar = tl.arange(0, B)
        x = tl.load(x_ptr + ar)            # vector
        b = x.to(tl.int32, bitcast=True)   # <-- vector bitcast
        tl.store(out_ptr + ar, b)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    x = torch.tensor([args.val], dtype=torch.float32, device=dev)
    ref = f32_bits(args.val)

    def run_scalar():
        out = torch.zeros(1, dtype=torch.int32, device=dev)
        scalar_bitcast[(1,)](x, out)
        return int(out.item())

    def run_vector():
        out = torch.zeros(1, dtype=torch.int32, device=dev)
        vector_bitcast[(1,)](x, out, B=1)
        return int(out.item())

    bad_s = bad_v = 0
    for _ in range(args.repeat):
        bad_s += run_scalar() != ref
        bad_v += run_vector() != ref
    print(f"val={args.val} ref_bits=0x{ref:08x} repeat={args.repeat}")
    print(f"  scalar bitcast : wrong = {bad_s}/{args.repeat}")
    print(f"  vector bitcast : wrong = {bad_v}/{args.repeat}")
    if bad_s and not bad_v:
        print("REPRODUCED: scalar bitcast wrong; vector bitcast ok.")
        return 0
    if not bad_s:
        print("Not reproduced (may already be fixed).")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
