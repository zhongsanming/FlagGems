#!/usr/bin/env python3
"""Defect 1.1-1: tl.where(mask, vec, 0.0) scalar-zero branch miscompiles.

Doc claim: `tl.where(mask, vec, 0.0)` (vector + *scalar constant 0* branch)
compiles but is probabilistically wrong on device. Workaround: use
`vec * mask.to(f32)`. The wrongness is independent of the mask content; only the
"scalar constant 0 as a branch" matters.

This script runs a tiny kernel doing mask-zeroing two ways and compares to a
numpy/torch reference:
  A) masked_zero_where : tl.where(m, x, 0.0)          (suspect)
  B) masked_zero_mul   : x * m.to(tl.float32)          (correct impl)

A and B must be numerically identical. If A diverges (probabilistically) from
the reference while B does not, the defect is reproduced.

Run:  python repro_1_1_1_where_scalar_zero.py [--repeat 50] [--n 4096]
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def masked_zero_where(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK)
        m = o < n
        x = tl.load(x_ptr + o, mask=m)
        y = tl.where(m, x, 0.0)          # <-- suspect: scalar 0 branch
        tl.store(out_ptr + o, y, mask=m)

    @triton.jit
    def masked_zero_mul(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK)
        m = o < n
        x = tl.load(x_ptr + o, mask=m)
        y = x * m.to(tl.float32)         # <-- workaround
        tl.store(out_ptr + o, y, mask=m)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    torch.manual_seed(0)
    n = args.n
    BLOCK = triton.next_power_of_2(n)
    x = torch.randn(n, dtype=torch.float32, device=dev)

    # reference: where keeps x in-range, 0 out-of-range; here all lanes in range,
    # so both must equal x.
    ref = x.detach().to("cpu")

    def run(kernel):
        out = torch.empty(n, dtype=torch.float32, device=dev)
        kernel[(1,)](x, out, n, BLOCK=BLOCK)
        return out.detach().to("cpu")

    bad_where = bad_mul = 0
    for i in range(args.repeat):
        a = run(masked_zero_where)
        b = run(masked_zero_mul)
        if not torch.equal(a, ref):
            bad_where += 1
        if not torch.equal(b, ref):
            bad_mul += 1

    print(f"n={n} repeat={args.repeat} BLOCK={BLOCK}")
    print(f"  where(m, x, 0.0): wrong runs = {bad_where}/{args.repeat}")
    print(f"  x * m.to(f32)   : wrong runs = {bad_mul}/{args.repeat}")
    if bad_where and not bad_mul:
        print("REPRODUCED: scalar-zero where branch is wrong; mul workaround ok.")
        return 0
    if not bad_where:
        print("Not reproduced this run (defect is probabilistic; increase --repeat "
              "and vary shape).")
        return 1
    print("Both wrong -> different cause (check load/reduction).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
