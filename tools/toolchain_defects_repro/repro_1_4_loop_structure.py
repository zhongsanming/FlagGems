#!/usr/bin/env python3
"""Defect 1.4-16: static_range + range in one function; jit helper tuple/branch.

Doc claims:
  * using `tl.static_range` (e.g. dot loop) and `tl.range` (tridiagonalization
    loop) in the same function -> compile error "cannot reassign constexpr";
  * extracting to a jit helper -> MLIR ConvertLinalgIRToBinary crash;
  * jit helpers with multiple return values (tuple) are fine;
  * a scalar `if hi == 0.0` branch must be rewritten as default + `if hi > 0.0`
    to be inlinable.

This tries:
  A) mixed_loops      : static_range + range in one function      (suspect)
  B) called_helper    : same loops behind a jit helper call       (suspect)
  C) helper_tuple     : jit helper returning a tuple             (should be ok)
  D) scalar_eq_branch : `if hi == 0.0:` scalar branch            (suspect)
  E) scalar_gt_branch : default + `if hi > 0.0:`                  (workaround)

Run: python repro_1_4_loop_structure.py
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    B = args.b

    @triton.jit
    def _helper(a):
        return a + 1.0, a * 2.0

    @triton.jit
    def _loop_helper(acc, x, IN: tl.constexpr):
        for j in tl.static_range(IN):
            acc += x + j
        return acc

    @triton.jit
    def mixed_loops(x_ptr, out_ptr, B: tl.constexpr, R: tl.constexpr,
                    S: tl.constexpr):
        ar = tl.arange(0, B)
        x = tl.load(x_ptr + ar)
        acc = tl.zeros((B,), dtype=tl.float32)
        for i in tl.static_range(S):        # static
            acc += x
        for j in tl.range(R):               # dynamic, same function
            acc += x + j
        tl.store(out_ptr + ar, acc)

    @triton.jit
    def called_helper(x_ptr, out_ptr, B: tl.constexpr, R: tl.constexpr,
                      S: tl.constexpr):
        ar = tl.arange(0, B)
        x = tl.load(x_ptr + ar)
        acc = tl.zeros((B,), dtype=tl.float32)
        for i in tl.static_range(S):
            acc = _loop_helper(acc, x, S)   # call a jit helper inside loop
        for j in tl.range(R):
            acc += x + j
        tl.store(out_ptr + ar, acc)

    @triton.jit
    def helper_tuple(x_ptr, out_ptr, B: tl.constexpr):
        ar = tl.arange(0, B)
        x = tl.load(x_ptr + ar)
        a, b = _helper(x)
        tl.store(out_ptr + ar, a + b)

    @triton.jit
    def scalar_eq_branch(x_ptr, out_ptr, hi, B: tl.constexpr):
        ar = tl.arange(0, B)
        x = tl.load(x_ptr + ar)
        if hi == 0.0:                       # <-- suspect scalar equality branch
            x = x + 1.0
        tl.store(out_ptr + ar, x)

    @triton.jit
    def scalar_gt_branch(x_ptr, out_ptr, hi, B: tl.constexpr):
        ar = tl.arange(0, B)
        x = tl.load(x_ptr + ar)
        y = x
        if hi > 0.0:                        # <-- workaround
            y = x + 1.0
        tl.store(out_ptr + ar, y)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    x = torch.randn(B, dtype=torch.float32, device=dev)

    def try_kernel(name, fn):
        try:
            out = torch.zeros(B, dtype=torch.float32, device=dev)
            fn(out)
            print(f"  {name:22s}: ok")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:22s}: {type(e).__name__}: {str(e)[:110]}")

    print(f"B={B}")
    try_kernel("mixed_loops", lambda out: mixed_loops[(1,)](x, out, B=B, R=8, S=4))
    try_kernel("called_helper", lambda out: called_helper[(1,)](x, out, B=B, R=8, S=4))
    try_kernel("helper_tuple", lambda out: helper_tuple[(1,)](x, out, B=B))
    try_kernel("scalar_eq_branch", lambda out: scalar_eq_branch[(1,)](x, out, 0.0, B=B))
    try_kernel("scalar_gt_branch", lambda out: scalar_gt_branch[(1,)](x, out, 0.0, B=B))
    print("\nCompare 'suspect' rows (mixed_loops/called_helper/scalar_eq_branch)")
    print("against the workaround rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
