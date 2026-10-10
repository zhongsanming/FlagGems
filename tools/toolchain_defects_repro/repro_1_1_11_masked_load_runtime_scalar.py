#!/usr/bin/env python3
"""Defect 1.1-11: masked load + runtime scalar cannot be legalized.

Doc claim: a vectorized expression mixing a *masked load* with any *runtime
scalar* fails to legalize, stuck on MLIR `() -> tensor` materialization, in five
different formulations. Workaround: do the scalar correction on the host (or use
a normalized flag), not in a Triton early-exit kernel.

This tries several formulations of `masked_load + runtime_scalar` and reports
which fail to compile. A formulation that only uses a *constexpr* scalar
(`--as-constexpr`) should compile, showing the runtime scalar is the trigger.

Run: python repro_1_1_11_masked_load_runtime_scalar.py
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=256)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    B = args.b
    results = {}

    # five formulations mixing a masked load with a runtime scalar
    @triton.jit
    def f1(x_ptr, out_ptr, s, n, B: tl.constexpr):
        ar = tl.arange(0, B)
        m = ar < n
        x = tl.load(x_ptr + ar, mask=m, other=0.0)
        tl.store(out_ptr + ar, x + s, mask=m)                 # add

    @triton.jit
    def f2(x_ptr, out_ptr, s, n, B: tl.constexpr):
        ar = tl.arange(0, B)
        m = ar < n
        x = tl.load(x_ptr + ar, mask=m, other=0.0)
        y = tl.where(x > s, x, s)                             # where with scalar
        tl.store(out_ptr + ar, y, mask=m)

    @triton.jit
    def f3(x_ptr, out_ptr, s, n, B: tl.constexpr):
        ar = tl.arange(0, B)
        m = ar < n
        x = tl.load(x_ptr + ar, mask=m, other=0.0)
        y = tl.maximum(x, tl.full((B,), 0.0, tl.float32) + s)  # broadcast scalar
        tl.store(out_ptr + ar, y, mask=m)

    @triton.jit
    def f4(x_ptr, out_ptr, s, n, B: tl.constexpr):
        ar = tl.arange(0, B)
        m = ar < n
        x = tl.load(x_ptr + ar, mask=m, other=0.0)
        y = x * (s * 2.0)
        tl.store(out_ptr + ar, y, mask=m)

    @triton.jit
    def f5(x_ptr, out_ptr, s, n, B: tl.constexpr):
        ar = tl.arange(0, B)
        m = ar < n
        x = tl.load(x_ptr + ar, mask=m, other=0.0)
        y = tl.where(x > s, x - s, x + s)
        tl.store(out_ptr + ar, y, mask=m)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    x = torch.randn(B, dtype=torch.float32, device=dev)
    s = torch.tensor(0.25, dtype=torch.float32, device=dev)

    for name, k in (("add", f1), ("where", f2), ("maximum+bcast", f3),
                    ("mul", f4), ("where+-", f5)):
        try:
            out = torch.empty(B, dtype=torch.float32, device=dev)
            k[(1,)](x, out, float(s.item()), B, B=B)
            results[name] = "ok"
        except Exception as e:  # noqa: BLE001
            results[name] = f"{type(e).__name__}: {str(e)[:120]}"

    print(f"B={B}")
    for name, r in results.items():
        print(f"  {name:14s}: {r}")
    n_fail = sum(1 for r in results.values() if r != "ok")
    if n_fail:
        print(f"REPRODUCED ({n_fail}/5 formulations fail to compile).")
        return 0
    print("Not reproduced: all formulations compiled.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
