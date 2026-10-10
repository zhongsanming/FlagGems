#!/usr/bin/env python3
"""Defect 1.5-17: JIT launch/dispatch overhead is ~460us/launch on Ascend.

Doc claim: a triton JIT dispatch costs ~460us per launch (torch native op ~31us);
pre-binding via `jit_fn.warmup()` + `CompiledKernel.run(...)` reduces it to
~12us/launch. This is a *performance* ceiling, not a correctness bug.

This script measures wall-clock per launch for a trivial kernel via the normal
`kernel[grid](...)` path, so you can confirm the ~0.4ms floor on your machine.

Run: python repro_1_5_jit_dispatch_latency.py [--iters 200]
"""

from __future__ import annotations

import argparse
import os
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--b", type=int, default=256)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def triv(x_ptr, out_ptr, B: tl.constexpr):
        ar = tl.arange(0, B)
        tl.store(out_ptr + ar, tl.load(x_ptr + ar) + 1.0)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    B = args.b
    x = torch.randn(B, dtype=torch.float32, device=dev)
    out = torch.empty(B, dtype=torch.float32, device=dev)

    # warm compile
    triv[(1,)](x, out, B=B)
    torch.npu.synchronize() if hasattr(torch, "npu") else torch.cuda.synchronize()

    n = args.iters
    t0 = time.perf_counter()
    for _ in range(n):
        triv[(1,)](x, out, B=B)
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"B={B} iters={n}")
    print(f"  triton JIT launch: {dt/n*1e6:.1f} us/launch")

    # torch native baseline
    t0 = time.perf_counter()
    for _ in range(n):
        out = torch.add(x, 1.0)
    torch.npu.synchronize() if hasattr(torch, "npu") else torch.cuda.synchronize()
    dt2 = time.perf_counter() - t0
    print(f"  torch native add : {dt2/n*1e6:.1f} us/launch")
    print("\nDoc expects triton ~460us vs torch ~31us. This is a performance")
    print("observation, not a correctness defect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
