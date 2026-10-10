#!/usr/bin/env python3
"""Defect 1.1-7: pure dynamic `while` carrying a tensor crashes on device.

Doc claim: `while j < K-1` carrying a tile as loop state raises a device
exception; full `tl.static_range` unrolling >~32 steps blows up compile time
(>10 min). Workaround: `tl.range` dynamic outer x `tl.static_range` inner <=8.

  A) while_dyn : `while` loop carrying a tile                       (suspect)
  B) range_hyb : tl.range(C) outer, static_range(INNER) inner       (workaround)

Both compute the same recurrence sum over C steps; compare to reference.

Run: python repro_1_1_7_dynamic_while.py [--c 16] [--repeat 10]
"""

from __future__ import annotations

import argparse
import os
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=16)
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    B, C, INNER = args.b, args.c, args.inner

    @triton.jit
    def while_dyn(x_ptr, out_ptr, B: tl.constexpr, C: tl.constexpr):
        ar = tl.arange(0, B)
        acc = tl.zeros((B,), dtype=tl.float32)
        j = 0
        while j < C:                       # <-- pure dynamic while, tile state
            acc += tl.load(x_ptr + ar + j)
            j += 1
        tl.store(out_ptr + ar, acc)

    @triton.jit
    def range_hyb(x_ptr, out_ptr, B: tl.constexpr, C: tl.constexpr,
                  INNER: tl.constexpr):
        ar = tl.arange(0, B)
        acc = tl.zeros((B,), dtype=tl.float32)
        for j0 in tl.range(0, C, INNER):                 # dynamic outer
            for jj in tl.static_range(INNER):            # static inner
                acc += tl.load(x_ptr + ar + j0 + jj)
        tl.store(out_ptr + ar, acc)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    torch.manual_seed(0)
    # x laid out [C, B] so row j is at offset j*B
    x = torch.randn(C, B, dtype=torch.float32, device=dev).reshape(-1).contiguous()
    # our indexing `x_ptr + ar + j` assumes row-major with stride 1 in B and
    # step 1 between rows -> emulate by tiling x with B-contiguous rows:
    x = torch.randn(C, B, dtype=torch.float32, device=dev).reshape(-1)
    ref = x.reshape(C, B).sum(0).detach().to("cpu")

    def safe(kernel):
        t0 = time.time()
        try:
            out = torch.empty(B, dtype=torch.float32, device=dev)
            kernel[(1,)](x, out, B=B, C=C, INNER=INNER) if kernel is range_hyb else \
                kernel[(1,)](x, out, B=B, C=C)
            return out.detach().to("cpu"), time.time() - t0, None
        except Exception as e:  # noqa: BLE001
            return None, time.time() - t0, f"{type(e).__name__}: {str(e)[:160]}"

    # NOTE: fix indexing for while_dyn: x_ptr + ar + j assumes row j contiguous
    # after B, i.e. offset j. That is the intended simple recurrence here.
    errs_w = errs_r = 0
    dt_w = dt_r = 0.0
    first = None
    for _ in range(args.repeat):
        a, dt, ea = safe(while_dyn)
        b, dt2, eb = safe(range_hyb)
        dt_w += dt; dt_r += dt2
        if ea or a is None or not torch.allclose(a, ref, atol=1e-3):
            errs_w += 1
            first = first or ea
        if eb or b is None or not torch.allclose(b, ref, atol=1e-3):
            errs_r += 1
    print(f"B={B} C={C} INNER={INNER} repeat={args.repeat}")
    print(f"  while(tile state) : errs = {errs_w}/{args.repeat}  avg {dt_w/args.repeat:.3f}s")
    print(f"  tl.range+static   : errs = {errs_r}/{args.repeat}  avg {dt_r/args.repeat:.3f}s")
    if first:
        print("  first error:", first)
    if errs_w and not errs_r:
        print("REPRODUCED: dynamic while carrying a tile fails; hybrid loop ok.")
        return 0
    if not errs_w:
        print("Not reproduced (may need larger C / different layout).")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
