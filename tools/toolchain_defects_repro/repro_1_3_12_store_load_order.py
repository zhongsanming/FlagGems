#!/usr/bin/env python3
"""Defect 1.3-12: within one kernel, store (MTE3) and load (MTE2) are not ordered.

Doc claim: storing D/E and loading them back **in the same kernel** races (MTE3
and MTE2 are independent hardware queues), giving 2/8 wrong on the register
version; results drift. Workaround: write in one kernel, read in another
(kernel-boundary ordering), or make the write idempotent.

This kernel writes `mid = x * 2` into a GM buffer, then reads it back and adds
`+1`, all in one kernel. Reference = x*2 + 1.
  A) fused_writeread : store then load in the same kernel    (suspect)
  B) split           : two kernels (write, then read)        (workaround)

Run: python repro_1_3_12_store_load_order.py [--repeat 50] [--n 8192]
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def fused_writeread(x_ptr, mid_ptr, out_ptr, n, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK)
        m = o < n
        x = tl.load(x_ptr + o, mask=m, other=0.0)
        tl.store(mid_ptr + o, x * 2.0, mask=m)     # MTE3 store
        y = tl.load(mid_ptr + o, mask=m, other=0.0)  # MTE2 load of same addr
        tl.store(out_ptr + o, y + 1.0, mask=m)

    @triton.jit
    def write_only(x_ptr, mid_ptr, n, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK)
        m = o < n
        x = tl.load(x_ptr + o, mask=m, other=0.0)
        tl.store(mid_ptr + o, x * 2.0, mask=m)

    @triton.jit
    def read_only(mid_ptr, out_ptr, n, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK)
        m = o < n
        y = tl.load(mid_ptr + o, mask=m, other=0.0)
        tl.store(out_ptr + o, y + 1.0, mask=m)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    torch.manual_seed(0)
    n = args.n
    BLOCK = triton.next_power_of_2(n)
    x = torch.randn(n, dtype=torch.float32, device=dev)
    ref = (x.detach().to("cpu") * 2.0 + 1.0)

    def run_fused():
        mid = torch.empty(n, dtype=torch.float32, device=dev)
        out = torch.empty(n, dtype=torch.float32, device=dev)
        fused_writeread[(1,)](x, mid, out, n, BLOCK=BLOCK)
        return out.detach().to("cpu")

    def run_split():
        mid = torch.empty(n, dtype=torch.float32, device=dev)
        out = torch.empty(n, dtype=torch.float32, device=dev)
        write_only[(1,)](x, mid, n, BLOCK=BLOCK)
        read_only[(1,)](mid, out, n, BLOCK=BLOCK)
        return out.detach().to("cpu")

    bad_f = bad_s = 0
    for _ in range(args.repeat):
        if not torch.allclose(run_fused(), ref, atol=1e-5):
            bad_f += 1
        if not torch.allclose(run_split(), ref, atol=1e-5):
            bad_s += 1
    print(f"n={n} BLOCK={BLOCK} repeat={args.repeat}")
    print(f"  fused write->read : wrong = {bad_f}/{args.repeat}")
    print(f"  split two kernels : wrong = {bad_s}/{args.repeat}")
    if bad_f and not bad_s:
        print("REPRODUCED: same-kernel store->load is not ordered; split ok.")
        return 0
    if not bad_f:
        print("Not reproduced this run (race is timing-dependent; vary n/repeat, "
              "and run under load).")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
