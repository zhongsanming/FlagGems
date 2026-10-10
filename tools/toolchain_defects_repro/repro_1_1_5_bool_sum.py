#!/usr/bin/env python3
"""Defect 1.1-5: `tl.sum(bool_mask)` returns 1 instead of the true count.

Doc claim: summing a boolean tensor returns 1 (always), not the number of true
entries. Workaround: cast to int32 first (`bool_mask.to(tl.int32)`).

  A) bool_sum : tl.sum(mask_bool)              (suspect)
  B) int_sum  : tl.sum(mask_bool.to(tl.int32)) (workaround)
Both must equal the true count of nonzero lanes.

Run: python repro_1_1_5_bool_sum.py [--repeat 20]
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
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

    @triton.jit
    def count_bool(x_ptr, out_ptr, B: tl.constexpr):
        ar = tl.arange(0, B)
        x = tl.load(x_ptr + ar)
        mask = x > 0.0
        c = tl.sum(mask)                    # <-- suspect (bool reduction)
        tl.store(out_ptr, c)

    @triton.jit
    def count_int(x_ptr, out_ptr, B: tl.constexpr):
        ar = tl.arange(0, B)
        x = tl.load(x_ptr + ar)
        mask = x > 0.0
        c = tl.sum(mask.to(tl.int32))       # <-- workaround
        tl.store(out_ptr, c)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    torch.manual_seed(0)
    x = torch.randn(B, dtype=torch.float32, device=dev)
    ref = int((x > 0).sum().item())

    def run(kernel):
        out = torch.zeros(1, dtype=torch.int32, device=dev)
        kernel[(1,)](x, out, B=B)
        return int(out.item())

    bad_b = bad_i = 0
    got_b = set()
    for _ in range(args.repeat):
        a = run(count_bool); b = run(count_int)
        bad_b += a != ref; bad_i += b != ref
        got_b.add(a)
    print(f"B={B} ref_count={ref} repeat={args.repeat}")
    print(f"  tl.sum(bool)          : wrong = {bad_b}/{args.repeat}  "
          f"returned values={sorted(got_b)[:5]}")
    print(f"  tl.sum(bool.to(i32))  : wrong = {bad_i}/{args.repeat}")
    if bad_b and not bad_i:
        mode = "ALWAYS returns 1" if got_b == {1} else \
               f"returns wrong value(s) {sorted(got_b)[:5]}"
        print(f"REPRODUCED: bool reduction is wrong ({mode}); int32 cast ok.")
        return 0
    if not bad_b:
        print("Not reproduced (this toolchain may already fix it).")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
