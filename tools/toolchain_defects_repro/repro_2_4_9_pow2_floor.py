#!/usr/bin/env python3
"""Defect 2.4-9: `exp2(floor(log2(x)))` is imprecise; bitmask must be on vectors.

Doc claims (scale-normalization section):
  * `exp2(floor(log2(x)))` is inaccurate on this backend (0.5 -> 0.49999997);
    a pure *bitmask* exponent trick is exact.
  * the bitmask trick must be applied to a *vector* (scalar bitcast is broken,
    see 1.1-9).

For powers-of-two inputs, the true `power_of_two_floor(x)` is x itself. We compute
it two ways:
  A) exp2_log2 : tl.exp2(tl.floor(tl.log2(x)))     (suspect, imprecise)
  B) bitmask   : reinterpret the f32 exponent and mask it (exact)

Run: python repro_2_4_9_pow2_floor.py [--repeat 5]
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=5)
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
    def pow2_exp2_log2(x_ptr, out_ptr, B: tl.constexpr):
        ar = tl.arange(0, B)
        x = tl.load(x_ptr + ar)
        p = tl.exp2(tl.floor(tl.log2(x)))          # <-- suspect
        tl.store(out_ptr + ar, p)

    @triton.jit
    def pow2_bitmask(x_ptr, out_ptr, B: tl.constexpr):
        ar = tl.arange(0, B)
        x = tl.load(x_ptr + ar)
        # exponent-only bitmask: clear the mantissa bits (23 low bits of f32)
        bits = x.to(tl.int32, bitcast=True)
        masked = bits & 0xFF800000
        p = masked.to(tl.float32, bitcast=True)     # <-- exact (vector bitcast)
        tl.store(out_ptr + ar, p)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    # powers of two -> true floor-pow2 equals the value
    vals = torch.tensor([0.5, 1.0, 2.0, 4.0, 0.25, 8.0, 16.0, 0.125] * (B // 8),
                        dtype=torch.float32, device=dev)[:B]
    ref = vals.detach().to("cpu")

    def run(kernel):
        out = torch.empty(B, dtype=torch.float32, device=dev)
        kernel[(1,)](vals, out, B=B)
        return out.detach().to("cpu")

    bad_e = bad_b = 0
    for _ in range(args.repeat):
        if not torch.equal(run(pow2_exp2_log2), ref):
            bad_e += 1
        if not torch.equal(run(pow2_bitmask), ref):
            bad_b += 1
    a = run(pow2_exp2_log2)
    print(f"B={B} repeat={args.repeat}")
    print(f"  exp2(floor(log2)) : exact-match runs = {args.repeat-bad_e}/{args.repeat}")
    print(f"  exponent bitmask  : exact-match runs = {args.repeat-bad_b}/{args.repeat}")
    if bad_e and not bad_b:
        diff = (a != ref)
        idx = int(diff.nonzero()[0]) if diff.any() else 0
        print(f"  e.g. x={float(ref[idx])} -> exp2log2={float(a[idx])} "
              f"(should be {float(ref[idx])})")
        print("REPRODUCED: exp2/log2 path is inexact; bitmask is exact.")
        return 0
    if not bad_e:
        print("Not reproduced: exp2/log2 matched exactly this run.")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
