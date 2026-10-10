#!/usr/bin/env python3
"""Defect 1.1-4: `tl.arange == const` used as a mask raises 507035.

Doc claim: comparing `tl.arange` to a constant with `==` and using it as a mask
triggers a 507035 device exception; a strict inequality (`rows > c`) is fine.
Workaround: rewrite the equality mask as an interval
`(x > c-1) & (x < c+1)`.

  A) eq_mask  : sel = tl.arange(0,B) == C ; use as load/store mask   (suspect)
  B) int_mask : sel = (ar > C-1) & (ar < C+1)                        (workaround)

Run: python repro_1_1_4_arange_eq_mask.py [--repeat 20]
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--c", type=int, default=5)
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    B, C = args.b, args.c

    @triton.jit
    def copy_eq(x_ptr, out_ptr, C: tl.constexpr, B: tl.constexpr):
        ar = tl.arange(0, B)
        sel = ar == C                       # <-- suspect equality mask
        x = tl.load(x_ptr + ar, mask=sel, other=0.0)
        tl.store(out_ptr + ar, x, mask=sel)

    @triton.jit
    def copy_int(x_ptr, out_ptr, C: tl.constexpr, B: tl.constexpr):
        ar = tl.arange(0, B)
        sel = (ar > C - 1) & (ar < C + 1)   # <-- interval workaround
        x = tl.load(x_ptr + ar, mask=sel, other=0.0)
        tl.store(out_ptr + ar, x, mask=sel)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    torch.manual_seed(0)
    x = torch.randn(B, dtype=torch.float32, device=dev)
    ref = torch.zeros(B, dtype=torch.float32)
    ref[C] = x[C].detach().to("cpu")

    def safe(kernel):
        try:
            out = torch.empty(B, dtype=torch.float32, device=dev)
            kernel[(1,)](x, out, C=C, B=B)
            return out.detach().to("cpu"), None
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"

    bad_e = exc_e = bad_i = exc_i = 0
    first_err = None
    for _ in range(args.repeat):
        a, ea = safe(copy_eq)
        b, eb = safe(copy_int)
        exc_e += ea is not None; bad_e += (a is None or not torch.equal(a, ref))
        exc_i += eb is not None; bad_i += (b is None or not torch.equal(b, ref))
        if ea and first_err is None:
            first_err = ea
    print(f"B={B} C={C} repeat={args.repeat}")
    print(f"  arange==C mask : wrong/err = {bad_e}/{args.repeat}  exceptions={exc_e}")
    print(f"  interval mask  : wrong/err = {bad_i}/{args.repeat}  exceptions={exc_i}")
    if first_err:
        print("  first exception:", first_err[:200])
    if (bad_e or exc_e) and not (bad_i or exc_i):
        print("REPRODUCED: arange==const mask miscompiles; interval mask ok.")
        return 0
    if not (bad_e or exc_e):
        print("Not reproduced (version/hardware-dependent).")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
