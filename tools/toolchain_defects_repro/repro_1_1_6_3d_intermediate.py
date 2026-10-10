#!/usr/bin/env python3
"""Defect 1.1-6: 3D intermediate `[BP, BLOCK, BLOCK]` where/reduce fails to compile.

Doc claim: a 3D intermediate tile (e.g. `[BP, BLOCK, BLOCK]`) used in
where/reduction (multi-pair parallel Jacobi) fails to compile with an empty
error. Workaround: do everything in 2D.

  A) build_3d : allocate/where on a [BP, B, B] tensor   (suspect: compile error)
  B) build_2d : flatten BP*B rows, operate on [BP*B, B] (workaround)

Both compute, per pair p, the row-normalized matrix; we just check they compile
and run. The defect is a *compile* failure, so catching the exception is enough.

Run: python repro_1_1_6_3d_intermediate.py
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", type=int, default=4)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    BP, B = args.bp, args.b

    @triton.jit
    def build_3d(x_ptr, out_ptr, BP: tl.constexpr, B: tl.constexpr):
        p = tl.arange(0, BP)
        r = tl.arange(0, B)
        c = tl.arange(0, B)
        off = p[:, None, None] * B * B + r[None, :, None] * B + c[None, None, :]
        g = tl.load(x_ptr + off)                     # [BP, B, B]
        sel = tl.where(r[None, :, None] == c[None, None, :], g, 0.0)  # 3D where
        s = tl.sum(sel, axis=2)                      # 3D reduce
        o2 = p[:, None] * B + r[None, :]
        tl.store(out_ptr + o2, s)

    @triton.jit
    def build_2d(x_ptr, out_ptr, BP: tl.constexpr, B: tl.constexpr):
        rows = tl.arange(0, BP * B)
        c = tl.arange(0, B)
        g = tl.load(x_ptr + rows[:, None] * B + c[None, :])   # [BP*B, B]
        rr = rows[:, None] % B
        sel = tl.where(rr == c[None, :], g, 0.0)
        s = tl.sum(sel, axis=1)                       # 2D reduce
        tl.store(out_ptr + rows, s)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    torch.manual_seed(0)
    x = torch.randn(BP * B * B, dtype=torch.float32, device=dev)

    def try_kernel(kernel, out_shape):
        try:
            out = torch.zeros(out_shape, dtype=torch.float32, device=dev)
            kernel[(1,)](x, out, BP=BP, B=B)
            return "ok", out
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {str(e)[:200]}", None

    s3, _ = try_kernel(build_3d, (BP, B))
    s2, _ = try_kernel(build_2d, (BP * B,))
    print(f"BP={BP} B={B}")
    print("  3D kernel:", s3)
    print("  2D kernel:", s2)
    if s3 != "ok" and s2 == "ok":
        print("REPRODUCED: 3D intermediate fails to compile; 2D works.")
        return 0
    if s3 == "ok":
        print("Not reproduced: 3D kernel compiled (fixed/other toolchain).")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
