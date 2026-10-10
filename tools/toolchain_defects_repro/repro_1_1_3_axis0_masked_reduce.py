#!/usr/bin/env python3
"""Defect 1.1-3: axis-0 masked reduction wrong / 507035 vector-core exception.

Doc claim: `tl.sum(tl.where(rows == j, g, 0.0), axis=0)` (reducing a 2D tile
along axis 0) either raises 507035 or gives a wrong result. Workaround: reduce
along axis 1 on the transpose (`tl.trans` then axis=1).

Here `pick` extracts row j of a 2D tile by:
  A) axis0 : tl.sum(tl.where(rows==j, g, 0.0), axis=0)     (suspect)
  B) trans : tl.sum(tl.where((trows==j)[:,None], tl.trans(g), 0.0), axis=0) on
             the transposed tile -- equivalently trans + axis=1   (workaround)
Both must equal g[j, :].

Run: python repro_1_1_3_axis0_masked_reduce.py [--repeat 20]
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--j", type=int, default=3)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    B, J = args.b, args.j

    @triton.jit
    def row_via_axis0(g_ptr, out_ptr, J: tl.constexpr, B: tl.constexpr):
        rows = tl.arange(0, B)
        cols = tl.arange(0, B)
        g = tl.load(g_ptr + rows[:, None] * B + cols[None, :])
        sel = tl.where(rows == J, g, 0.0)          # <-- suspect mask+reduce
        v = tl.sum(sel, axis=0)                    # axis-0 reduction
        tl.store(out_ptr + cols, v)

    @triton.jit
    def row_via_trans(g_ptr, out_ptr, J: tl.constexpr, B: tl.constexpr):
        rows = tl.arange(0, B)
        cols = tl.arange(0, B)
        g = tl.load(g_ptr + rows[:, None] * B + cols[None, :])
        gt = tl.trans(g)                            # [B, B] -> [B, B]
        sel = tl.where((cols == J)[:, None], gt, 0.0)
        v = tl.sum(sel, axis=0)                     # axis-1 on transpose
        tl.store(out_ptr + cols, v)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    torch.manual_seed(0)
    g = torch.randn(B, B, dtype=torch.float32, device=dev)
    ref = g[J, :].detach().to("cpu")

    def run(kernel):
        out = torch.empty(B, dtype=torch.float32, device=dev)
        kernel[(1,)](g, out, J=J, B=B)
        return out.detach().to("cpu")

    import traceback
    def safe(kernel):
        try:
            return run(kernel), None
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"

    bad0 = badt = exc0 = 0
    for _ in range(args.repeat):
        a, e0 = safe(row_via_axis0)
        b, et = safe(row_via_trans)
        exc0 += e0 is not None
        if a is None or not torch.allclose(a, ref, atol=1e-4):
            bad0 += 1
        if b is None or not torch.allclose(b, ref, atol=1e-4):
            badt += 1
    print(f"B={B} J={J} repeat={args.repeat}")
    print(f"  axis-0 masked reduce : wrong/err = {bad0}/{args.repeat} "
          f"(exceptions={exc0})")
    print(f"  trans + axis-1       : wrong/err = {badt}/{args.repeat}")
    if (bad0 or exc0) and not badt:
        print("REPRODUCED: axis-0 masked reduce is wrong/crashes; trans ok.")
        return 0
    if not (bad0 or exc0):
        print("Not reproduced (probabilistic/version-dependent).")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
