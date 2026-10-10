#!/usr/bin/env python3
"""Defects 1.4-14 / 1.4-15: resource & dtype compile limits on Ascend.

1.4-14 (UB 192KB wall): a (128,128) fp32 tile is not compilable, while
(64,128)/(128,64) are. We try a few tile shapes and report which fail.
1.4-15 (fp64 uncompilable): any fp64 kernel fails at compile with
MLIRCompilationError.

Run: python repro_1_4_ub_and_fp64.py
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    # A trivial kernel that just does an elementwise op over a (BM,BN) fp32 tile.
    @triton.jit
    def tile_op(x_ptr, out_ptr, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        x = tl.load(x_ptr + rm[:, None] * BN + rn[None, :])
        tl.store(out_ptr + rm[:, None] * BN + rn[None, :], x + 1.0)

    @triton.jit
    def fp64_op(x_ptr, out_ptr, B: tl.constexpr):
        ar = tl.arange(0, B)
        x = tl.load(x_ptr + ar)
        tl.store(out_ptr + ar, x + 1.0)

    dev = "npu" if hasattr(torch, "npu") else "cuda"

    def try_tile(BM, BN):
        try:
            x = torch.zeros(BM * BN, dtype=torch.float32, device=dev)
            out = torch.zeros(BM * BN, dtype=torch.float32, device=dev)
            tile_op[(1,)](x, out, BM=BM, BN=BN)
            return "ok"
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {str(e)[:120]}"

    print("1.4-14 UB / tile-size compilability (fp32):")
    for BM, BN in [(64, 64), (64, 128), (128, 64), (128, 128), (256, 64)]:
        print(f"  tile ({BM:3d},{BN:3d}): {try_tile(BM, BN)}")

    print("\n1.4-15 fp64 compilability:")
    try:
        x = torch.zeros(64, dtype=torch.float64, device=dev)
        out = torch.zeros(64, dtype=torch.float64, device=dev)
        fp64_op[(1,)](x, out, B=64)
        print("  fp64 kernel: ok (unexpected)")
    except Exception as e:  # noqa: BLE001
        print(f"  fp64 kernel: {type(e).__name__}: {str(e)[:160]}")
        print("  (doc: expect MLIRCompilationError -> fp64 unsupported)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
