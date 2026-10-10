#!/usr/bin/env python3
"""Defect 1.1-2: reduction-extracted vector in `[:,None]` broadcast miscompiles.

Doc claim: `v[:,None] * w[None,:]` where v, w come from a masked *reduction*
produces wrong outer products (6/6 wrong, error ~10); the same code with plain
`tl.load` vectors is correct (6/6 right). Workaround: `tl.reshape(v,(B,1)) *
tl.reshape(w,(1,B))`.

This kernel builds v, w by reducing 2D tiles along axis 0 (row sums), then forms
the outer product two ways:
  A) broadcast  : v[:,None] * w[None,:]     (suspect)
  B) reshape    : reshape(v,(B,1))*reshape(w,(1,B))  (workaround)
Both must equal the same reference (outer product of the row sums).

Run: python repro_1_1_2_reduce_broadcast.py [--repeat 6] [--bm 8] [--bn 8]
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--bm", type=int, default=8)
    ap.add_argument("--bn", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    BM, BN = args.bm, args.bn

    @triton.jit
    def outer_broadcast(x_ptr, out_ptr, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        tile = tl.load(x_ptr + rm[:, None] * BN + rn[None, :])
        v = tl.sum(tile, axis=1)          # reduce -> vector
        w = tl.sum(tile, axis=0)          # reduce -> vector
        out = v[:, None] * w[None, :]     # <-- suspect
        tl.store(out_ptr + rm[:, None] * BN + rn[None, :], out)

    @triton.jit
    def outer_reshape(x_ptr, out_ptr, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        tile = tl.load(x_ptr + rm[:, None] * BN + rn[None, :])
        v = tl.sum(tile, axis=1)
        w = tl.sum(tile, axis=0)
        vv = tl.reshape(v, (BM, 1))
        ww = tl.reshape(w, (1, BN))
        out = vv * ww                      # <-- workaround
        tl.store(out_ptr + rm[:, None] * BN + rn[None, :], out)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    torch.manual_seed(0)
    x = torch.randn(BM, BN, dtype=torch.float32, device=dev)
    v = x.sum(dim=1).to("cpu")
    w = x.sum(dim=0).to("cpu")
    ref = torch.outer(v, w)

    def run(kernel):
        out = torch.empty(BM, BN, dtype=torch.float32, device=dev)
        kernel[(1,)](x, out, BM=BM, BN=BN)
        return out.detach().to("cpu")

    bad_b = bad_r = 0
    for _ in range(args.repeat):
        if not torch.allclose(run(outer_broadcast), ref, atol=1e-4):
            bad_b += 1
        if not torch.allclose(run(outer_reshape), ref, atol=1e-4):
            bad_r += 1
    print(f"BM={BM} BN={BN} repeat={args.repeat}")
    print(f"  v[:,None]*w[None,:] (broadcast): wrong = {bad_b}/{args.repeat}")
    print(f"  reshape workaround             : wrong = {bad_r}/{args.repeat}")
    if bad_b and not bad_r:
        print("REPRODUCED: reduce-vector broadcast is wrong; reshape ok.")
        return 0
    if not bad_b:
        print("Not reproduced (probabilistic; try other --bm/--bn).")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
