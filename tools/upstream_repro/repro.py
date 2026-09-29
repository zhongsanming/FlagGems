#!/usr/bin/env python3
"""Minimal repro: auto-blockify (TRITON_ALL_BLOCKS_PARALLEL=1) makes an in-place
triton kernel nondeterministic on Ascend.

    TRITON_ALL_BLOCKS_PARALLEL=1 python repro.py   # flaky (different hashes)
    python repro.py                                # stable control (one hash)

flagtree 0.7.0+ascend.git0afb1367, AscendNPU-IR 3545d1cb, triton 3.5.1,
torch-npu 2.9.0.post2, Ascend 910B.
"""

import hashlib
import os

import torch
import triton
import triton.language as tl

os.environ["TRITON_ALWAYS_COMPILE"] = "1"  # flag is not part of the cache key


@triton.jit
def k(x_ptr, s, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tl.store(x_ptr + o, tl.load(x_ptr + o, mask=m) * s, mask=m)   # x = x * s


n, BLOCK = 1 << 20, 1024
dev = "npu" if hasattr(torch, "npu") else "cuda"
base = torch.randn(n, dtype=torch.float32, device=dev)
for _ in range(10):
    x = base.clone()
    k[(triton.cdiv(n, BLOCK),)](x, -0.999, n, BLOCK=BLOCK)
    print(hashlib.sha1(x.cpu().numpy().tobytes()).hexdigest()[:12])
