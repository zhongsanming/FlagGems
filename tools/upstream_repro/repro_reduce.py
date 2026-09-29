#!/usr/bin/env python3
"""Minimal repro: auto-blockify (TRITON_ALL_BLOCKS_PARALLEL=1) makes
tl.max(..., return_indices=True) / tl.argmax nondeterministic on Ascend.

    TRITON_ALL_BLOCKS_PARALLEL=1 python repro_reduce.py   # flaky, while
                                                          # tl.max/tl.sum are stable
    python repro_reduce.py                                # stable control

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
def max_idx(x_ptr, v_ptr, i_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + o, mask=o < n, other=float("-inf"))
    v, i = tl.max(x, axis=0, return_indices=True)   # nondeterministic
    tl.store(v_ptr + tl.program_id(0), v)
    tl.store(i_ptr + tl.program_id(0), i)


@triton.jit
def max_only(x_ptr, v_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + o, mask=o < n, other=float("-inf"))
    tl.store(v_ptr + tl.program_id(0), tl.max(x, axis=0))   # deterministic


n, BLOCK = 1 << 20, 2048
nb = triton.cdiv(n, BLOCK)
dev = "npu" if hasattr(torch, "npu") else "cuda"
base = torch.randn(n, dtype=torch.float32, device=dev)


def h(t):
    return hashlib.sha1(t.cpu().numpy().tobytes()).hexdigest()[:12]


for name in ("max_only", "max_idx"):
    for _ in range(10):
        v = torch.empty(nb, dtype=torch.float32, device=dev)
        i = torch.empty(nb, dtype=torch.int64, device=dev)
        if name == "max_only":
            max_only[(nb,)](base, v, n, BLOCK=BLOCK)
            out = v
        else:
            max_idx[(nb,)](base, v, i, n, BLOCK=BLOCK)
            out = torch.cat([v.view(torch.int32), i.to(torch.int32)])
        print(name, h(out))
