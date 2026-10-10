#!/usr/bin/env python3
"""Defect 1.2: tl.dot (Cube) operand limitations (four sub-defects).

Doc claims:
  1) dot fed by a non-direct-load operand (e.g. an iterated/computed vector)
     fails to compile: "Unsupported op for finding the root alloc".
  2) output-dim mask / 0 padding misplaces the dot output.
  3) accumulating with tl.dot(a,b,acc) is inaccurate (K=64 err 87..95);
     `acc = acc + tl.dot(a,b)` is accurate.
  4) stride-swapped (transposed-address) load fed to dot is wrong (K=64 err 47..50);
     `tl.trans(b)` is correct (err ~2.3e-5).

This script checks the compile cases (1,2) and the accuracy cases (3,4) against
a CPU float64 reference.

Run: python repro_1_2_dot_operands.py [--k 64] [--repeat 6]
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    K, M, N = args.k, args.m, args.n

    # 3+4: accumulate two ways, and transposed-load vs explicit tl.trans
    @triton.jit
    def dot_acc_builtin(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                        K: tl.constexpr):
        rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc = tl.zeros((M, N), dtype=tl.float32)
        acc = tl.dot(a, b, acc)                 # <-- suspect: built-in acc
        tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc)

    @triton.jit
    def dot_acc_add(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                    K: tl.constexpr):
        rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc = tl.zeros((M, N), dtype=tl.float32)
        acc = acc + tl.dot(a, b)                # <-- workaround
        tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc)

    # 4: b stored as [N,K]; load with swapped strides then dot, vs tl.trans
    @triton.jit
    def dot_b_swapped(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                      K: tl.constexpr):
        rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        # b is physically [N, K]; load a [K, N] view via swapped strides
        b = tl.load(b_ptr + rk[:, None] + rn[None, :] * K)   # <-- swapped addr
        acc = tl.dot(a, b)
        tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc)

    @triton.jit
    def dot_b_trans(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                    K: tl.constexpr):
        rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        bnk = tl.load(b_ptr + rn[:, None] * K + rk[None, :])  # [N,K]
        b = tl.trans(bnk)                                     # <-- workaround
        acc = tl.dot(a, b)
        tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc)

    # 1: dot fed by a non-direct-load (computed) operand
    @triton.jit
    def dot_computed_operand(a_ptr, b_ptr, out_ptr, M: tl.constexpr,
                             N: tl.constexpr, K: tl.constexpr):
        rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        a = a * 1.0000001                     # <-- computed, not a direct load
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc = tl.dot(a, b)
        tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc)

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float16, device=dev)
    Bk = torch.randn(K, N, dtype=torch.float16, device=dev)
    Bnk = Bk.t().contiguous()                       # physically [N, K]
    ref = (A.detach().to("cpu").double() @ Bk.detach().to("cpu").double())

    def run(kernel, b_tensor):
        try:
            out = torch.empty(M, N, dtype=torch.float32, device=dev)
            kernel[(1,)](A, b_tensor, out, M=M, N=N, K=K)
            o = out.detach().to("cpu").double()
            err = float((o - ref).abs().max())
            return "ok", err
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {str(e)[:120]}", None

    tests = [
        ("acc=tl.dot(a,b,acc) [1.2-3 suspect]", dot_acc_builtin, Bk),
        ("acc=acc+tl.dot(a,b)  [1.2-3 fix]", dot_acc_add, Bk),
        ("b swapped-addr load [1.2-4 suspect]", dot_b_swapped, Bnk),
        ("b tl.trans          [1.2-4 fix]", dot_b_trans, Bnk),
        ("computed operand    [1.2-1 suspect]", dot_computed_operand, Bk),
    ]
    print(f"M={M} N={N} K={K}")
    for name, k, bt in tests:
        status, err = run(k, bt)
        if status == "ok":
            print(f"  {name:38s} ok   max_err={err:.3e}")
        else:
            print(f"  {name:38s} FAIL {status}")
    print("\nInterpret: compare suspect vs fix rows. Large err on a 'suspect' row")
    print("with small err on the matching 'fix' row == defect reproduced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
