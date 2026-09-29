#!/usr/bin/env python3
"""Minimal reproducer: tl.max(..., return_indices=True) is nondeterministic.

Bug summary
-----------
On Ascend (AscendNPU-IR build bundled with flagtree 0.7.0+ascend.git0afb1367,
i.e. AscendNPU-IR revision 3545d1cba9b1bdd3cb300724d4626282ad1679ee), a block
reduction that requests the index of the maximum

    v, i = tl.max(x, axis=0, return_indices=True)

returns a DIFFERENT value/index on every run of an identical input. The plain
reductions ``tl.max(x, axis=0)``, ``tl.sum(x, axis=0)`` and (used below)
``tl.min`` are all deterministic, so the fault is specific to the
reduce-with-index combiner. ``tl.argmax`` uses the same combiner and is affected
too; it is what makes the two-pass global argmax (argmax_kernel_1 ->
argmax_kernel_2) return an intermittent wrong index.

Scope / ownership
-----------------
This script uses ONLY torch + triton, no FlagGems; the kernels are plain
``@triton.jit`` functions, so the nondeterminism is produced by the
triton/flagtree -> AscendNPU-IR compilation+execution path. flagtree's linalg
lowering of ``tt.reduce`` is deterministic in form; the wrong result appears
after ``bishengir-compile`` (AscendNPU-IR), pointing at its reduction lowering
(per-block reduction over the vector core / the index combiner).

Controls in this script
-----------------------
    reduce_max        tl.max(x)                     -> expect deterministic
    reduce_sum        tl.sum(x)                     -> expect deterministic
    reduce_max_idx    tl.max(x, return_indices=True) -> expect NONDETERMINISTIC
    reduce_argmax     tl.argmax(x)                  -> expect NONDETERMINISTIC
    reduce_max_where  index via tl.min(where(x==max)) -> expect deterministic
                      (this is the workaround shape; safe primitives only)

Run
---
    python repro_reduce_return_indices_nondeterminism.py
    python repro_reduce_return_indices_nondeterminism.py --repeat 20 --n 1048576
    python repro_reduce_return_indices_nondeterminism.py --device 2

Expected: reduce_max_where / reduce_max / reduce_sum deterministic; reduce_max_idx
and reduce_argmax nondeterministic (multiple distinct hashes).
"""

from __future__ import annotations

import argparse
import hashlib
import os


def env_fingerprint() -> dict:
    import importlib.metadata as md

    fp = {}
    try:
        import triton

        fp["triton_version"] = getattr(triton, "__version__", None)
        fp["triton_file"] = getattr(triton, "__file__", None)
    except Exception as exc:  # noqa: BLE001
        fp["triton_version"] = f"ERR:{exc}"
    for pkg in ("flagtree", "torch", "torch-npu"):
        try:
            fp[f"{pkg}_version"] = md.version(pkg)
        except Exception:  # noqa: BLE001
            fp[f"{pkg}_version"] = None
    return fp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1024 * 1024)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--device", default=None,
                    help="accelerator id to confine to (sets ASCEND_RT_VISIBLE_DEVICES)")
    ap.add_argument("--import-flaggems", action="store_true",
                    help="import flag_gems before running (the diagnose tool does; "
                         "use to test whether the bug depends on flag_gems setup)")
    args = ap.parse_args()

    if args.device is not None:
        for var in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[var] = str(args.device)

    import json

    import torch
    import triton
    import triton.language as tl

    dev = "npu" if hasattr(torch, "npu") else (
        "cuda" if torch.cuda.is_available() else "cpu")
    if args.import_flaggems:
        import flag_gems  # noqa: F401
        from flag_gems.runtime import torch_device_fn

        try:
            torch_device_fn.manual_seed_all(0)
        except Exception:  # noqa: BLE001
            pass
        dev = flag_gems.device

    # ---- kernels -----------------------------------------------------------
    @triton.jit
    def k_reduce_max(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + off, mask=off < n, other=float("-inf"))
        tl.store(out_ptr + pid, tl.max(x, axis=0))

    @triton.jit
    def k_reduce_sum(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + off, mask=off < n, other=0.0)
        tl.store(out_ptr + pid, tl.sum(x, axis=0))

    @triton.jit
    def k_reduce_max_idx(x_ptr, v_ptr, i_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + off, mask=off < n, other=float("-inf"))
        v, i = tl.max(x, axis=0, return_indices=True)
        tl.store(v_ptr + pid, v)
        tl.store(i_ptr + pid, i)

    @triton.jit
    def k_reduce_argmax(x_ptr, i_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + off, mask=off < n, other=float("-inf"))
        tl.store(i_ptr + pid, tl.argmax(x, axis=0))

    @triton.jit
    def k_reduce_max_where(x_ptr, v_ptr, i_ptr, n, BLOCK: tl.constexpr):
        # Deterministic workaround: max value via plain tl.max, then the first
        # lane equal to it via plain tl.min (no index combiner).
        pid = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        off = pid * BLOCK + lane
        x = tl.load(x_ptr + off, mask=off < n, other=float("-inf"))
        v = tl.max(x, axis=0)
        cand = tl.where(x == v, lane, BLOCK)
        i = tl.min(cand, axis=0)
        tl.store(v_ptr + pid, v)
        tl.store(i_ptr + pid, i)

    def _hash(t):
        return hashlib.sha1(
            t.detach().to("cpu").contiguous().numpy().tobytes()
        ).hexdigest()[:16]

    torch.manual_seed(0)
    base = torch.randn(args.n, dtype=torch.float32, device=dev)
    nblocks = triton.cdiv(args.n, args.block)
    print(json.dumps({"env": env_fingerprint(), "n": args.n,
                      "block": args.block, "nblocks": nblocks,
                      "repeat": args.repeat}, indent=2))

    def outputs(kind):
        v = torch.empty(nblocks, dtype=torch.float32, device=dev)
        i = torch.empty(nblocks, dtype=torch.int64, device=dev)
        grid = (nblocks,)
        if kind == "max":
            k_reduce_max[grid](base, v, args.n, BLOCK=args.block)
            return v
        if kind == "sum":
            k_reduce_sum[grid](base, v, args.n, BLOCK=args.block)
            return v
        if kind == "idx":
            k_reduce_max_idx[grid](base, v, i, args.n, BLOCK=args.block)
            return torch.cat([v.view(torch.int32), i.to(torch.int32)])
        if kind == "argmax":
            k_reduce_argmax[grid](base, i, args.n, BLOCK=args.block)
            return i
        if kind == "where":
            k_reduce_max_where[grid](base, v, i, args.n, BLOCK=args.block)
            return torch.cat([v.view(torch.int32), i.to(torch.int32)])
        raise ValueError(kind)

    results = {}
    for name, kind in (("reduce_max", "max"), ("reduce_sum", "sum"),
                       ("reduce_max_idx", "idx"), ("reduce_argmax", "argmax"),
                       ("reduce_max_where", "where")):
        hs = [_hash(outputs(kind)) for _ in range(args.repeat)]
        results[name] = {"deterministic": len(set(hs)) == 1,
                         "distinct": len(set(hs))}
        print(f"[{name:17s}] deterministic={results[name]['deterministic']} "
              f"distinct={results[name]['distinct']}/{args.repeat}")

    reproduced = (results["reduce_max"]["deterministic"]
                  and results["reduce_sum"]["deterministic"]
                  and not results["reduce_max_idx"]["deterministic"])
    if reproduced:
        print("\nREPRODUCED: tl.max/tl.sum are deterministic; "
              "tl.max(return_indices=True) is not.")
    else:
        print("\nDid NOT reproduce the expected pattern this run; increase "
              "--repeat or check the compiler/device.")
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
