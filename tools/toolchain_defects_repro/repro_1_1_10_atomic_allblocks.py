#!/usr/bin/env python3
"""Defect 1.1-10: `TRITON_ALL_BLOCKS_PARALLEL=1` makes atomic_add sums wrong.

Doc claim: with TRITON_ALL_BLOCKS_PARALLEL set (globally, by another module at
import), a matvec/apply kernel's `tl.atomic_add` accumulation produces a wrong
sum. Workaround: unset the variable while compiling that kernel.

This kernel accumulates a vector into a single output with atomic_add; each
program adds one element. Reference = plain sum.
  A) with the env var set   -> run this whole script with TRITON_ALL_BLOCKS_PARALLEL=1
  B) without it             -> run normally

Run:
    python repro_1_1_10_atomic_allblocks.py                 # expect correct
    TRITON_ALL_BLOCKS_PARALLEL=1 python repro_1_1_10_atomic_allblocks.py   # suspect

For a single-process A/B, pass --toggle to also flip the env var between two
compilations in the same process (uses a distinct cache dir per run).
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--device", default=None)
    ap.add_argument("--toggle", action="store_true",
                    help="run both with and without the env var in one process")
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def atomic_sum(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        ar = pid * BLOCK + tl.arange(0, BLOCK)
        m = ar < n
        x = tl.load(x_ptr + ar, mask=m, other=0.0)
        tl.atomic_add(out_ptr, tl.sum(x, axis=0))

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    torch.manual_seed(0)
    n = args.n
    BLOCK = 256
    x = torch.randn(n, dtype=torch.float32, device=dev)
    ref = float(x.detach().to("cpu").double().sum())

    def run():
        out = torch.zeros(1, dtype=torch.float32, device=dev)
        atomic_sum[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
        return float(out.item())

    def check(label):
        bad = 0
        vals = []
        for _ in range(args.repeat):
            v = run()
            vals.append(v)
            # atomic_add reorders summation -> fp error is expected and small;
            # the defect shows as LARGE error, not rounding.
            if abs(v - ref) > max(1e-2, 1e-3 * abs(ref)):
                bad += 1
        print(f"  {label:24s} wrong={bad}/{args.repeat}  "
              f"sample={vals[:3]}  ref={ref:.4f}")
        return bad

    abp = os.environ.get("TRITON_ALL_BLOCKS_PARALLEL")
    print(f"n={n} BLOCK={BLOCK} repeat={args.repeat} "
          f"TRITON_ALL_BLOCKS_PARALLEL={abp!r}")
    bad = check(f"abp={abp!r}")
    if args.toggle:
        # recompile with the other setting (distinct cache dir)
        import tempfile
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp()
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "0"
        bad2 = check("abp='0'")
        if bad and not bad2:
            print("REPRODUCED: atomic_add wrong with auto-blockify on, ok off.")
            return 0
        return 1 if bad2 else 0
    if bad:
        print("FAILURES present -> suspect reproduced (compare with abp unset).")
        return 0
    print("No large errors this run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
