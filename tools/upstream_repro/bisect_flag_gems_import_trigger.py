#!/usr/bin/env python3
"""Bisect which part of `import flag_gems` triggers the Ascend nondeterminism.

Observations so far:
  * A triton-only script (no flag_gems) does NOT reproduce the nondeterminism.
  * Adding `import flag_gems` DOES reproduce it (aliased in-place elementwise,
    and tl.max(return_indices=True)) on
    flagtree 0.7.0+ascend.git0afb1367 / AscendNPU-IR 3545d1cb.

This helper imports flag_gems progressively and re-runs the aliased kernel after
each step, printing whether the kernel is deterministic. The first import that
flips it from deterministic to nondeterministic is the trigger.

Run (must be a fresh process per step; this script does that itself):

    python bisect_flag_gems_import_trigger.py --device 2
    python bisect_flag_gems_import_trigger.py --device 2 --repeat 20

It re-executes itself once per step in a subprocess (`--step N`) so that import
side effects from a previous step cannot leak into the next.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

# Ordered candidate imports. Each entry is (label, import statement). We import
# them cumulatively (the i-th step execs statements[0..i]) because some require
# predecessors (e.g. flag_gems.runtime before backends).
STEPS = [
    ("none", []),
    ("triton", ["import triton"]),
    ("torch_npu", ["import torch_npu"]),
    ("flag_gems.runtime", ["import flag_gems.runtime"]),
    ("flag_gems.testing", ["import flag_gems.testing"]),
    ("flag_gems.backend", ["import flag_gems.runtime.backend"]),
    ("flag_gems.ops", ["import flag_gems.ops"]),
    ("flag_gems", ["import flag_gems"]),
]


def _run_kernel(repeat: int, n: int, block: int, scalar: float, dev: str) -> dict:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def mul_inplace(x_ptr, s, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        x = tl.load(x_ptr + off, mask=m)
        tl.store(x_ptr + off, x * s, mask=m)

    def _hash(t):
        return hashlib.sha1(
            t.detach().to("cpu").contiguous().numpy().tobytes()
        ).hexdigest()[:16]

    torch.manual_seed(0)
    base = torch.randn(n, dtype=torch.float32, device=dev)
    grid = (triton.cdiv(n, block),)
    hs = []
    for _ in range(repeat):
        x = base.clone()
        mul_inplace[grid](x, scalar, n, BLOCK=block)
        hs.append(_hash(x))
    return {"deterministic": len(set(hs)) == 1, "distinct": len(set(hs))}


def run_step(step: int, repeat: int, n: int, block: int, scalar: float) -> int:
    label, imports = STEPS[step]
    for stmt in imports:
        exec(stmt, {})  # noqa: S102 - intentional cumulative import
    import torch

    dev = "npu" if hasattr(torch, "npu") else (
        "cuda" if torch.cuda.is_available() else "cpu")
    res = _run_kernel(repeat, n, block, scalar, dev)
    print(json.dumps({"step": step, "label": label,
                      "imports": imports, **res}, default=str))
    return 0 if res["deterministic"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--n", type=int, default=1024 * 1024)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--scalar", type=float, default=-0.999)
    ap.add_argument("--step", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.step is not None:
        if args.device is not None:
            for var in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
                os.environ[var] = str(args.device)
        return run_step(args.step, args.repeat, args.n, args.block, args.scalar)

    print(f"{'step':>4} {'label':30s} {'deterministic':>13s}  distinct")
    trigger = None
    for i, (label, _imports) in enumerate(STEPS):
        env = os.environ.copy()
        if args.device is not None:
            for var in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
                env[var] = str(args.device)
        cmd = [sys.executable, __file__, "--step", str(i),
               "--repeat", str(args.repeat), "--n", str(args.n),
               "--block", str(args.block), "--scalar", str(args.scalar)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        last = proc.stdout.strip().splitlines()
        info = json.loads(last[-1]) if last and last[-1].startswith("{") else \
            {"deterministic": None, "distinct": "ERR"}
        print(f"{i:>4} {label:30s} {str(info.get('deterministic')):>13s}  "
              f"{info.get('distinct')}")
        if info.get("deterministic") is False and trigger is None:
            trigger = label
            print(f"     -> first nondeterministic import: {label}")
    if trigger is None:
        print("\nNo step reproduced nondeterminism; the trigger is not one of "
              "these imports alone (may need torch.manual_seed/device seeding "
              "as the diagnose tool does).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
