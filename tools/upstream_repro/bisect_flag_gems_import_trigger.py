#!/usr/bin/env python3
"""Bisect what triggers the Ascend nondeterminism observed via the diagnose tool.

Facts:
  * A triton-only script does NOT reproduce it.
  * The standalone repro with `--import-flaggems` DOES (per the reporter).
  * Importing flag_gems submodules progressively did NOT reproduce it.

So the trigger is not a single import; it is some *action* the diagnose tool
takes. This script runs the aliased kernel in a fresh subprocess under a matrix
of candidate triggers and reports which combination flips it from deterministic
to nondeterministic. Candidates (cumulative `--setup` levels):

  0 triton_only         import torch/triton, run kernel
  1 seed_cpu            + torch.manual_seed
  2 seed_device         + torch.npu.manual_seed_all (or torch_device_fn)
  3 import_flag_gems    + import flag_gems ; dev = flag_gems.device
  4 gems_device_seed    + torch_device_fn.manual_seed_all + default_generators
  5 use_gems_once       + with flag_gems.use_gems(): a trivial op, then kernel
  6 all                 everything above in diagnose-tool order

Run:

    python bisect_flag_gems_import_trigger.py --device 2 --repeat 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

SETUPS = [
    "triton_only",
    "seed_cpu",
    "seed_device",
    "all_blocks_parallel",
    "import_flag_gems",
    "gems_device_seed",
    "use_gems_once",
    "all",
]


# Env vars that flag_gems sets at import time (module-level), which leak into
# every later kernel compiled in the process. Listed for bisection.
LEAKED_ENV_VARS = [
    "TRITON_ALL_BLOCKS_PARALLEL",
]


def _setup(level: str) -> str:
    """Apply a setup level; return the device string to use."""
    import torch

    dev = "npu" if hasattr(torch, "npu") else (
        "cuda" if torch.cuda.is_available() else "cpu")

    if level == "triton_only":
        return dev

    torch.manual_seed(0)
    if level == "seed_cpu":
        return dev

    if level == "all_blocks_parallel":
        # Reproduce the single global flag that flag_gems sets at import.
        os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
        return dev

    if level in ("seed_device", "import_flag_gems", "gems_device_seed",
                 "use_gems_once", "all"):
        msa = None
        if dev == "npu" and hasattr(torch, "npu"):
            msa = getattr(torch.npu, "manual_seed_all", None)
        elif dev == "cuda":
            msa = getattr(torch.cuda, "manual_seed_all", None)
        if callable(msa):
            msa(0)
    if level == "seed_device":
        return dev

    import flag_gems  # noqa: F401
    from flag_gems.runtime import torch_device_fn

    dev = flag_gems.device
    if level == "import_flag_gems":
        return dev

    msa = getattr(torch_device_fn, "manual_seed_all", None)
    if callable(msa):
        msa(0)
    dg = getattr(torch_device_fn, "default_generators", None)
    if dg is not None:
        try:
            for gen in dg:
                gen.manual_seed(0)
        except Exception:  # noqa: BLE001
            pass
    if level == "gems_device_seed":
        return dev

    if level in ("use_gems_once", "all"):
        try:
            x = torch.randn(8, dtype=torch.float32, device=dev)
            with flag_gems.use_gems():
                _ = torch.mul(x, 2.0)
        except Exception:  # noqa: BLE001
            pass
    return dev


def run_level(level: str, repeat: int, n: int, block: int, scalar: float) -> int:
    import torch
    import triton
    import triton.language as tl

    dev = _setup(level)

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
    res = {"level": level, "dev": dev,
           "deterministic": len(set(hs)) == 1, "distinct": len(set(hs))}
    print(json.dumps(res, default=str))
    return 0 if res["deterministic"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--n", type=int, default=1024 * 1024)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--scalar", type=float, default=-0.999)
    ap.add_argument("--fresh-cache", action="store_true",
                    help="give every setup its own TRITON_CACHE_DIR / "
                         "FLAGGEMS_CACHE_DIR and set TRITON_ALWAYS_COMPILE=1, "
                         "so a reused compiled binary cannot confound the result")
    ap.add_argument("--level", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.level is not None:
        if args.device is not None:
            for var in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
                os.environ[var] = str(args.device)
        return run_level(args.level, args.repeat, args.n, args.block, args.scalar)

    import tempfile

    print(f"{'setup':20s} {'deterministic':>13s}  distinct  dev")
    trigger = None
    for level in SETUPS:
        env = os.environ.copy()
        if args.device is not None:
            for var in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
                env[var] = str(args.device)
        if args.fresh_cache:
            td = tempfile.mkdtemp(prefix=f"abdiag-{level}-")
            env["TRITON_CACHE_DIR"] = os.path.join(td, "triton")
            env["FLAGGEMS_CACHE_DIR"] = os.path.join(td, "fg")
            env["TRITON_ALWAYS_COMPILE"] = "1"
        cmd = [sys.executable, __file__, "--level", level,
               "--repeat", str(args.repeat), "--n", str(args.n),
               "--block", str(args.block), "--scalar", str(args.scalar)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        lines = proc.stdout.strip().splitlines()
        info = json.loads(lines[-1]) if lines and lines[-1].startswith("{") \
            else {"deterministic": None, "distinct": "ERR", "dev": "?"}
        print(f"{level:20s} {str(info.get('deterministic')):>13s}  "
              f"{info.get('distinct')}  {info.get('dev')}")
        if not lines or not lines[-1].startswith("{"):
            print("   stderr:", proc.stderr.strip().splitlines()[-1:] )
        if info.get("deterministic") is False and trigger is None:
            trigger = level
    print()
    if trigger:
        print(f"FIRST nondeterministic setup: {trigger}")
        print("Compare with the previous row to see which setup call flips it.")
    else:
        print("No setup reproduced it; the trigger needs another factor "
              "(e.g. a specific op compiled/run first, or a real gem op).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
