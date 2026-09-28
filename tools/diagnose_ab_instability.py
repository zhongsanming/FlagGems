#!/usr/bin/env python3
"""Diagnose A/B accuracy instability for a single op (default: argmax, mul_).

Bisects "the result is unstable" into one of:

  (I)  INPUT_NONDETERMINISM   the seeded input differs between two fresh
                              constructions -> RNG seeding is not effective.
  (II) KERNEL_NONDETERMINISM  the SAME input yields different gem outputs
                              across repeated in-process runs -> the kernel
                              (or its launch/sync) is nondeterministic.
  (III)CONFIG_DEPENDENCE      ON and OFF disagree on the SAME input -> the
                              compiler pass changes numerics (should be
                              impossible if TTIR is identical).
  (IV) REFERENCE_TOLERANCE    gem is deterministic and matches itself, the
                              reference is the moving part.

For argmax it additionally reports whether the failure is a max-value TIE
(the classic argmax tie-break mismatch), by counting how many elements share
the global max.

Run on the NPU host, e.g.:

    python tools/diagnose_ab_instability.py --op argmax --config on
    python tools/diagnose_ab_instability.py --op mul_   --config off
    python tools/diagnose_ab_instability.py --op mul_ --config both --repeat 5
    python tools/diagnose_ab_instability.py --op argmax --both-configs
    python tools/diagnose_ab_instability.py --op mul_ --repeat 10 --sync
    python tools/diagnose_ab_instability.py --triton-probe   # backend-only

Respects FLAG_GEMS_SEED (like the A/B harness). Prints a JSON verdict.

The diagnostic also runs an out-of-place gem path for mul_ and reports
`oop_deterministic` / `ALIASING_SUSPECT`: in-place is nondeterministic while
out-of-place is stable points at an aliased load/store lowering bug. Use
--triton-probe to confirm it with a bare triton kernel and multibuffer on/off.

--sync inserts accelerator synchronizations around each gem run. If the
instability disappears with --sync but is present without it, the cause is a
missing launch/stream sync (e.g. an async input producer racing an in-place
kernel), not the kernel math.

--------------------------------------------------------------------------
Harness-level bisection (per-test RNG fingerprint)
--------------------------------------------------------------------------
To capture fingerprints for a whole A/B run, export FLAG_GEMS_AB_DIAG=1 before
launching tools/run_ab_interleaved.py. The runner then writes one JSONL per
(config, NPU) under <out>/<config>/diag/, with one record per test:

    {"time":..., "nodeid":...,
     "fingerprint": {"cpu_seed":..., "dev_seed":..., "dev_state":"..."}}

Then compare a test that flips between attempt/retry or between ON/OFF:

  * same fingerprint, different outcome  -> KERNEL (or launch) nondeterminism,
    because the seeded input was identical.
  * different fingerprint               -> the seed is not reaching the input
    generator (INPUT_NONDETERMINISM); fix seeding, not the kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


SEED = int(os.environ.get("FLAG_GEMS_SEED", "0"))


def _seed(seed: int) -> None:
    import random

    import torch

    from flag_gems.runtime import torch_device_fn

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    msa = getattr(torch_device_fn, "manual_seed_all", None)
    if callable(msa):
        msa(seed)
    dg = getattr(torch_device_fn, "default_generators", None)
    if dg is not None:
        try:
            for gen in dg:
                gen.manual_seed(seed)
        except TypeError:
            dg[torch_device_fn.current_device()].manual_seed(seed)


def _hash_tensor(t) -> str:
    import torch

    t = t.detach().to("cpu").contiguous()
    return hashlib.sha1(t.numpy().tobytes()).hexdigest()[:16]


def _argmax_input(shape, dtype, device):
    import torch

    _seed(SEED)
    return torch.randn(shape, dtype=dtype, device=device)


def _mul_input(shape, dtype, device):
    import torch

    _seed(SEED)
    return torch.randn(shape, dtype=dtype, device=device)


def diagnose(op: str, shape, dtype, scalar, device, repeat: int,
             sync: bool = False) -> dict:
    import torch

    import flag_gems
    from flag_gems.runtime import torch_device_fn

    def _sync():
        if sync:
            fn = getattr(torch_device_fn, "synchronize", None)
            if callable(fn):
                fn()

    out: dict = {"op": op, "shape": list(shape), "dtype": str(dtype),
                 "scalar": scalar, "seed": SEED, "repeat": repeat}

    # ---- (I) input determinism under the seed --------------------------------
    if op == "argmax":
        a = _argmax_input(shape, dtype, device)
        b = _argmax_input(shape, dtype, device)
    else:
        a = _mul_input(shape, dtype, device)
        b = _mul_input(shape, dtype, device)
    out["input_hash_1"] = _hash_tensor(a)
    out["input_hash_2"] = _hash_tensor(b)
    out["input_deterministic"] = out["input_hash_1"] == out["input_hash_2"]

    # ---- reference (CPU), deterministic by construction ----------------------
    ref_in = a.to("cpu")
    if op == "argmax":
        ref = torch.argmax(ref_in, dim=None, keepdim=False)
        out["ref_scalar"] = int(ref)
        # tie analysis: how many elements equal the global max
        mx = ref_in.reshape(-1).max()
        out["n_ties_at_max"] = int((ref_in.reshape(-1) == mx).sum())
        out["max_value"] = float(mx)
    else:
        ref = ref_in.double().mul_(scalar)

    # ---- (II) kernel determinism: same input, repeated gem runs --------------
    gem_hashes = []
    gem_vals = []
    last_res = None
    for _ in range(repeat):
        inp = a.clone()
        _sync()
        with flag_gems.use_gems():
            if op == "argmax":
                res = torch.argmax(inp, dim=None, keepdim=False)
            else:
                res = inp.mul_(scalar)
        _sync()
        last_res = res
        gem_hashes.append(_hash_tensor(res))
        gem_vals.append(int(res) if op == "argmax" else None)

    # For mul_: if the outputs differ, characterise the mismatch against the
    # reference (unwritten-element signature = ~1.999*|input|).
    if op == "mul_" and last_res is not None:
        r = last_res.to("cpu")
        refc = ref.to(r.dtype)
        diff = (r.float() - refc.float()).abs()
        nbad = int((~torch.isclose(r.float(), refc.float(),
                                   atol=1e-4, rtol=0.016)).sum())
        out["n_mismatch_last"] = nbad
        if nbad:
            idx = int(diff.reshape(-1).argmax())
            out["worst_index"] = idx
            out["worst_input"] = float(a.reshape(-1)[idx].to("cpu"))
            out["worst_gem"] = float(r.reshape(-1)[idx])
            out["worst_ref"] = float(refc.reshape(-1)[idx])

    out["gem_hashes"] = gem_hashes
    out["gem_scalars"] = gem_vals
    out["kernel_deterministic"] = len(set(gem_hashes)) == 1

    # ---- mul_ aliasing probe: in-place vs out-of-place ----------------------
    # An in-place elementwise kernel aliases x_ptr == output_ptr, which some
    # backends lower incorrectly (load/store reordering), leaving elements
    # unmultiplied. Compare against the out-of-place gem path on the same input.
    if op == "mul_":
        oop_hashes = []
        for _ in range(repeat):
            inp = a.clone()
            _sync()
            with flag_gems.use_gems():
                res = torch.mul(inp, scalar)  # out-of-place, fresh output
            _sync()
            oop_hashes.append(_hash_tensor(res))
        out["oop_hashes"] = oop_hashes
        out["oop_deterministic"] = len(set(oop_hashes)) == 1
        if oop_hashes and last_res is not None:
            out["inplace_matches_oop_last"] = (
                gem_hashes[-1] == oop_hashes[-1]
            )

    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", default="argmax", choices=["argmax", "mul_"])
    ap.add_argument("--shape", default=None,
                    help="comma list; default argmax=200,2560,3 / mul_=1024,1024")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--scalar", type=float, default=-0.999)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--sync", action="store_true",
                    help="insert torch_device_fn.synchronize() around each gem "
                         "run; if instability disappears, the cause is a "
                         "missing launch sync (input-producer vs in-place race)")
    ap.add_argument("--config", default=None,
                    help="informational only; env must already be set")
    ap.add_argument("--both-configs", action="store_true",
                    help="re-exec this script under ON and OFF env and compare "
                         "the deterministic-input gem output hashes")
    ap.add_argument("--compare", metavar="A,B",
                    help="compare two configs in child processes, e.g. "
                         "'off,off' (cross-process determinism) or 'on,off' "
                         "(config dependence); A and B in {on,off}")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--analyze-diag", metavar="GLOB",
                    help="analyze per-test fingerprint JSONL files produced with "
                         "FLAG_GEMS_AB_DIAG=1 (e.g. '<out>/*/diag/*.jsonl>')")
    ap.add_argument("--triton-probe", action="store_true",
                    help="minimal aliased in-place triton kernel, run with "
                         "multibuffer on/off, to isolate an Ascend auto "
                         "multi-buffer bug independent of FlagGems")
    args = ap.parse_args(argv)

    if args.triton_probe:
        return _triton_probe(args)

    if args.analyze_diag:
        return _analyze_diag(args.analyze_diag)

    if args.compare:
        a, _, b = args.compare.partition(",")
        return _run_pair(a.strip(), b.strip(), args)

    if args.both_configs and not args.child:
        return _run_both_configs(args)

    import torch

    import flag_gems

    device = flag_gems.device
    dt = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[args.dtype]
    if args.shape:
        shape = tuple(int(x) for x in args.shape.split(","))
    else:
        shape = (200, 2560, 3) if args.op == "argmax" else (1024, 1024)

    env = {k: os.environ.get(k) for k in (
        "FLAG_GEMS_SEED", "TRITON_DISABLE_LANE_VECTORIZE",
        "TRITON_ENABLE_LANE_VECTORIZE_BLOCK_MODE",
        "TRITON_LANE_VECTORIZE_ALLOW_CONCAT",
        "TRITON_LANE_VECTORIZE_ALLOW_ADDRESS_CONES",
        "FLAGGEMS_CACHE_DIR")}
    res = diagnose(args.op, shape, dt, args.scalar, device, args.repeat,
                   sync=args.sync)
    res["config_label"] = args.config
    res["env"] = env

    verdict = []
    if not res["input_deterministic"]:
        verdict.append("INPUT_NONDETERMINISM (seed not effective)")
    if not res["kernel_deterministic"]:
        verdict.append("KERNEL_NONDETERMINISM (same input -> different gem out)")
    if res.get("oop_deterministic") is True and not res["kernel_deterministic"]:
        verdict.append("ALIASING_SUSPECT (in-place nondeterministic but "
                       "out-of-place gem path is stable)")
    if res.get("oop_deterministic") is False:
        verdict.append("OOP_ALSO_NONDETERMINISTIC")
    if res.get("n_ties_at_max", 0) > 1:
        verdict.append(f"MAX_TIE (n_ties={res['n_ties_at_max']})")
    if not verdict:
        verdict.append("OK on this config/dtype/shape")
    res["verdict"] = verdict

    print(json.dumps(res, indent=2, default=str))
    return 0


CONFIG_ENV = {
    "off": {"TRITON_DISABLE_LANE_VECTORIZE": "1"},
    "on": {
        "TRITON_DISABLE_LANE_VECTORIZE": "0",
        "TRITON_ENABLE_LANE_VECTORIZE_BLOCK_MODE": "1",
        "TRITON_LANE_VECTORIZE_ALLOW_CONCAT": "1",
        "TRITON_LANE_VECTORIZE_ALLOW_ADDRESS_CONES": "0",
    },
}


def _extract_json(text: str):
    """Return the first JSON object in text, tolerating leading noise.

    Backend warnings (e.g. "[WARNING] triton.backends.ascend.utils not found")
    can be printed to stdout before our report, so a plain json.loads fails.
    """
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj
    except json.JSONDecodeError:
        return None


def _run_pair(label_a: str, label_b: str, args) -> int:
    """Run the child under two configs (may be the same) and diff results.

    'off,off' isolates cross-process nondeterminism (same compiler config).
    'on,off' isolates config dependence.
    """
    import subprocess
    import tempfile

    if label_a not in CONFIG_ENV or label_b not in CONFIG_ENV:
        print(f"configs must be in {sorted(CONFIG_ENV)}", file=sys.stderr)
        return 2
    results = {}
    with tempfile.TemporaryDirectory(prefix="abdiag-") as td:
        for label, tag in ((label_a, "A"), (label_b, "B")):
            env = os.environ.copy()
            env.update(CONFIG_ENV[label])
            env["TRITON_ALWAYS_COMPILE"] = "1"
            env["TRITON_CACHE_DIR"] = os.path.join(td, tag, "triton")
            env["FLAGGEMS_CACHE_DIR"] = os.path.join(td, tag, "fg")
            cmd = [sys.executable, __file__, "--child", "--config", f"{label}:{tag}",
                   "--op", args.op, "--dtype", args.dtype,
                   "--scalar", str(args.scalar), "--repeat", str(args.repeat)]
            if args.shape:
                cmd += ["--shape", args.shape]
            if args.sync:
                cmd += ["--sync"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            results[tag] = _extract_json(proc.stdout) or {
                "error": proc.stdout + proc.stderr}

    a, b = results.get("A", {}), results.get("B", {})
    if "error" in a or "error" in b:
        print(json.dumps({k: results[k] for k in results}, indent=2,
                         default=str))
        print("VERDICT: child run failed (see 'error' above); cannot compare",
              file=sys.stderr)
        return 1
    report = {
        "op": args.op, "dtype": args.dtype, "shape": args.shape,
        "A_config": label_a, "B_config": label_b,
        "input_hash_A": a.get("input_hash_1"),
        "input_hash_B": b.get("input_hash_1"),
        "gem_hashes_A": a.get("gem_hashes"),
        "gem_hashes_B": b.get("gem_hashes"),
        "input_matches": a.get("input_hash_1") == b.get("input_hash_1"),
        "gem_matches": a.get("gem_hashes", [])[:1] == b.get("gem_hashes", [])[:1],
        "kernel_deterministic_A": a.get("kernel_deterministic"),
        "kernel_deterministic_B": b.get("kernel_deterministic"),
    }
    print(json.dumps(report, indent=2, default=str))

    if not report["input_matches"]:
        print("VERDICT: INPUT differs across processes -> fix RNG seeding",
              file=sys.stderr)
    elif not (report["kernel_deterministic_A"] and report["kernel_deterministic_B"]):
        bad = [n for n, ok in (("A", report["kernel_deterministic_A"]),
                               ("B", report["kernel_deterministic_B"])) if not ok]
        print(f"VERDICT: KERNEL/LAUNCH_NONDETERMINISM ({'+'.join(bad)}): the same "
              "input produced different outputs across repeated runs in the "
              "SAME process/config -> this is not an RNG or LaneVectorize issue. "
              "Re-run with --sync to test a missing launch/stream sync.",
              file=sys.stderr)
    elif not report["gem_matches"]:
        if label_a == label_b:
            print("VERDICT: CROSS_PROCESS_NONDETERMINISM (same config/input, "
                  "different output across processes)", file=sys.stderr)
        else:
            print("VERDICT: CONFIG_DEPENDENCE (same input, ON/OFF gem differs)",
                  file=sys.stderr)
    else:
        print("VERDICT: identical input and output -> not reproducible this run",
              file=sys.stderr)
    return 0


def _run_both_configs(args) -> int:
    """Backwards-compatible alias for --compare on,off (as A=on, B=off)."""
    return _run_pair("on", "off", args)

def _analyze_diag(pattern: str) -> int:
    """Group per-test fingerprints across diag JSONL files.

    Reports, for every test nodeid, the distinct (cpu_seed, dev_seed,
    dev_state) fingerprints seen. A test with >1 distinct fingerprint ran with
    different RNG state across runs (input nondeterminism candidate); a test
    with a single fingerprint but a recorded outcome flip is kernel-side.
    """
    import glob as _glob
    from collections import defaultdict

    per_test = defaultdict(set)
    files = sorted(_glob.glob(pattern))
    if not files:
        print(f"no files match {pattern!r}", file=sys.stderr)
        return 2
    for path in files:
        for line in Path(path).read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp = rec.get("fingerprint", {})
            key = (fp.get("cpu_seed"), fp.get("dev_seed"), fp.get("dev_state"))
            per_test[rec.get("nodeid")].add(key)

    unstable = {k: v for k, v in per_test.items() if len(v) > 1}
    print(f"files: {len(files)}  tests: {len(per_test)}  "
          f"tests with >1 fingerprint: {len(unstable)}")
    for nodeid, keys in list(unstable.items())[:50]:
        print(f"  {nodeid}")
        for k in sorted(keys, key=str):
            print(f"      cpu_seed={k[0]} dev_seed={k[1]} dev_state={k[2]}")
    if not unstable:
        print("All tests saw a single RNG fingerprint -> seeding is effective;"
              " any remaining instability is kernel/launch-side.")
    else:
        print("Some tests saw different fingerprints -> the seed is not reaching"
              " the input generator for those "
              "(INPUT_NONDETERMINISM).")
    return 0

def _triton_probe(args) -> int:
    """Minimal aliased in-place triton kernels to isolate the nondeterminism.

    Runs several variants of `x = x * s` (and one out-of-place control) and
    reports which are deterministic. Variants:

      oop            : y = x * s          (separate buffers, control)
      inplace        : x = x * s          (aliased, BLOCK=1024)
      inplace_b1     : x = x * s          (aliased, BLOCK=1, one elem/row)
      inplace_b32    : x = x * s          (aliased, BLOCK=32)
      inplace_barrier: aliased + tl.debug_barrier() before the store
      inplace_ns1    : aliased + num_stages=1
      inplace_nombs1 : aliased + multibuffer=False + num_stages=1

    Also reports the positions (and position mod BLOCK) of the bad elements,
    to reveal a per-program / per-vector-lane boundary. Interpretation: if `oop`
    is stable but every aliased variant flakes, the bug is triggered purely by
    x_ptr == out_ptr; if BLOCK=1 is stable too, it needs vectorization.
    """
    import hashlib

    import torch
    import triton
    import triton.language as tl

    import flag_gems

    @triton.jit
    def _mul_oop(x_ptr, y_ptr, s, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * s, mask=m)

    @triton.jit
    def _mul_inplace(x_ptr, s, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m)
        tl.store(x_ptr + offs, x * s, mask=m)

    @triton.jit
    def _mul_inplace_barrier(x_ptr, s, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m)
        tl.debug_barrier()
        tl.store(x_ptr + offs, x * s, mask=m)

    device = flag_gems.device
    n = 1024 * 1024
    scalar = -0.999

    def _hash(t):
        return hashlib.sha1(
            t.detach().to("cpu").contiguous().numpy().tobytes()
        ).hexdigest()[:16]

    _seed(SEED)
    base = torch.randn(n, dtype=torch.float32, device=device)
    ref = (base.to("cpu").double() * scalar).to(torch.float32)

    def _bad_positions(out, block):
        r = out.to("cpu")
        bad = (~torch.isclose(r, ref, atol=1e-4, rtol=1e-4)).nonzero()
        pos = bad.reshape(-1).tolist()[:16]
        return {
            "n": int(bad.numel()),
            "first_positions": pos,
            "positions_mod_block": sorted({p % block for p in pos}),
        }

    def run_oop(block, **kw):
        y = torch.empty_like(base)
        _mul_oop[(triton.cdiv(n, block),)](base, y, scalar, n, BLOCK=block, **kw)
        return y

    def run_inplace(kernel, block, **kw):
        x = base.clone()
        kernel[(triton.cdiv(n, block),)](x, scalar, n, BLOCK=block, **kw)
        return x

    variants = {
        "oop": (lambda kw: run_oop(1024, **kw), 1024),
        "inplace": (lambda kw: run_inplace(_mul_inplace, 1024, **kw), 1024),
        "inplace_b1": (lambda kw: run_inplace(_mul_inplace, 1, **kw), 1),
        "inplace_b32": (lambda kw: run_inplace(_mul_inplace, 32, **kw), 32),
        "inplace_barrier": (
            lambda kw: run_inplace(_mul_inplace_barrier, 1024, **kw), 1024),
        "inplace_ns1": (
            lambda kw: run_inplace(_mul_inplace, 1024,
                                   **{**kw, "num_stages": 1}), 1024),
        "inplace_nombs1": (
            lambda kw: run_inplace(_mul_inplace, 1024,
                                   **{**kw, "num_stages": 1,
                                      "multibuffer": False}), 1024),
    }

    report = {"n": n, "scalar": scalar, "repeat": args.repeat, "runs": {}}
    for name, (fn, block) in variants.items():
        hashes, nbad, last_pos = [], [], None
        try:
            for _ in range(args.repeat):
                out = fn({})
                hashes.append(_hash(out))
                info = _bad_positions(out, block)
                nbad.append(info["n"])
                last_pos = info
        except Exception as exc:  # noqa: BLE001
            report["runs"][name] = f"ERROR: {type(exc).__name__}: {exc}"
            continue
        report["runs"][name] = {
            "hashes": hashes, "n_mismatch": nbad,
            "last_positions": last_pos,
            "deterministic": len(set(hashes)) == 1,
        }
    print(json.dumps(report, indent=2, default=str))

    r = report["runs"]

    def det(name):
        v = r.get(name)
        return isinstance(v, dict) and v["deterministic"]

    if det("oop") and not det("inplace"):
        print("VERDICT: ALIASING is the trigger (out-of-place stable, "
              "in-place flaky).", file=sys.stderr)
        if det("inplace_b1"):
            print("  BLOCK=1 in-place is stable -> needs vectorization "
                  "(multi-element tile) to trigger.", file=sys.stderr)
        else:
            print("  BLOCK=1 in-place is ALSO flaky -> pure aliasing, "
                  "independent of tile width.", file=sys.stderr)
        if det("inplace_ns1") and det("inplace_nombs1"):
            print("  FIX: num_stages=1 makes it stable.", file=sys.stderr)
        elif det("inplace_nombs1"):
            print("  FIX: multibuffer=False + num_stages=1 makes it stable.",
                  file=sys.stderr)
        else:
            print("  neither num_stages=1 nor multibuffer=False stabilises it"
                  " -> the only reliable fix is to avoid aliasing.",
                  file=sys.stderr)
    elif all(det(n) for n in ("oop", "inplace")):
        print("VERDICT: all variants stable in this probe run; rerun with a "
              "larger --repeat or a larger n to surface the race.",
              file=sys.stderr)
    else:
        print("VERDICT: see per-variant results above.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
