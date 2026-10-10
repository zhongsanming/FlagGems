#!/usr/bin/env python3
"""Fast targeted flaky-case runner: run ONLY the known flaky cases, twice, under
`TRITON_ALL_BLOCKS_PARALLEL` on and off, to test whether the Ascend
auto-blockify pass is the common trigger.

Reads the flaky node ids written by tools/diagnose_flaky_ops.py (or the helper
in this repo) from a file (default results/flaky_diag/flaky_cases.txt), runs
them with a small number of repeats as a single pytest invocation per condition
(far faster than the per-op full suite), and prints a comparison.

Usage:
    python tools/run_flaky_probe.py                       # default file, 2 repeats
    python tools/run_flaky_probe.py --repeat 3
    python tools/run_flaky_probe.py --file results/flaky_diag/flaky_cases.txt

It reports, per condition (abp=1, abp=0, and abp unset), how many of the given
cases fail. If `abp=0` has markedly fewer failures than `abp=1`/unset, the
auto-blockify pass is the trigger.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_condition(cases: list[str], env_extra: dict, label: str,
                  repeat: int, out_dir: Path) -> dict:
    fails_total = 0
    per_case = {}
    for i in range(repeat):
        env = os.environ.copy()
        env.update(env_extra)
        out_json = out_dir / f"_probe_{label}_{i}.json"
        if out_json.exists():
            out_json.unlink()
        cmd = ["pytest", *cases, "--record", "json", "--output", str(out_json),
               "--ref", "cpu", "--continue-on-collection-errors", "-q"]
        subprocess.run(cmd, cwd=str(ROOT), env=env, timeout=3600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        data = {}
        if out_json.exists():
            try:
                data = json.loads(out_json.read_text())
            except Exception:  # noqa: BLE001
                data = {}
        for k, v in data.items():
            st = per_case.setdefault(k, {"pass": 0, "fail": 0})
            if v.get("result") == "passed":
                st["pass"] += 1
            elif v.get("result") != "skipped":
                st["fail"] += 1
                fails_total += 1
    flaky = {k: v for k, v in per_case.items() if v["fail"] > 0 and v["pass"] > 0}
    return {"label": label, "fails_total": fails_total,
            "n_cases": len(per_case), "n_flaky": len(flaky),
            "per_case": per_case}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="results/flaky_diag/flaky_cases.txt")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--level", default="core")
    args = ap.parse_args(argv)

    f = ROOT / args.file
    if not f.exists():
        print(f"missing {f}; generate it first (see tools/FLAKY_OPS.md)", file=sys.stderr)
        return 2
    cases = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
    out_dir = f.parent
    print(f"running {len(cases)} cases x {args.repeat} repeats per condition\n")

    conditions = [
        ("unset", {}),
        ("abp=1", {"TRITON_ALL_BLOCKS_PARALLEL": "1"}),
        ("abp=0", {"TRITON_ALL_BLOCKS_PARALLEL": "0"}),
    ]
    results = []
    for label, extra in conditions:
        print(f"### condition {label} ...")
        r = run_condition(cases, extra, label.replace("=", ""), args.repeat, out_dir)
        results.append(r)
        print(f"    total failures={r['fails_total']}  "
              f"flaky cases={r['n_flaky']}/{r['n_cases']}")

    print("\n=== summary (flaky cases per condition) ===")
    for r in results:
        print(f"  {r['label']:6s}: flaky={r['n_flaky']:3d}  fails={r['fails_total']}")

    # verdict
    by = {r["label"]: r for r in results}
    if by["abp=0"]["n_flaky"] == 0 and by["unset"]["n_flaky"] > 0:
        print("\nVERDICT: flakiness disappears with TRITON_ALL_BLOCKS_PARALLEL=0 "
              "-> Ascend auto-blockify is the trigger.")
    elif by["abp=0"]["n_flaky"] < by["unset"]["n_flaky"]:
        print("\nVERDICT: fewer flakes with auto-blockify off, but not zero "
              "-> auto-blockify is a major contributor; residual is another "
              "nondeterminism (see matrix_rank doc defects 1.3-12/1.3-13).")
    else:
        print("\nVERDICT: auto-blockify does not explain it; look at the other "
              "documented defects (MTE3/MTE2 race, atomics, where-with-0).")

    (out_dir / "flaky_probe_report.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out_dir/'flaky_probe_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
