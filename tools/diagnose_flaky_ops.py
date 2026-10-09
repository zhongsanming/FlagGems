#!/usr/bin/env python3
"""Diagnose flaky FlagGems accuracy tests on Ascend.

The A/B comparison flagged a set of operators whose cases intermittently fail
(attempt1 failed -> retry1 passed), on BOTH compiler configs, e.g.:

    linear, scatter_reduce, adaptive_max_pool3d, cat, index_put(_),
    sum_dim, reflection_pad2d(_out), slice_backward, var_dim, moe_sum,
    avg_pool3d, tril(_out), var_correction, index_copy(_), conj_physical

This tool helps find the reason. It runs the *actual* pytest tests for one or
more operators N times and reports, per case, how often it passes/fails, plus
the first failure reason. It can also flip candidate compiler knobs between
runs (notably TRITON_ALL_BLOCKS_PARALLEL, which enables the Ascend
auto-blockify pass and is the prime suspect for nondeterministic codegen), so
you can see whether the flakiness tracks that knob.

Examples
--------
    # how flaky is linear? run its 90 cases 20 times
    python tools/diagnose_flaky_ops.py --op linear --repeat 20

    # a few ops at once, custom repeat, fresh triton cache each run
    python tools/diagnose_flaky_ops.py --op cat,index_put,sum_dim --repeat 10 --fresh-cache

    # test the auto-blockify hypothesis: same runs with the flag forced off
    python tools/diagnose_flaky_ops.py --op linear --repeat 20 --no-all-blocks-parallel

    # exact node ids
    python tools/diagnose_flaky_ops.py --cases "tests/test_linear.py::test_linear_2d_with_bias[7-256-192-dtype0]" --repeat 30

It writes a JSON report (<out>/flaky_report.json) and a JSONL log of every run
(<out>/flaky_runs.jsonl) for later analysis.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def selector_for(op: str) -> list[str]:
    """pytest selector for an operator: -m <marker> (mirrors run_tests)."""
    marker = op
    if op.startswith("_"):
        marker = "underscore_" + op.lstrip("_")
    return ["-m", marker]


def run_pytest(selector: list[str], out_json: Path, env: dict,
               timeout: int) -> tuple[int, dict, str]:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists():
        out_json.unlink()
    cmd = ["pytest", *selector, "--record", "json", "--output", str(out_json),
           "--ref", "cpu", "--continue-on-collection-errors", "-q"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, timeout=timeout,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True)
        rc = proc.returncode
        tail = proc.stdout[-4000:]
    except subprocess.TimeoutExpired as e:
        rc, tail = 124, f"[TIMEOUT after {timeout}s]\n{(e.stdout or '')[-2000:]}"
    data = {}
    if out_json.exists():
        try:
            data = json.loads(out_json.read_text())
        except Exception:  # noqa: BLE001
            data = {}
    return rc, data, tail


def env_fingerprint() -> dict:
    import importlib.metadata as md

    fp = {}
    for pkg in ("triton", "flagtree", "flag-gems", "torch", "torch-npu"):
        try:
            fp[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            fp[pkg] = None
    fp["env"] = {k: os.environ.get(k) for k in (
        "TRITON_ALL_BLOCKS_PARALLEL", "TRITON_DISABLE_LANE_VECTORIZE",
        "TRITON_ENABLE_LANE_VECTORIZE_BLOCK_MODE",
        "TRITON_LANE_VECTORIZE_ALLOW_CONCAT",
        "TRITON_LANE_VECTORIZE_ALLOW_ADDRESS_CONES",
        "TRITON_ALWAYS_COMPILE", "TRITON_CACHE_DIR", "FLAGGEMS_CACHE_DIR")}
    return fp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", default=None,
                    help="comma-separated op ids (pytest markers), e.g. linear,cat")
    ap.add_argument("--cases", default=None,
                    help="comma-separated explicit pytest node ids (overrides --op)")
    ap.add_argument("--repeat", type=int, default=10,
                    help="number of runs per op/case selection")
    ap.add_argument("--level", default="core",
                    choices=["core", "comprehensive"])
    ap.add_argument("--out", default="results/flaky_diag")
    ap.add_argument("--fresh-cache", action="store_true",
                    help="fresh TRITON_CACHE_DIR + TRITON_ALWAYS_COMPILE=1 per "
                         "run (isolate stale-binary effects)")
    ap.add_argument("--all-blocks-parallel", dest="abp", default=None,
                    choices=["1", "0"],
                    help="force TRITON_ALL_BLOCKS_PARALLEL on/off for every run "
                         "(default: leave the environment unchanged)")
    ap.add_argument("--disable-lane-vectorize", action="store_true",
                    help="set TRITON_DISABLE_LANE_VECTORIZE=1 (OFF config)")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-run pytest timeout (s)")
    args = ap.parse_args(argv)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    runs_log = (out / "flaky_runs.jsonl").open("a")

    sel_list: list[tuple[str, list[str]]] = []
    if args.cases:
        for node in [c.strip() for c in args.cases.split(",") if c.strip()]:
            sel_list.append((node, [node]))
    else:
        for op in [o.strip() for o in (args.op or "").split(",") if o.strip()]:
            sel_list.append((op, selector_for(op)))
    if not sel_list:
        print("nothing to do: pass --op or --cases", file=sys.stderr)
        return 2

    print(json.dumps({"started": now(), "env_fingerprint": env_fingerprint(),
                      "selections": [s[0] for s in sel_list],
                      "repeat": args.repeat, "fresh_cache": args.fresh_cache,
                      "all_blocks_parallel": args.abp,
                      "disable_lane_vectorize": args.disable_lane_vectorize},
                     indent=2))

    report = {"env_fingerprint": env_fingerprint(), "selections": {}}

    for label, selector in sel_list:
        print(f"\n########## {label}  ({args.repeat} runs) ##########")
        per_case = defaultdict(lambda: {"pass": 0, "fail": 0, "skip": 0,
                                        "reasons": []})
        run_summaries = []
        for i in range(args.repeat):
            env = os.environ.copy()
            # Pin the compiler config for this diagnostic run.
            env["TRITON_DISABLE_LANE_VECTORIZE"] = (
                "1" if args.disable_lane_vectorize else "0")
            if not args.disable_lane_vectorize:
                env.setdefault("TRITON_ENABLE_LANE_VECTORIZE_BLOCK_MODE", "1")
                env.setdefault("TRITON_LANE_VECTORIZE_ALLOW_CONCAT", "1")
                env.setdefault("TRITON_LANE_VECTORIZE_ALLOW_ADDRESS_CONES", "1")
            if args.abp is not None:
                env["TRITON_ALL_BLOCKS_PARALLEL"] = args.abp
            if args.fresh_cache:
                td = tempfile.mkdtemp(prefix=f"flaky-{label}-")
                env["TRITON_CACHE_DIR"] = os.path.join(td, "triton")
                env["FLAGGEMS_CACHE_DIR"] = os.path.join(td, "fg")
                env["TRITON_ALWAYS_COMPILE"] = "1"
            # allow quick/comprehensive test filtering to match the harness
            sel = list(selector)
            out_json = out / f"_run_{label.replace('/','_').replace('::','__')}_{i}.json"
            rc, data, tail = run_pytest(sel, out_json, env, args.timeout)
            npass = nfail = nskip = 0
            for case, rec in data.items():
                res = rec.get("result")
                if res == "passed":
                    per_case[case]["pass"] += 1; npass += 1
                elif res == "skipped":
                    per_case[case]["skip"] += 1; nskip += 1
                else:
                    per_case[case]["fail"] += 1; nfail += 1
                    reason = (rec.get("reason") or "")[:400]
                    per_case[case]["reasons"].append(reason)
            run_summaries.append({"run": i, "rc": rc, "pass": npass,
                                  "fail": nfail, "skip": nskip})
            runs_log.write(json.dumps({
                "time": now(), "label": label, "run": i, "rc": rc,
                "pass": npass, "fail": nfail, "skip": nskip}) + "\n")
            runs_log.flush()
            print(f"  run {i:2d}: rc={rc} pass={npass} fail={nfail} skip={nskip}")
            if npass == 0 and nfail == 0:
                print("    (no results parsed; tail follows)")
                print("    " + tail.replace("\n", "\n    ")[:1200])

        flaky = {c: v for c, v in per_case.items()
                 if v["fail"] > 0 and v["pass"] > 0}
        always_fail = {c: v for c, v in per_case.items()
                       if v["fail"] > 0 and v["pass"] == 0}
        print(f"  => flaky cases (passed sometimes, failed sometimes): {len(flaky)}")
        for c, v in list(flaky.items())[:20]:
            print(f"     {c.split('::')[-1][:70]}  pass={v['pass']} fail={v['fail']}")
            if v["reasons"]:
                print("        e.g. " + v["reasons"][0].replace("\n", " | ")[:160])
        if always_fail:
            print(f"  => always-failing cases: {len(always_fail)} "
                  f"(pre-existing, not flaky)")

        report["selections"][label] = {
            "runs": run_summaries,
            "flaky_cases": {c: {k: v[k] for k in ("pass", "fail", "skip")}
                            for c, v in flaky.items()},
            "always_fail_cases": {c: {k: v[k] for k in ("pass", "fail", "skip")}
                                  for c, v in always_fail.items()},
        }

    runs_log.close()
    (out / "flaky_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out/'flaky_report.json'} and {out/'flaky_runs.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
