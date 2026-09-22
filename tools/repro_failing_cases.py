#!/usr/bin/env python3
"""Reproduce specific FlagGems accuracy failures and diff their TTIR.

Given a comparison_result.json (from tools/compare_results.py) or an explicit
list of pytest node ids, this re-runs ONLY those cases under both compiler
configurations, on the same NPU, with a fresh per-case cache, then diffs the
dumped TTIR.

    OFF : TRITON_DISABLE_LANE_VECTORIZE=1
    ON  : TRITON_DISABLE_LANE_VECTORIZE=0
          TRITON_ENABLE_LANE_VECTORIZE_BLOCK_MODE=1

Each case gets its own cache (TRITON_CACHE_DIR + TRITON_ALWAYS_COMPILE=1), so
the .ttir dumped for the case is unambiguous. Dumps use TRITON_MLIR_PRINT_OP_GENERIC=1
so they are canonical/diffable. For every case that fails in at least one
config, the script pairs the off/on kernels by name and writes a normalized
TTIR unified diff.

Examples
--------
    # list the MLIRCompilationError regressions selected (no execution)
    python tools/repro_failing_cases.py \
        --comparison results/comparison-lv-off-vs-lv-on-block-on/comparison_result.json \
        --category MLIRCompilationError --limit 5

    # actually run them
    python tools/repro_failing_cases.py ... --run

    # explicit cases
    python tools/repro_failing_cases.py \
        --cases "tests/test_mm.py::test_mm[True-dtype0-1-1-32]" --run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

import run_tests as rt  # noqa: E402
import run_ab_interleaved as ab  # noqa: E402  (configs + env builder)

SSA = re.compile(rb"%[A-Za-z0-9_]+")
TMP = re.compile(rb"/tmp/[A-Za-z0-9_./-]+")


def now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path):
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def norm_ttir(path: Path) -> str:
    data = path.read_bytes()
    data = SSA.sub(b"%V", data)
    data = TMP.sub(b"/tmp/N", data)
    return data.decode("utf-8", "replace")


def strip_tests_prefix(node_id: str) -> str:
    return node_id[len("tests/"):] if node_id.startswith("tests/") else node_id


def run_one(node_id: str, op: str, config: str, npu: int, work: dict):
    """Run a single pytest node id under one config. Returns (rc, entry)."""
    out = work["case_dir"] / config
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "result.json"
    if raw.exists():
        raw.unlink()
    so = out / "stdout.log"
    se = out / "stderr.log"
    env = ab.build_env(npu, config, out / "cache", work["device_vars"],
                       generic_ir=work["generic_ir"])
    cmd = ["pytest", strip_tests_prefix(node_id), "--record", "json",
           "--output", str(raw)]
    if op not in work["skip_cpu"]:
        cmd += ["--ref", "cpu"]
    cmd += ["-vs"]
    with so.open("w") as fho, se.open("w") as fhe:
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT / "tests"), env=env,
                                  stdout=fho, stderr=fhe, timeout=work["timeout"])
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -100
    data = read_json(raw) or {}
    entry = next(iter(data.values()), {}) if data else {}
    return rc, entry


def ttir_pairs(case_dir: Path):
    """Yield (kernel_name, off_path, on_path) for kernels present in both."""
    off = case_dir / "off" / "cache"
    on = case_dir / "on" / "cache"
    kernels = {}
    for root, p in ((off, "off"), (on, "on")):
        if not root.is_dir():
            continue
        for ttir in root.rglob("*.ttir"):
            kernels.setdefault(ttir.name, {})[p] = ttir
    for name in sorted(kernels):
        rec = kernels[name]
        yield name, rec.get("off"), rec.get("on")


def classify_reason(reason: str | None) -> str:
    reason = reason or ""
    for key in ("MLIRCompilationError", "ACL stream synchronize",
                "are not close", "are not equal"):
        if key in reason:
            return key
    return "other" if reason else "passed"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--comparison", default=None,
                   help="comparison_result.json to select failing cases from")
    p.add_argument("--category", default="MLIRCompilationError",
                   help="substring filter on the ON failure reason "
                        "(default MLIRCompilationError; 'any' disables)")
    p.add_argument("--ops", default=None, help="comma-separated op filter")
    p.add_argument("--cases", default=None,
                   help="explicit comma-separated pytest node ids")
    p.add_argument("--limit", type=int, default=10, help="max cases (0 = all)")
    p.add_argument("--configs", default="off,on",
                   help="configs to run (default off,on)")
    p.add_argument("--gpus", default="0", help="NPU id(s); case 0 uses the first")
    p.add_argument("--output-dir", default=None,
                   help="default results/repro-failing-<ts>")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--generic-ir", dest="generic_ir", action="store_true", default=True)
    p.add_argument("--no-generic-ir", dest="generic_ir", action="store_false")
    p.add_argument("--run", action="store_true", help="actually execute pytest")
    args = p.parse_args(argv)

    rt.probe_env()
    skip_cpu = {o["id"] for o in rt.get_ops_from_inventory()
                if "NoCPU" in o.get("labels", [])}
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    # ---- select cases -----------------------------------------------------
    cases = []  # (op, node_id, on_reason)
    if args.cases:
        for node_id in (c.strip() for c in args.cases.split(",") if c.strip()):
            stem = Path(node_id.split("::")[0]).stem  # test_mm -> mm
            op = stem[5:] if stem.startswith("test_") else stem
            cases.append((op, node_id, None))
    elif args.comparison:
        payload = read_json(Path(args.comparison))
        if not payload:
            print(f"[repro] cannot read {args.comparison}")
            return 1
        for c in payload.get("accuracy", {}).get("cases", []):
            if c.get("change") != "regression":
                continue
            reason = c.get("on_reason") or ""
            if args.category.lower() != "any" and args.category not in reason:
                continue
            cases.append((c["op"], c["test_id"], reason))
    else:
        print("[repro] pass --comparison or --cases")
        return 1

    if args.ops:
        ops = {o.strip() for o in args.ops.split(",")}
        cases = [c for c in cases if c[0] in ops]
    # de-dupe, keep order
    seen = set()
    cases = [c for c in cases if not (c[1] in seen or seen.add(c[1]))]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        print("[repro] no cases selected")
        return 1

    out_root = Path(args.output_dir) if args.output_dir else (
        ROOT / "results" / f"repro-failing-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)
    device_vars = ab.device_vars()
    npu = int(args.gpus.split(",")[0])

    print(f"[repro] {len(cases)} case(s), configs={configs}, npu={npu}")
    print(f"[repro] output={out_root}  run={args.run}")
    man = {
        "created": now(),
        "comparison": args.comparison,
        "category": args.category,
        "configs": configs,
        "flagtree": rt.ENV_INFO.get("flagtree"),
        "triton": rt.ENV_INFO.get("triton", {}).get("version"),
        "package": ab.this_pkg_env().get("version"),
        "cases": [{"op": op, "test_id": nid} for op, nid, _ in cases],
    }
    (out_root / "manifest.json").write_text(json.dumps(man, indent=2, default=str))

    report = []
    for i, (op, node_id, reason) in enumerate(cases, 1):
        case_dir = out_root / op / sanitize(strip_tests_prefix(node_id))
        case_dir.mkdir(parents=True, exist_ok=True)
        rec = {"op": op, "test_id": node_id, "on_reason_before": reason,
               "configs": {}, "ttir": []}
        if not args.run:
            print(f"[repro] [{i}/{len(cases)}] {op:20s} {strip_tests_prefix(node_id)}"
                  + (f"  ({classify_reason(reason)})" if reason else ""))
            report.append(rec)
            continue
        if args.run:
            for config in configs:
                work = {"case_dir": case_dir, "device_vars": device_vars,
                        "skip_cpu": skip_cpu, "timeout": args.timeout,
                        "generic_ir": args.generic_ir}
                rc, entry = run_one(node_id, op, config, npu, work)
                rec["configs"][config] = {
                    "rc": rc, "result": entry.get("result"),
                    "reason": (entry.get("reason") or "")[:2000],
                }
                print(f"[repro] [{i}/{len(cases)}] {config:3s} "
                      f"{'PASS' if entry.get('result')=='passed' else entry.get('result') or 'NORESULT'}"
                      f"  {op} {strip_tests_prefix(node_id)}")

            # ---- TTIR diff -------------------------------------------------
            for name, po, pn in ttir_pairs(case_dir):
                if po is None or pn is None:
                    rec["ttir"].append({"kernel": name, "off": po is not None,
                                        "on": pn is not None, "diff": "one-sided"})
                    continue
                a, b = norm_ttir(po), norm_ttir(pn)
                if a == b:
                    rec["ttir"].append({"kernel": name, "diff": "identical"})
                    continue
                d = list(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                              "off", "on", lineterm=""))
                (case_dir / f"{name}.ttir.diff").write_text("\n".join(d))
                add = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
                rem = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
                rec["ttir"].append({"kernel": name, "diff": "DIFFERS",
                                    "added_lines": add, "removed_lines": rem,
                                    "diff_file": str((case_dir / f"{name}.ttir.diff").relative_to(out_root))})
                print(f"[repro]      ttir {name}: +{add}/-{rem} lines "
                      f"-> {name}.ttir.diff")
        report.append(rec)

    (out_root / "report.json").write_text(json.dumps(report, indent=2, default=str))

    # ---- markdown ---------------------------------------------------------
    md = ["# Failing-case repro", "",
          f"- comparison: `{args.comparison}`", f"- category: `{args.category}`",
          f"- run: {args.run}", ""]
    for rec in report:
        md.append(f"## `{rec['op']}` `{strip_tests_prefix(rec['test_id'])}`")
        for config in configs:
            c = rec["configs"].get(config)
            if c:
                md.append(f"- {config}: **{c['result']}** (rc={c['rc']}) "
                          f"{classify_reason(c['reason'])}")
        for t in rec["ttir"]:
            md.append(f"- ttir `{t['kernel']}`: {t['diff']}"
                      + (f" +{t.get('added_lines')}/-{t.get('removed_lines')}"
                         f" ({t.get('diff_file')})" if t.get("diff_file") else ""))
        md.append("")
    (out_root / "report.md").write_text("\n".join(md))
    print(f"\n[repro] wrote {out_root}/report.md and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
