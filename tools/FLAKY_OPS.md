# Diagnosing flaky FlagGems accuracy tests (Ascend)

The LaneVectorize A/B comparison flagged a number of operators whose cases
intermittently fail (`attempt1` failed, `retry1` passed), on **both** compiler
configs. That is nondeterminism, not an A/B regression:

`linear, scatter_reduce, adaptive_max_pool3d, cat, index_put(_), sum_dim,
reflection_pad2d(_out), slice_backward, var_dim, moe_sum, avg_pool3d,
tril(_out), var_correction, index_copy(_), conj_physical`

Three tools help find the reason.

## 1. Run the real pytest tests N times — `diagnose_flaky_ops.py`

```bash
python tools/diagnose_flaky_ops.py --op linear --repeat 20
python tools/diagnose_flaky_ops.py --op "cat,index_put,sum_dim" --repeat 10 --fresh-cache
python tools/diagnose_flaky_ops.py --op linear --repeat 20 --no-all-blocks-parallel   # test auto-blockify
python tools/diagnose_flaky_ops.py --cases "tests/test_linear.py::test_linear_2d_with_bias[7-256-192-dtype0]" --repeat 30
```

For every case it reports how many runs passed/failed and the first failure
reason, and writes `results/flaky_diag/flaky_report.json` +
`flaky_runs.jsonl`. A case with `pass>0 and fail>0` is flaky; `fail>0,
pass==0` is a pre-existing hard failure.

## 2. Reproduce in-process at the op level — `diagnose_flaky_op_raw.py`

Calls the FlagGems op directly on a fixed seeded input, N times, and prints the
per-run output fingerprint. A changing fingerprint on identical input is the
proof of nondeterminism, independent of pytest.

```bash
python tools/diagnose_flaky_op_raw.py --op linear --repeat 20
python tools/diagnose_flaky_op_raw.py --op sum_dim --repeat 20
```

Because the trigger is a compile-time knob, run each variation in its **own
process** and compare the `deterministic` field:

```bash
# baseline
python tools/diagnose_flaky_op_raw.py --op linear --repeat 20
# auto-blockify on (suspect)
TRITON_ALL_BLOCKS_PARALLEL=1 python tools/diagnose_flaky_op_raw.py --op linear --repeat 20
# force a fresh compile (the knob is not part of the cache key)
TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR=/tmp/tc1 python tools/diagnose_flaky_op_raw.py --op linear --repeat 20
# lane-vectorize off (OFF config)
TRITON_DISABLE_LANE_VECTORIZE=1 python tools/diagnose_flaky_op_raw.py --op linear --repeat 20
```

## 3. Reduce to a bare triton kernel — `upstream_repro/repro.py`

If an op is nondeterministic, `tools/upstream_repro/repro.py` shows the same
nondeterminism with a trivial standalone kernel and isolates
`TRITON_ALL_BLOCKS_PARALLEL` (auto-blockify) as the trigger. Use it to file the
upstream bug (flagtree / AscendNPU-IR). See `tools/upstream_repro/README.md`.

## Interpreting the results

* Fingerprint changes across runs on identical, re-seeded input
  => kernel/compiler nondeterminism (the flakiness cause), **not** an input
  problem and **not** a LaneVectorize regression when it also happens with
  `TRITON_DISABLE_LANE_VECTORIZE=1`.
* Same case fails at the same index/value in ON and OFF
  => config-independent; treat as a pre-existing flaky test.
* Flakiness disappears with `TRITON_ALL_BLOCKS_PARALLEL` unset (or with
  `num_stages=1`) => auto-blockify codegen is the trigger.

## Known so far

* `avg_pool3d[dtype0-shape6-2-2-0-False-False-None]`: ON and OFF fail
  identically on `attempt1` (4096/4096 mismatch) and recover on retry — a flaky
  test, not an IR change. Its fp16 TTIR is byte-identical ON vs OFF.
* `linear`: OFF attempt1 has 18 failures, ON 42, both fully recover; failures
  are byte-identical between configs (e.g. `nan` at `(0,0)`).
* Prime suspect for all of them: the Ascend auto-blockify pass
  (`TRITON_ALL_BLOCKS_PARALLEL=1` / `--enable-auto-blockify-loop`) producing
  nondeterministic code. Run tool #1 with `--no-all-blocks-parallel` and tool
  #2 under the two env settings to confirm per op.
