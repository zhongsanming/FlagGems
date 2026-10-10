# Flaky ops vs documented Ascend toolchain defects

`matrix_rank_昇腾工具链缺陷与精度问题.md` documents defects found while
developing `matrix_rank` on Ascend 910B (CANN 8.5.0 / triton-ascend 3.2.0 /
BiShengIR). Most are **backend-wide** and explain the flaky ops from the A/B run.

Key point from that doc: the defects are **probabilistic and source-perturbation
sensitive**, and "a validated result is only valid for the source + toolchain
generation it was verified on" (meta-lesson #2). So flakiness across ops is
expected, not an A/B artifact.

## Mapping: flaky op → most likely documented defect(s)

| Flaky op | Relevant constructs in its kernel | Documented defect(s) |
|---|---|---|
| `linear` (GEMM path) | `tl.dot` (line 308), `tl.sum` (81), **`tl.atomic_add` (333)** | §1.2 (dot operand limits), §1.1-10 (atomic miscompile under `TRITON_ALL_BLOCKS_PARALLEL`) |
| `index_put`, `index_put_` | generated kernel emits `tl.atomic_add` (`write_atomic`) | §1.1-10, §1.3-13 (cross-program sync unreliable) |
| `scatter_reduce` | atomics / claim+lock (`_GRAPH_LOCK`, "replay atomics") | §1.1-10, §1.3-13 |
| `cat` | copy kernels; indexing | §1.3-12 (MTE3 store → MTE2 load not ordered) if it writes then reads |
| `sum_dim` (test_sum.py) | masked load + `tl.sum(axis=…)` reduction | §1.1-3 (axis-0 masked reduction wrong), §1.1-5 (bool `tl.sum`), §1.2 |
| `var_dim`, `var_correction` (test_var.py) | reductions over masked tiles | §1.1-3, §1.1-5, §2.4-9 (transcendental/bitcast) |
| `adaptive_max_pool3d` | heavy `tl.where`, `tl.where(...,0)`, `tl.maximum` w/ NaN workaround | §1.1-1 (`where(mask, vec, 0.0)` miscompile), §1.1-3 |
| `avg_pool3d` | `sum_acc += tl.where(in_mask, current_val, 0.0)` | §1.1-1 |
| `reflection_pad2d(_out)` | `tl.where` on boundary indices (`ih = tl.where(m_h < H_in, m_h, …)`) | §1.1-1, §1.1-4 (`arange == const` mask) |
| `tril`, `tril_out` | masked writes | §1.1-1, §1.1-4 |
| `slice_backward` | masked store into `torch.zeros` buffer | §1.1-1, §1.3-12 (zeros then store+read) |
| `moe_sum` | nested masked loads + accumulate | §1.1-1, §1.1-3 |
| `index_copy`, `index_copy_` | scatter writes | §1.1-10, §1.3-13 |
| `conj_physical` | elementwise | §1.1-1 (if it uses `where`) |

## The strongest cross-cutting cause: `TRITON_ALL_BLOCKS_PARALLEL`

Defect **§1.1-10** in the doc is decisive:

> `TRITON_ALL_BLOCKS_PARALLEL` (**其他模块 import 时全局设置** = set globally at
> import by another module) 使大路径的 matvec/apply kernel 的 atomic_add 求和出错.

That "another module" is `flag_gems/runtime/backend/_ascend/fused/sparse_attention.py`,
which used to do `os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"` at import
(now scoped, see memory/commit `ca3563fa`). Under it, the flagtree Ascend
**auto-blockify** pass runs for every kernel and, per the doc, **atomic_add
sums come out wrong**. We independently reproduced nondeterministic codegen under
this knob with a bare kernel (`tools/upstream_repro/repro.py`).

This predicts that the atomic-using ops (`linear`, `index_put`, `scatter_reduce`,
`index_copy`) are the ones most likely to be broken by the knob.

## Other backend-wide races that need no knob

- **§1.3-12**: within one kernel, `tl.store` (MTE3) and `tl.load` (MTE2) are on
  independent hardware queues and are **not ordered** — a store-then-load of the
  same address races. This is exactly the aliased in-place nondeterminism we
  reproduced (`x = x*s`), and it hits any op that writes then reads a buffer
  (`slice_backward`, in-place ops). `tl.debug_barrier()` does **not** fix it and
  itself perturbs other miscompiles.
- **§1.3-13**: cross-program sync (atomics/spin barriers) is unreliable for
  ≥8 programs — hits reduction/scatter kernels.

## Fast diagnosis plan (the full suite is slow)

1. Generate the exact flaky node ids once:

   ```bash
   # already produced: results/flaky_diag/flaky_cases.txt (213 cases)
   ```

2. Run **only those** under the three conditions (fast, one pytest call each):

   ```bash
   python tools/run_flaky_probe.py --repeat 2
   ```

   It runs with `TRITON_ALL_BLOCKS_PARALLEL` unset / `=1` / `=0` and prints how
   many of the 213 cases fail in each. If `=0` shows far fewer failures, the
   auto-blockify pass is the common trigger (matching §1.1-10).

3. For ops still flaky with the knob off, use the raw reproducer per op and then
   reduce to the bare-kernel repro, guided by the specific defect (atomics →
   §1.1-10/§1.3-13; store-then-load → §1.3-12; masked reduction / where-0 →
   §1.1-1/–3):

   ```bash
   python tools/diagnose_flaky_op_raw.py --op linear --repeat 20
   TRITON_ALL_BLOCKS_PARALLEL=1 python tools/diagnose_flaky_op_raw.py --op linear --repeat 20
   ```

## Expected conclusion

- Most flakiness is **auto-blockify nondeterminism** (§1.1-10) — a backend
  (flagtree / AscendNPU-IR) issue, reachable whenever the knob is set, plus the
  store→load race (§1.3-12) and cross-program sync (§1.3-13) for in-place /
  atomic / reduction kernels.
- None of it is caused by LaneVectorize; it is pre-existing, config-independent
  flakiness surfaced by the A/B retry bookkeeping.
