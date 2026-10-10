# Reproducers for the documented Ascend toolchain defects

Standalone reproducers (torch + triton only; **no FlagGems import**) for the
issues in `matrix_rank_昇腾工具链缺陷与精度问题.md`. Each script prints a
verdict: whether the defect reproduced, with a *correct implementation* run
alongside the suspect one so you can confirm the difference is real.

Methodology follows the doc: a tiny kernel compared against a CPU reference,
run N times to separate deterministic miscompiles from races.

## Run everything

```bash
for f in tools/toolchain_defects_repro/repro_*.py; do
    echo "===== $f ====="; python "$f"; echo
done
```

Set a device with `--device <id>` (each script accepts it), and use a fresh
cache when testing a compile-time knob:
`TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR=$(mktemp -d) python <script>`.

## Coverage

### Backend miscompiles / crashes (§1.1)

| doc § | script | suspect vs correct impl |
|---|---|---|
| 1.1-1 | `repro_1_1_1_where_scalar_zero.py` | `tl.where(m, x, 0.0)` vs `x * m.to(f32)` |
| 1.1-2 | `repro_1_1_2_reduce_broadcast.py` | `v[:,None]*w[None,:]` vs `tl.reshape` outer |
| 1.1-3 | `repro_1_1_3_axis0_masked_reduce.py` | axis-0 masked reduce vs `tl.trans`+axis-1 |
| 1.1-4 | `repro_1_1_4_arange_eq_mask.py` | `arange==C` mask vs interval mask |
| 1.1-5 | `repro_1_1_5_bool_sum.py` | `tl.sum(bool)` vs `tl.sum(bool.to(int32))` |
| 1.1-6 | `repro_1_1_6_3d_intermediate.py` | 3D `[BP,B,B]` where/reduce vs 2D |
| 1.1-7 | `repro_1_1_7_dynamic_while.py` | `while` carrying a tile vs `tl.range`+`static_range` |
| 1.1-9 | `repro_1_1_9_scalar_bitcast.py` | scalar bitcast vs vector bitcast |
| 1.1-10 | `repro_1_1_10_atomic_allblocks.py` | atomic_add with/without `TRITON_ALL_BLOCKS_PARALLEL` |
| 1.1-11 | `repro_1_1_11_masked_load_runtime_scalar.py` | 5 formulations of masked-load + runtime scalar |

### Cube / dot (§1.2)

| doc § | script | checks |
|---|---|---|
| 1.2 | `repro_1_2_dot_operands.py` | `tl.dot(a,b,acc)` vs `acc+tl.dot(a,b)`; swapped-addr load vs `tl.trans`; computed operand compile |

### Memory order / sync (§1.3)

| doc § | script | suspect vs correct impl |
|---|---|---|
| 1.3-12 | `repro_1_3_12_store_load_order.py` | store→load in one kernel vs two kernels |
| 1.3-13 | `repro_1_3_13_cross_program_sync.py` | atomic spin barrier across P programs |

### Resource / compile limits (§1.4)

| doc § | script | checks |
|---|---|---|
| 1.4-14/15 | `repro_1_4_ub_and_fp64.py` | (128,128) fp32 tile UB wall; fp64 kernel compile |
| 1.4-16 | `repro_1_4_loop_structure.py` | `static_range`+`range` same fn; jit helper; scalar `==` branch |

### Performance (§1.5)

| doc § | script | checks |
|---|---|---|
| 1.5-17 | `repro_1_5_jit_dispatch_latency.py` | triton JIT ~460us vs torch native ~31us |

### Precision (§2.4)

| doc § | script | suspect vs correct impl |
|---|---|---|
| 2.4-9 | `repro_2_4_9_pow2_floor.py` | `exp2(floor(log2 x))` vs exponent bitmask (vector) |

## Not turned into a standalone kernel (documented, not simple backend bugs)

These are algorithm/kernel-specific or environment observations, so a generic
repro would be misleading. They are recorded here for reference:

* **1.1-8 BLOCK=64 fused-kernel "miscompile lottery"** — needs the *actual*
  bidiag+Sturm fused kernel; the doc lists failing K values (33,51,53,…,62) and
  notes it drifts across toolchain generations. To reproduce, use the original
  kernel and scan K=3..64 with a CPU reference. BLOCK<=32 is stable.
* **1.4-16** partially covered; the exact `ConvertLinalgRToBinary` crash needs
  the original loop+helper shape.
* **1.5-18/19/20** — perf ceilings (`tl.argmax` cost, `tl.trans`/gather cost,
  `do_bench_npu` KERNEL-mode broken for multi-kernel ops); measurement advice,
  not correctness bugs.
* **1.6 environment/infra** — shared-machine flakiness, CANN 8.5.0→9.1.1 UB
  regressions, Triton 3.2→3.5 launcher ABI, `triton.knobs` import error. These
  are environment-generation facts; verify by changing the toolchain.
* **2.1-1/2/3** — precision of *algorithm choices* (Gram squared-domain floor,
  fp32 dd/ee, unpivoted QR |R_ii|≠σ_i). Reproduce by comparing the given
  algorithm vs an SVD-grade path on low-rank / slowly-decaying spectra, not with
  a tiny kernel.
* **2.2-4/5/6** — torch threshold *semantics* (tol=max(atol,rtol·σmax), strict
  `>`, double-negative tolerance, 0-D tensor tolerance). These are correctness of
  the FlagGems implementation vs torch, verifiable by comparing
  `matrix_rank` outputs; not a backend miscompile.
* **2.3-7/8**, **2.4-10** — bidiagonalization LAPACK conventions, tau² overflow
  grouping, single-matrix dynamic range. Algorithmic; need the matrix_rank
  kernels and adversarial spectra.

## Notes

* The scripts deliberately avoid importing FlagGems, so they are safe to run
  anywhere torch+triton (Ascend) is available.
* Several defects are probabilistic ("lottery"); bump `--repeat` and vary shapes
  if a run reports "not reproduced".
* §1.1-10 is the one most directly tied to the A/B flakiness: the env var is
  set globally at import by FlagGems' fused sparse-attention module (fixed), and
  it miscompiles `atomic_add` (used by `linear`, `index_put`, `scatter_reduce`).
