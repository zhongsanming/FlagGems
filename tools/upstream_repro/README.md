# Upstream reproducers: Ascend NPU nondeterminism

Two self-contained reproducers for nondeterministic results on Ascend
(confirmed on `Ascend910B4`, `torch-npu 2.9.0.post2`, `triton 3.5.1`,
`flagtree 0.7.0+ascend.git0afb1367`, bundled AscendNPU-IR
`3545d1cba9b1bdd3cb300724d4626282ad1679ee`).

Both use **only `torch` + `triton`** (no FlagGems). The kernels are plain
`@triton.jit` functions, so the nondeterminism is produced by the
triton/flagtree -> AscendNPU-IR compilation-and-execution path.

| script | bug |
| --- | --- |
| `repro_inplace_alias_nondeterminism.py` | an aliased in-place elementwise kernel (`x = x * s`, `x_ptr == out_ptr`) returns a different, partially-updated result every run |
| `repro_reduce_return_indices_nondeterminism.py` | `tl.max(x, axis=0, return_indices=True)` (and `tl.argmax`) return a different value/index every run; plain `tl.max`/`tl.sum`/`tl.min` are stable |

Run:

```bash
python repro_inplace_alias_nondeterminism.py --repeat 20 --device 0
python repro_reduce_return_indices_nondeterminism.py --repeat 20 --device 0
```

Each prints the env fingerprint, per-variant determinism, and (for the alias
one) the number of mismatching elements.

## Where the bug lives (flagtree vs AscendNPU-IR)

The compilation split is:

```
TTIR --(flagtree passes: ... add_triton_to_linalg)--> linalg / TTAdapter   [flagtree]
     --(bishengir-compile)--> npubin                                       [AscendNPU-IR]
```

* Flagtree's output for the aliased in-place kernel is well-formed and already
  ordered: it loads `x_ptr` into a distinct `memref.alloc()` temp, computes, and
  only then stores into the `output_ptr` memref (see the tail of
  `repro_inplace_alias_nondeterminism.py`). There is no aliasing ambiguity in
  the IR, so flagtree is not the origin.
* The wrong result appears only after `bishengir-compile`, whose first
  memory-ordering stage is the HIVM graph sync solver
  (`bishengir/lib/Dialect/HIVM/Transforms/GraphSyncSolver/`), followed by
  memory planning / auto multi-buffer. That is AscendNPU-IR code.

Ruled out as causes on the affected build (tested via the same kernels):
`num_stages=1`, `multibuffer=False`, `tl.debug_barrier()` before the store, an
explicit device `synchronize()`, and whether the buffer came from `clone()`. None
of them stabilise the result.

## How to bisect the flagtree/AscendNPU-IR boundary (if requested)

Dump the flagtree linalg and replay it through AscendNPU-IR alone:

```bash
# 1. produce kernel.ttadapter.mlir for one kernel (flagtree output), e.g. by
#    compiling the repro kernel once; flagtree writes it under the Triton cache.
# 2. feed it to the standalone AscendNPU-IR driver and execute the result.
bishengir-compile kernel.ttadapter.mlir --target=Ascend910B4-1 -o kernel.o
```

If replaying the same `.ttadapter` directly through `bishengir-compile` (no
flagtree/triton Python) is nondeterministic, the defect is unambiguously in
AscendNPU-IR. To see the sync decisions, add:

```bash
BISHENGIR_DUMP_IR_AFTER_ALL=1 bishengir-compile kernel.ttadapter.mlir ... 2> ir.log
# then inspect the `hivm-graph-sync-solver` stage for a missing sync between
# the load and the store (alias case) or around the reduce (index case).
```

## Suggested upstream titles

* "hivm-graph-sync-solver / memory planning produce nondeterministic results for
  an aliased in-place elementwise kernel (`x = x * s`)"
* "Block reduction with `return_indices=True` (`tl.max`/`tl.argmax`) is
  nondeterministic; plain `tl.max`/`tl.sum` are stable"
