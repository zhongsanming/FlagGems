# Upstream reproducers: Ascend NPU nondeterminism

Nondeterministic results on Ascend (confirmed on `Ascend910B4`,
`torch-npu 2.9.0.post2`, `triton 3.5.1`,
`flagtree 0.7.0+ascend.git0afb1367`, bundled AscendNPU-IR
`3545d1cba9b1bdd3cb300724d4626282ad1679ee`).

## Root cause (confirmed by cache-isolated bisection)

The nondeterminism is only produced when the flagtree Ascend **auto-blockify**
pass is enabled, which is controlled by the env var
**`TRITON_ALL_BLOCKS_PARALLEL`**.

`bisect_flag_gems_import_trigger.py --fresh-cache`:

```
triton_only            deterministic=True
seed_cpu               deterministic=True
seed_device            deterministic=True
all_blocks_parallel    deterministic=False   <-- TRITON_ALL_BLOCKS_PARALLEL=1 alone
import_flag_gems       deterministic=True    <-- after the FlagGems leak fix
```

`TRITON_ALL_BLOCKS_PARALLEL=1` is read by flagtree at TRITON COMPILE time
(`third_party/ascend/backend/utils.py::_is_auto_map_parallel_blocks_enabled`);
when set, `compiler.py` uses `metadata.auto_blockify_size` instead of 1 and adds
`--enable-auto-blockify-loop`, i.e. it runs auto-blockify for every kernel
compiled afterwards in the process.

There are two distinct issues:

1. **FlagGems (fixed here, `ca3563fa`).** `flag_gems/runtime/backend/_ascend/
   fused/sparse_attention.py` set `os.environ["TRITON_ALL_BLOCKS_PARALLEL"]="1"`
   at module import, and `flag_gems/__init__.py` does
   `from flag_gems.fused import *`, so importing flag_gems enabled auto-blockify
   globally for the process. Now scoped to the sparse-attention launch via a
   context manager.

2. **flagtree / AscendNPU-IR (still upstream).** With auto-blockify enabled, the
   emitted code is nondeterministic for the kernels below. This is the part to
   report upstream.

## The two upstream reproducers

Both use **only `torch` + `triton`** (no FlagGems), and **set
`TRITON_ALL_BLOCKS_PARALLEL=1` explicitly** (no FlagGems import needed).

| script | bug |
| --- | --- |
| `repro_inplace_alias_nondeterminism.py` | aliased in-place elementwise kernel (`x = x * s`, `x_ptr == out_ptr`) returns a different, partially-updated result every run; out-of-place control is stable |
| `repro_reduce_return_indices_nondeterminism.py` | `tl.max(x, axis=0, return_indices=True)` and `tl.argmax` are nondeterministic; plain `tl.max`/`tl.sum`/`tl.min` and the `tl.min(where(x==max))` shape are stable |

Run (always with a fresh cache — `TRITON_ALL_BLOCKS_PARALLEL` is NOT part of the
triton cache key, so a stale binary can mask the effect):

```bash
TRITON_ALL_BLOCKS_PARALLEL=1 python repro_inplace_alias_nondeterminism.py --device 0 --repeat 20 --fresh-cache
TRITON_ALL_BLOCKS_PARALLEL=1 python repro_reduce_return_indices_nondeterminism.py --device 0 --repeat 20 --fresh-cache
```

Each prints the env fingerprint and per-variant determinism (the alias one also
prints the number of mismatching elements).

Ruled out as causes on the affected build: `num_stages=1`,
`multibuffer=False`, `tl.debug_barrier()` before the store, an explicit device
`synchronize()`, and whether the buffer came from `clone()`.

## Suggested upstream title (flagtree / AscendNPU-IR)

> "Auto-blockify (`TRITON_ALL_BLOCKS_PARALLEL=1`, `--enable-auto-blockify-loop`)
> produces nondeterministic code: aliased in-place elementwise stores and
> `tl.max(..., return_indices=True)` return different results on identical input."

Attach the two repro scripts and the `all_blocks_parallel=False` bisect row.

## Optional: pin the boundary if asked

flagtree's linalg output for the aliased kernel is well-formed (loads into a
distinct `memref.alloc` temp before storing), so the defect is in the
auto-blockify/scheduling stage. To replay through AscendNPU-IR alone, dump the
linalg and feed it to `bishengir-compile` with and without
`--enable-auto-blockify-loop`:

```bash
bishengir-compile kernel.ttadapter.mlir --target=Ascend910B4-1 \
    --enable-auto-blockify-loop -o kernel.o
BISHENGIR_DUMP_IR_AFTER_ALL=1 bishengir-compile kernel.ttadapter.mlir ... 2> ir.log
```
