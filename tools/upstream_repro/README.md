# Upstream repro: Ascend auto-blockify nondeterminism

Minimal, standalone reproducers (`torch` + `triton` only) for nondeterministic
results on Ascend when the flagtree **auto-blockify** pass is enabled.

Confirmed on: `Ascend910B4`, `torch-npu 2.9.0.post2`, `triton 3.5.1`,
`flagtree 0.7.0+ascend.git0afb1367`, bundled AscendNPU-IR
`3545d1cba9b1bdd3cb300724d4626282ad1679ee`.

## Trigger

`TRITON_ALL_BLOCKS_PARALLEL=1` (flagtree reads it at compile time to enable
auto-blockify / `--enable-auto-blockify-loop`,
`third_party/ascend/backend/utils.py::_is_auto_map_parallel_blocks_enabled`).

```bash
# flaky: different hash (and partially-updated elements) every run
TRITON_ALL_BLOCKS_PARALLEL=1 python repro.py

# stable control: same hash every run
python repro.py
```

`repro.py` is an aliased in-place elementwise kernel (`x = x * s`,
`x_ptr == out_ptr`). `repro_reduce.py` is the reduction case:
`tl.max(..., return_indices=True)` / `tl.argmax` are flaky while `tl.max`/`tl.sum`
are stable — this is what breaks the two-pass global argmax.

`TRITON_ALWAYS_COMPILE=1` is set inside the scripts because the flag is **not**
part of the triton cache key.

## Note on FlagGems

FlagGems used to leak `TRITON_ALL_BLOCKS_PARALLEL=1` at import time
(`sparse_attention.py` module level + `from flag_gems.fused import *`), which is
what made `import flag_gems` look like the trigger for unrelated kernels. That is
fixed separately; these repros need no FlagGems import.

`bisect_flag_gems_import_trigger.py` is the cache-isolated bisection that pinned
the trigger down (setup levels: triton_only / seed_cpu / seed_device /
all_blocks_parallel / import_flag_gems).

## Suggested upstream title

> "Auto-blockify (`TRITON_ALL_BLOCKS_PARALLEL=1`, `--enable-auto-blockify-loop`)
> produces nondeterministic code: aliased in-place elementwise stores and
> `tl.max(..., return_indices=True)` return different results on identical input."
