#!/usr/bin/env python3
"""Minimal reproducer: aliased in-place elementwise kernel is nondeterministic.

Bug summary
-----------
On Ascend (AscendNPU-IR build bundled with flagtree 0.7.0+ascend.git0afb1367,
i.e. AscendNPU-IR revision 3545d1cba9b1bdd3cb300724d4626282ad1679ee), a triton
kernel that reads and writes the SAME buffer (an in-place elementwise op,
``x = x * s``) returns a DIFFERENT, partially-updated result on every run of an
identical input.

The equivalent out-of-place kernel (``y = x * s``, separate buffers) is stable,
so the fault is specific to the aliased load/store dataflow.

Scope / ownership
-----------------
This script uses ONLY torch + triton. It does not import FlagGems, and the
kernel is a plain ``@triton.jit`` function, so the nondeterminism is produced by
the triton/flagtree -> AscendNPU-IR compilation+execution path.

Flagtree's own lowering is correct: the linalg/ttadapter IR it emits for this
kernel loads x into a distinct ``memref.alloc`` temp, computes, then stores, so
the read fully precedes the write (see the note at the bottom). The bad result
appears only after ``bishengir-compile`` (AscendNPU-IR) turns that IR into a
binary, which points at AscendNPU-IR memory planning / HIVM sync / (auto)
multi-buffer handling of the two aliased memrefs.

What is NOT the cause (already ruled out on the affected build)
---------------------------------------------------------------
* num_stages=1                          -> still nondeterministic
* multibuffer=False (+ num_stages=1)    -> still nondeterministic
* tl.debug_barrier() before the store   -> still nondeterministic
* an explicit device synchronize()      -> still nondeterministic
* a preceding clone() vs a fresh buffer -> still nondeterministic

Run
---
    python repro_inplace_alias_nondeterminism.py
    python repro_inplace_alias_nondeterminism.py --repeat 20 --n 1048576
    python repro_inplace_alias_nondeterminism.py --device 2

Expected output: the ``aliased`` block prints ``deterministic: False`` with
several distinct hashes and ~BLOCK non-zero mismatches, while the ``oop`` control
prints ``deterministic: True`` with 0 mismatches.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys


def env_fingerprint() -> dict:
    import importlib.metadata as md

    fp = {}
    try:
        import triton

        fp["triton_version"] = getattr(triton, "__version__", None)
        fp["triton_file"] = getattr(triton, "__file__", None)
    except Exception as exc:  # noqa: BLE001
        fp["triton_version"] = f"ERR:{exc}"
    for pkg in ("flagtree", "torch", "torch-npu"):
        try:
            fp[f"{pkg}_version"] = md.version(pkg)
        except Exception:  # noqa: BLE001
            fp[f"{pkg}_version"] = None
    return fp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1024 * 1024)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--scalar", type=float, default=-0.999)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--device", default=None,
                    help="accelerator id to confine to (sets ASCEND_RT_VISIBLE_DEVICES)")
    args = ap.parse_args()

    if args.device is not None:
        for var in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[var] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    # ---- kernels -----------------------------------------------------------
    @triton.jit
    def mul_oop(x_ptr, y_ptr, s, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        tl.store(y_ptr + off, tl.load(x_ptr + off, mask=m) * s, mask=m)

    @triton.jit
    def mul_inplace(x_ptr, s, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        x = tl.load(x_ptr + off, mask=m)
        tl.store(x_ptr + off, x * s, mask=m)

    def _hash(t):
        return hashlib.sha1(
            t.detach().to("cpu").contiguous().numpy().tobytes()
        ).hexdigest()[:16]

    dev = "npu" if hasattr(torch, "npu") else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    base = torch.randn(args.n, dtype=torch.float32, device=dev)
    ref = (base.to("cpu").double() * args.scalar).to(torch.float32)
    grid = (triton.cdiv(args.n, args.block),)

    def run(kernel, *extra, inplace):
        x = base.clone()
        if inplace:
            kernel[grid](x, args.scalar, args.n, BLOCK=args.block, *extra)
            return x
        y = torch.empty_like(base)
        kernel[grid](base, y, args.scalar, args.n, BLOCK=args.block, *extra)
        return y

    import json

    print(json.dumps({"env": env_fingerprint(),
                      "n": args.n, "block": args.block,
                      "scalar": args.scalar, "repeat": args.repeat}, indent=2))

    results = {}
    for name, onerun in (
        ("oop", lambda: run(mul_oop, inplace=False)),
        ("aliased", lambda: run(mul_inplace, inplace=True)),
    ):
        hashes, nbad = [], []
        for _ in range(args.repeat):
            out = onerun()
            hashes.append(_hash(out))
            r = out.to("cpu")
            nbad.append(int((~torch.isclose(r, ref, atol=1e-4, rtol=1e-4)).sum()))
        results[name] = {
            "hashes": hashes,
            "n_mismatch": nbad,
            "deterministic": len(set(hashes)) == 1,
        }
        print(f"\n[{name}] deterministic={results[name]['deterministic']}")
        print(f"  distinct hashes: {len(set(hashes))}")
        print(f"  n_mismatch per run: {nbad}")

    ok = (results["oop"]["deterministic"]
          and not results["aliased"]["deterministic"])
    if ok:
        print("\nREPRODUCED: out-of-place is deterministic, aliased in-place is not.")
    else:
        print("\nDid NOT reproduce the expected pattern this run; increase "
              "--repeat or check the compiler/device.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Supporting evidence (flagtree linalg/ttadapter IR for the aliased kernel,
# i.e. the input to AscendNPU-IR's bishengir-compile). The read is fully
# materialised into a distinct temp before the write to the output memref:
#
#   %x_4 = memref.reinterpret_cast %x_ptr      ...   // read source
#   %x_5 = memref.alloc() : memref<1024xf16>          // temp buffer
#   memref.copy %x_11, %x_12                          // load into temp
#   %out_16 = arith.mulf %out, %out_15                // x * scalar
#   %reinterpret_cast = memref.reinterpret_cast %output_ptr ...  // write dest
#   bufferization.materialize_in_destination %extracted_slice in writable %subview
#
# So flagtree's lowering preserves the load-then-store ordering; the wrong
# result must be introduced downstream by AscendNPU-IR (memory planning / HIVM
# graph sync solver / multi-buffer over the two aliased memrefs).
# ---------------------------------------------------------------------------
