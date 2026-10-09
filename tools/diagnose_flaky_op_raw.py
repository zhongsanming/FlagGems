#!/usr/bin/env python3
"""In-process reproducer for flaky FlagGems Ascend ops.

Calls the FlagGems operator directly (exactly like the accuracy test) N times on
the SAME fixed input and reports, per run, an output fingerprint and the max
mismatch vs the CPU reference. If the fingerprint changes across runs on
identical input, the op is nondeterministic -> that is the flakiness.

It also lets you flip the prime suspect knob (TRITON_ALL_BLOCKS_PARALLEL, which
enables the Ascend auto-blockify pass) via the environment, so you can check
whether determinism is restored with it off.

Supported ops: linear, cat, sum_dim, index_put, index_put_, adaptive_max_pool3d,
avg_pool3d, tril, tril_out, scatter_reduce, slice_backward, var_dim,
reflection_pad2d, reflection_pad2d_out, moe_sum, index_copy, index_copy_,
var_correction, conj_physical. Unknown names fall back to `getattr(flag_gems, op)`.

IMPORTANT: run each environment variation in its OWN process, e.g.

    python tools/diagnose_flaky_op_raw.py --op linear --repeat 20
    TRITON_ALL_BLOCKS_PARALLEL=1 python tools/diagnose_flaky_op_raw.py --op linear --repeat 20
    TRITON_ALWAYS_COMPILE=1 python tools/diagnose_flaky_op_raw.py --op linear --repeat 20

The first prints a JSON summary; compare the `deterministic` field across the
three invocations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))


def _fp(t):
    import torch

    t = t.detach().to("cpu").contiguous()
    if t.is_floating_point():
        t = t.float()
    return hashlib.sha1(t.numpy().tobytes()).hexdigest()[:12]


def _maxdiff(a, b):
    import torch

    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        a = a[0] if isinstance(a, (tuple, list)) else a
        b = b[0] if isinstance(b, (tuple, list)) else b
    a = a.detach().to("cpu")
    b = b.detach().to("cpu")
    if a.shape != b.shape:
        return float("inf")
    if a.dtype != b.dtype:
        b = b.to(a.dtype)
    if a.is_floating_point():
        return float((a.float() - b.float()).abs().max())
    return float((a != b).sum())


# ---------------------------------------------------------------------------
# per-op: build fixed inputs (seeded) and return (call(), reference())
# ---------------------------------------------------------------------------

def build_linear(dev):
    import torch

    torch.manual_seed(0)
    x = torch.randn((7, 256), dtype=torch.float32, device=dev)
    w = torch.randn((192, 256), dtype=torch.float32, device=dev)
    b = torch.randn((192,), dtype=torch.float32, device=dev)

    def ref():
        return torch.nn.functional.linear(x.to("cpu"), w.to("cpu"), b.to("cpu"))

    return x, w, b, ref


def build_cat(dev):
    import torch

    torch.manual_seed(0)
    xs = [torch.randn((4, 8), dtype=torch.float32, device=dev) for _ in range(3)]

    def ref():
        return torch.cat([t.to("cpu") for t in xs], dim=0)

    return xs, ref


def build_sum_dim(dev):
    import torch

    torch.manual_seed(0)
    x = torch.randn((200, 2560, 3), dtype=torch.float32, device=dev)

    def ref():
        return torch.sum(x.to("cpu"), dim=1)

    return x, ref


def build_index_put(dev):
    import torch

    torch.manual_seed(0)
    x = torch.zeros((64, 32), dtype=torch.float32, device=dev)
    idx = torch.tensor([0, 3, 5, 60], device=dev)
    vals = torch.randn((4, 32), dtype=torch.float32, device=dev)

    def ref():
        y = torch.zeros((64, 32), dtype=torch.float32)
        y[idx.to("cpu")] = vals.to("cpu")
        return y

    return x, idx, vals, ref


def build_adaptive_max_pool3d(dev):
    import torch

    torch.manual_seed(0)
    x = torch.randn((2, 3, 8, 16, 16), dtype=torch.float32, device=dev)
    osz = (4, 4, 4)

    def ref():
        return torch.nn.functional.adaptive_max_pool3d(x.to("cpu"), osz)

    return x, osz, ref


def build_avg_pool3d(dev):
    import torch

    torch.manual_seed(0)
    x = torch.randn((2, 8, 8, 16, 16), dtype=torch.float16, device=dev)

    def ref():
        return torch.nn.functional.avg_pool3d(x.to("cpu").float(), 2, 2, 0)

    return x, ref


def build_tril(dev):
    import torch

    torch.manual_seed(0)
    x = torch.randn((64, 64), dtype=torch.float32, device=dev)

    def ref():
        return torch.tril(x.to("cpu"))

    return x, ref


def build_slice_backward(dev):
    import torch

    torch.manual_seed(0)
    g = torch.randn((1, 4, 4), dtype=torch.float32, device=dev)
    in_sizes = (1, 8, 8)

    def ref():
        # emulate the reference the test uses
        out = torch.zeros(in_sizes, dtype=torch.float32)
        out[:, 0:1, 0:1] = g.to("cpu")
        return out

    return g, in_sizes, ref


def build_generic(dev, op):
    import torch

    torch.manual_seed(0)
    x = torch.randn((128, 128), dtype=torch.float32, device=dev)

    def ref():
        return x.to("cpu")

    return x, ref


BUILDERS = {
    "linear": build_linear,
    "cat": build_cat,
    "sum_dim": build_sum_dim,
    "index_put": build_index_put,
    "index_put_": build_index_put,
    "adaptive_max_pool3d": build_adaptive_max_pool3d,
    "avg_pool3d": build_avg_pool3d,
    "tril": build_tril,
    "tril_out": build_tril,
    "slice_backward": build_slice_backward,
}


def call_op(op, dev, built):
    """Invoke the FlagGems op on the built inputs; return the output tensor."""
    import torch

    import flag_gems

    with flag_gems.use_gems():
        if op == "linear":
            x, w, b, _ = built
            return flag_gems.linear(x, w, b)
        if op == "cat":
            xs, _ = built
            return torch.cat(xs, dim=0)
        if op == "sum_dim":
            x, _ = built
            return torch.sum(x, dim=1)
        if op in ("index_put", "index_put_"):
            x, idx, vals, _ = built
            if op == "index_put_":
                with flag_gems.use_gems():
                    x.index_put_((idx,), vals)
                return x
            return x.index_put((idx,), vals)
        if op == "adaptive_max_pool3d":
            x, osz, _ = built
            return torch.nn.functional.adaptive_max_pool3d(
                x, osz, return_indices=False)
        if op == "avg_pool3d":
            x, _ = built
            return torch.nn.functional.avg_pool3d(x, 2, 2, 0)
        if op in ("tril", "tril_out"):
            x, _ = built
            return torch.tril(x)
        if op == "slice_backward":
            g, in_sizes, _ = built
            return flag_gems.slice_backward(g, in_sizes, 1, 0, 1, 1)
        fn = getattr(flag_gems, op, None)
        if fn is None:
            raise SystemExit(f"no builder/handler for op {op!r}")
        x, _ = built
        return fn(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", required=True)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--device", default=None,
                    help="accelerator id to confine to")
    args = ap.parse_args(argv)

    if args.device is not None:
        for var in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[var] = str(args.device)

    import torch

    import flag_gems

    dev = getattr(flag_gems, "device", None) or "npu"
    builder = BUILDERS.get(args.op, lambda d: build_generic(d, args.op))
    built = builder(dev)
    ref = built[-1]()

    hashes, dmax, fails = [], [], 0
    for _ in range(args.repeat):
        out = call_op(args.op, dev, built)
        oc = out[0] if isinstance(out, (tuple, list)) else out
        oc = oc.detach().to("cpu")
        hashes.append(_fp(oc))
        d = _maxdiff(ref, oc)
        dmax.append(round(d, 6) if d != float("inf") else "inf")
        fails += 0 if d == 0 else 1

    res = {
        "op": args.op,
        "repeat": args.repeat,
        "deterministic": len(set(hashes)) == 1,
        "distinct_fingerprints": len(set(hashes)),
        "fingerprints": hashes,
        "maxdiff_vs_ref": dmax,
        "env": {k: os.environ.get(k) for k in (
            "TRITON_ALL_BLOCKS_PARALLEL", "TRITON_DISABLE_LANE_VECTORIZE",
            "TRITON_ENABLE_LANE_VECTORIZE_BLOCK_MODE",
            "TRITON_LANE_VECTORIZE_ALLOW_CONCAT",
            "TRITON_LANE_VECTORIZE_ALLOW_ADDRESS_CONES",
            "TRITON_ALWAYS_COMPILE", "TRITON_CACHE_DIR")},
    }
    print(json.dumps(res, indent=2, default=str))
    if not res["deterministic"]:
        print(f"\nNONDETERMINISTIC: {res['distinct_fingerprints']} distinct "
              f"outputs over {args.repeat} runs on identical input.")
    return 0 if res["deterministic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
