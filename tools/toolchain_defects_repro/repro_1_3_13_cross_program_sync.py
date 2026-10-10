#!/usr/bin/env python3
"""Defect 1.3-13: cross-program synchronization (atomics/spin barrier) unreliable.

Doc claim: an atomic spin barrier across >=8 programs does not make the written
data visible (448/512 wrong); <=4 programs "happens to be" correct. Workaround:
never rely on cross-program sync; order via kernel boundaries.

This kernel has each program write its slot in pass 1, spin-wait on a flag until
all programs have arrived, then read a neighbour's slot and check it is visible.

  A) spin_barrier : atomic flag + spin across P programs   (suspect for P>=8)
  B) two_kernel   : write kernel, then read kernel          (workaround is
                    implicitly correct: kernel 1 finishes before kernel 2)

We report, for a range of program counts, how many programs saw stale/garbage
data in the spin version.

Run: python repro_1_3_13_cross_program_sync.py [--counts 2,4,8,16]
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="2,4,8,16")
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is not None:
        for v in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
            os.environ[v] = str(args.device)

    import torch
    import triton
    import triton.language as tl

    B = args.b

    @triton.jit
    def spin_barrier(data_ptr, flag_ptr, out_ptr, P, B: tl.constexpr):
        pid = tl.program_id(0)
        ar = tl.arange(0, B)
        # pass 1: every program writes its own row = pid, then bumps the flag
        tl.store(data_ptr + pid * B + ar, tl.full((B,), 1.0, tl.float32) * (pid + 1))
        tl.atomic_add(flag_ptr, 1)
        # spin until all P programs arrived
        while tl.load(flag_ptr, volatile=True) < P:
            pass
        # read the previous program's row (wrap); must be visible
        other = (pid - 1) % P
        v = tl.load(data_ptr + other * B + ar)
        tl.store(out_ptr + pid * B + ar, v)

    @triton.jit
    def write_rows(data_ptr, P, B: tl.constexpr):
        pid = tl.program_id(0)
        ar = tl.arange(0, B)
        tl.store(data_ptr + pid * B + ar,
                 tl.full((B,), 1.0, tl.float32) * (pid + 1))

    @triton.jit
    def read_prev(data_ptr, out_ptr, P, B: tl.constexpr):
        pid = tl.program_id(0)
        ar = tl.arange(0, B)
        other = (pid - 1) % P
        v = tl.load(data_ptr + other * B + ar)
        tl.store(out_ptr + pid * B + ar, v)

    def _exp(P):
        return torch.tensor([((p - 1) % P) + 1 for p in range(P)],
                            dtype=torch.float32).unsqueeze(1).expand(P, B)

    def run_spin(P):
        data = torch.zeros(P * B, dtype=torch.float32, device=dev)
        flag = torch.zeros(1, dtype=torch.int32, device=dev)
        out = torch.empty(P * B, dtype=torch.float32, device=dev)
        spin_barrier[(P,)](data, flag, out, P, B=B)
        o = out.detach().to("cpu").reshape(P, B)
        n_zero = int((o == 0).sum().item())
        bad = int((o != _exp(P)).sum().item())
        return bad, P * B, n_zero

    def run_two_kernel(P):
        data = torch.zeros(P * B, dtype=torch.float32, device=dev)
        out = torch.empty(P * B, dtype=torch.float32, device=dev)
        write_rows[(P,)](data, P, B=B)
        read_prev[(P,)](data, out, P, B=B)      # kernel boundary orders it
        o = out.detach().to("cpu").reshape(P, B)
        bad = int((o != _exp(P)).sum().item())
        return bad, P * B, int((o == 0).sum().item())

    dev = "npu" if hasattr(torch, "npu") else "cuda"
    counts = [int(x) for x in args.counts.split(",")]
    print(f"B={B} counts={counts} repeat={args.repeat}")
    reproduced = False
    for P in counts:
        bad_total = zero_total = 0
        err = None
        for _ in range(args.repeat):
            try:
                b, tot, nz = run_spin(P)
                bad_total = max(bad_total, b)
                zero_total = max(zero_total, nz)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:100]}"
                break
        if err:
            print(f"  spin  P={P:3d}: exception {err}")
        else:
            print(f"  spin  P={P:3d}: worst wrong = {bad_total}/{P*B} "
                  f"(stale-zero elems={zero_total})")
            if P >= 8 and bad_total > 0:
                reproduced = True
        # control: two kernels (kernel boundary orders the write before the read)
        cb, _, _ = run_two_kernel(P)
        print(f"  2krn  P={P:3d}: wrong = {cb}/{P*B}")
    if reproduced:
        print("REPRODUCED: cross-program spin barrier does not guarantee "
              "visibility for >=8 programs.")
        return 0
    print("Not reproduced (may need specific device/timing).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
