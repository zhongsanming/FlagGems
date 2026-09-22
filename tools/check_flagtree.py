#!/usr/bin/env python3
"""Diagnose FlagTree generic-IR printer / cache-invalidating-env mismatch.

Run from the FlagGems repo root:

    python tools/check_flagtree.py

It checks:
  1. installed flagtree/triton versions and the libtriton path/mtime
  2. get_cache_invalidating_env_vars() with flagtree + upstream vars set
  3. python cache.py has _serialize_ir
  4. the .so actually contains the flagtree env-var strings
  5. the generic printer binding (get_asm(print_generic_op_form=True)) works
"""

import os
import time
import importlib.metadata as md

# Set BEFORE importing triton so the C++ getenv sees them.
os.environ["TRITON_DISABLE_LANE_VECTORIZE"] = "1"
os.environ["TRITON_MLIR_PRINT_OP_GENERIC"] = "1"
os.environ["TRITON_F32_DEFAULT"] = "tf32"          # upstream invalidating var, for contrast

FLAGTREE_VARS = [
    "TRITON_DISABLE_LANE_VECTORIZE",
    "TRITON_ENABLE_LANE_VECTORIZE_BLOCK_MODE",
    "TRITON_LANE_VECTORIZE_ALLOW_CONCAT",
    "TRITON_LANE_VECTORIZE_ALLOW_ADDRESS_CONES",
    "TRITON_MLIR_PRINT_OP_GENERIC",
]
UPSTREAM_VAR = "TRITON_F32_DEFAULT"


def show(k, v):
    print(f"{k:28s}: {v}")


print("=" * 74)
print("FlagTree / triton environment diagnosis")
print("=" * 74)

for name in ("flagtree", "triton", "triton-ascend"):
    try:
        show(name, md.version(name))
    except Exception:
        show(name, "<not installed>")

import triton  # noqa: E402

show("triton.__file__", triton.__file__)

libtriton_path = None
try:
    import triton._C.libtriton as lt  # noqa: E402

    libtriton_path = lt.__file__
    show("libtriton.__file__", libtriton_path)
    show("libtriton mtime", time.ctime(os.path.getmtime(libtriton_path)))
except Exception as exc:  # noqa: BLE001
    show("libtriton", f"import failed: {exc!r}")

print("-" * 74)
print("1) get_cache_invalidating_env_vars()")
try:
    from triton._C.libtriton import get_cache_invalidating_env_vars as g

    env = g()
    print("    returned:", env)
    show("  flagtree var seen", "TRITON_DISABLE_LANE_VECTORIZE" in env)
    show("  generic var seen", "TRITON_MLIR_PRINT_OP_GENERIC" in env)
    show("  upstream var seen", UPSTREAM_VAR in env)
except Exception as exc:  # noqa: BLE001
    print("    FAILED:", repr(exc))

print("-" * 74)
print("2) python cache.py")
try:
    import triton.runtime.cache as cache_mod

    show("  cache.py", cache_mod.__file__)
    show("  has _serialize_ir", hasattr(cache_mod, "_serialize_ir"))
except Exception as exc:  # noqa: BLE001
    print("    FAILED:", repr(exc))

print("-" * 74)
print("3) flagtree env-var strings inside libtriton.so")
if libtriton_path and os.path.exists(libtriton_path):
    blob = open(libtriton_path, "rb").read()
    for var in FLAGTREE_VARS + [UPSTREAM_VAR]:
        show("  " + var, blob.count(var.encode()))
else:
    print("    (libtriton.so not found)")

print("-" * 74)
print("4) generic MLIR printer binding (get_asm)")
# NOTE: a dumped Ascend .ttir carries the 'hacc' dialect, which the standalone
# parser used here does not register, so it cannot be re-parsed. Prove the
# print_generic_op_form binding works on a minimal in-process module instead;
# the end-to-end generic dump is verified by the A/B run (the head of a fresh
# cached .ttir must be '"builtin.module"() ({').
try:
    from triton._C.libtriton import ir

    ctx = ir.context()
    ir.load_dialects(ctx)
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".mlir")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("module {}\n")
        mod = ir.parse_mlir_module(tmp, ctx)
        op = getattr(mod, "operation", mod)
        generic = op.get_asm(print_generic_op_form=True,
                             print_debug_info=False)
        show("  print_generic_op_form works",
             generic.lstrip().startswith('"builtin.module"'))
        show("  locations stripped", "loc(" not in generic)
        print("    head:", generic[:90].replace("\n", " "))
    finally:
        os.unlink(tmp)
except Exception as exc:  # noqa: BLE001
    print("    FAILED:", repr(exc))

print("=" * 74)
print("Verdict hints:")
print("  - all 3 'seen' booleans True -> rebuild is good")
print("  - upstream var seen but flagtree var not -> libtriton.so lacks the merge")
print("  - .so string counts 0 for flagtree vars -> building from an old tree")
print("  - section 4 FAILED -> libtriton.so lacks print_generic_op_form")
print("=" * 74)
