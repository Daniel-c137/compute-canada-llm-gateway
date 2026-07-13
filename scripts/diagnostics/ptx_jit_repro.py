#!/usr/bin/env python3
"""Minimal repro for Alliance Ticket#0317340: the deployed CUDA driver rejects the PTX
shipped inside wheelhouse vllm>=0.22 kernel libraries.

Method: extract the first embedded PTX member from a given .so (cuobjdump) and ask the
driver to JIT it directly via cuModuleLoadData — no vLLM, no torch, no model, any 1 GPU.
This sidesteps kernel-shape effects entirely: if the PTX ISA version exceeds what the
driver's JIT supports, cuModuleLoadData fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)
deterministically.

Expected on Rorqual H100 nodes (driver 580.159.04 = CUDA 13.0):
  vllm 0.22.0 _moe_C.abi3.so                (.version 9.2) -> CUDA_ERROR_UNSUPPORTED_PTX_VERSION
  vllm 0.24.0 _moe_C_stable_libtorch.abi3.so (.version 9.2) -> CUDA_ERROR_UNSUPPORTED_PTX_VERSION
  vllm 0.20.0 _moe_C.abi3.so                (.version 8.8) -> CUDA_SUCCESS

Usage:  module load StdEnv/2023 cuda python/3.11
        python3 ptx_jit_repro.py /path/to/kernels.abi3.so
"""
import ctypes
import os
import re
import subprocess
import sys
import tempfile

so = sys.argv[1]
cu = ctypes.CDLL("libcuda.so.1")


def err_name(rc):
    name = ctypes.c_char_p()
    cu.cuGetErrorName(rc, ctypes.byref(name))
    return (name.value or b"?").decode()


rc = cu.cuInit(0)
if rc != 0:
    sys.exit(f"cuInit failed: {err_name(rc)} — run on a node with (any) 1 GPU")
drv = ctypes.c_int()
cu.cuDriverGetVersion(ctypes.byref(drv))
print(f"driver CUDA version: {drv.value // 1000}.{(drv.value % 1000) // 10}")

listing = subprocess.run(["cuobjdump", "--list-ptx", so], capture_output=True, text=True).stdout
m = re.search(r"PTX file\s+\d+:\s+(\S+)", listing)
if not m:
    sys.exit(f"no PTX members embedded in {so}")
member = m.group(1)

with tempfile.TemporaryDirectory() as td:
    subprocess.run(["cuobjdump", "-xptx", member, so], cwd=td, check=True, capture_output=True)
    ptx = open(os.path.join(td, member), "rb").read()
ver = re.search(rb"\.version\s+([0-9.]+)", ptx)
print(f"member: {member} | PTX ISA .version {ver.group(1).decode() if ver else '?'} | {len(ptx)} bytes")

dev = ctypes.c_int()
cu.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
rc = cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
if rc != 0:
    sys.exit(f"cuCtxCreate failed: {err_name(rc)}")
mod = ctypes.c_void_p()
rc = cu.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(ptx + b"\0"))
if rc == 0:
    print("RESULT: CUDA_SUCCESS — the driver JIT-compiled this library's PTX fine")
else:
    print(f"RESULT: {err_name(rc)} (rc={rc}) — the driver cannot JIT this library's PTX")
