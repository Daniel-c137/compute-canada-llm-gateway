# DRAFT: Alliance wiki note — vLLM wheelhouse wheels ≥ 0.22 require `module load cuda/13`

Status: draft for the Alliance vLLM wiki page, offered on support ticket #0317340 (staff invited a
contribution). Written in wiki-neutral voice; trim to taste.

---

## vLLM ≥ 0.22: load `cuda/13`, not `cuda/12.x`

Wheelhouse builds of `vllm>=0.22` (and their bundled kernel libraries, e.g. `_moe_C`,
vllm-flash-attn) are compiled with a CUDA 13.2-class toolchain. Their embedded PTX carries ISA
`.version 9.2`, which the CUDA 13.0 node driver's JIT cannot compile. If the job loads
`cuda/12.9` — or bare `cuda`, which defaults to 12.6 — the process binds the system `libcuda`
and fails at runtime with:

```
CUDA error (...): the provided PTX was compiled with an unsupported toolchain.
```

or, at the driver API level, `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (rc=222)`.

**Fix:** load the cuda module matching the wheel's build vintage:

```bash
module load StdEnv/2023 gcc/12.3 cuda/13 python/3.11   # for vllm >= 0.22
```

`cuda/13` prepends `cudacompat/13.2` (NVIDIA forward-compatibility userspace,
`libcuda.so.595.*`) to `LD_LIBRARY_PATH`, which raises the driver's PTX-JIT ceiling so the
13.2-built PTX loads. Older wheels (e.g. `vllm==0.20.0+computecanada`, PTX ISA 8.8) work under
`cuda/12.x` as before.

### Symptoms this explains

- vLLM multi-node or single-node serve crashes at the first attention/MoE kernel launch with the
  "unsupported toolchain" message, sometimes many minutes into startup (after weight loading).
- Crashes that appear only on certain request shapes: common shapes hit precompiled SASS and
  work; shapes that need the embedded PTX fail. A serve can pass `/health` and small requests,
  then die on a slightly larger prompt.

### 30-second compatibility check (1 GPU, stdlib only)

The script below extracts an embedded PTX member from a kernel library and asks the driver to JIT
it directly (`cuModuleLoadData`) — deterministic, shape-independent:

```bash
salloc --gpus=h100:1 --time=00:15:00 --account=<acct>
module load StdEnv/2023 cuda/13 python/3.11   # test under the module set you plan to serve with
python3 ptx_jit_repro.py <venv>/lib/python3.11/site-packages/vllm/_moe_C.abi3.so
# CUDA_SUCCESS                        -> this wheel works under the loaded cuda module
# CUDA_ERROR_UNSUPPORTED_PTX_VERSION  -> wrong cuda module for this wheel
```

(Script: `ptx_jit_repro.py`, ~60 lines, ctypes + cuobjdump; attach or link on the wiki page.)

### Rules of thumb

- Pin the cuda module to the wheel vintage; never `module load cuda` bare (defaults to 12.6).
- Multi-node note: vLLM's data-parallel startup handshake timeouts are tight for first-launch
  cold caches on Lustre; point `TRITON_CACHE_DIR`, `XDG_CACHE_HOME`, and `CUDA_CACHE_PATH` at
  `$SLURM_TMPDIR` to avoid cross-node filelock stalls.
