# Field guide: serving large (multi-node, agentic) LLMs on Alliance / Compute Canada clusters

This guide captures roughly two weeks of debugging that it took to bring a 523 GB MoE model
(Kimi-K2.6, native INT4) to a **coherent, tool-calling** OpenAI-compatible endpoint on
3 nodes × 4×H100 on **Rorqual**, with vLLM from the Alliance wheelhouse. It complements this
repository's single-node gateway flow with the knowledge needed for bigger models, multi-node
topologies, and **agentic** (tool-calling, multi-turn) workloads.

Everything here was verified on real SLURM jobs; job IDs are cited so claims can be re-checked.
The support-ticket findings are Alliance ticket **#0317340**.

---

## 0. What is proven vs. what is assumed — read this first

All of this is battle-tested for **exactly one model on one cluster**. Do not assume it
generalizes without testing. Honest inventory:

| PROVEN (job-verified) | NOT tested |
|---|---|
| 1 model (Kimi-K2.6), native INT4, MoE, 523 GB | dense models, other sizes, FP8/FP16/AWQ/GPTQ |
| multi-node TP4 × DP3 + expert-parallel | TP-only, pipeline-parallel, other topologies |
| native tool calling (`kimi_k2` parser) | other models' tool parsers |
| 32K context | longer contexts, context scaling |
| Rorqual | Narval, Nibi, Fir, Killarney, TamIA |
| sequential multi-agent against one endpoint | concurrent multi-agent, high-concurrency serving |
| OpenAI-compatible client integration | MCP against a self-hosted model |
| — | measured throughput / tokens-per-sec |

The single most valuable finding (§1.1) is **cluster-universal**, not model-specific.

---

## 1. Cluster-universal findings (any GPU workload on Alliance clusters)

### 1.1 The wheel ↔ cuda-module contract (undocumented; cost us ~2 weeks)

Alliance wheelhouse builds of **vLLM ≥ 0.22** are compiled with a **CUDA 13.2-class toolchain**
(their PTX carries ISA `.version 9.2`), but H100 node drivers are CUDA **13.0** (580.x). If you
`module load cuda/12.9` — or bare `module load cuda`, which silently defaults to **12.6** — your
process binds the *system* `libcuda`, and every kernel that needs its embedded PTX JIT-compiled is
rejected at runtime:

```
CUDA error (...): the provided PTX was compiled with an unsupported toolchain
CUDA_ERROR_UNSUPPORTED_PTX_VERSION (rc=222)
```

**Fix: `module load cuda/13`.** That module prepends **`cudacompat/13.2`** (NVIDIA's
forward-compatibility userspace, `libcuda.so.595.*`) to `LD_LIBRARY_PATH`, raising the driver's
PTX-JIT ceiling. Verified by a 1-GPU A/B (job 15479377): under cuda/12.9 the vLLM 0.22/0.24
kernels' PTX is rejected rc=222; under cuda/13 all of them load `CUDA_SUCCESS`.

Rules of thumb:

- **Pin the cuda module to your wheel's vintage.** `cuda/12.9` for 0.20-era wheelhouse vLLM;
  `cuda/13` for ≥ 0.22. **Never load bare `cuda`.**
- The failure is *silent until runtime* and can hide behind shape effects (§1.2). A serve can pass
  `/health` and small probes on precompiled SASS, then die minutes later when a code path needs PTX.
- Staff confirmed the contract on ticket #0317340 ("recent vLLM (22+) are built with cuda 13") and
  are considering a wiki note. Until it's documented, treat every wheelhouse binary as suspect
  under any cuda module other than its build vintage.

### 1.2 A 30-second, 1-GPU diagnostic that cannot false-negative

`scripts/diagnostics/ptx_jit_repro.py` (stdlib only: ctypes + `cuobjdump`) extracts an embedded
PTX member from any kernel `.so` and asks the driver to JIT it directly via `cuModuleLoadData`.

```bash
salloc --gpus=h100:1 --time=00:15:00 ...
module load StdEnv/2023 cuda/13 python/3.11    # test under the module set you'll serve with
python3 ptx_jit_repro.py path/to/venv/lib/python3.11/site-packages/vllm/_moe_C.abi3.so
```

Why this and not a kernel-call probe: naive 1-GPU kernel tests **false-negative** — common shapes
hit precompiled SASS and pass, while the real serve's shapes need the PTX and die. We lost several
days to exactly this (jobs 15117560 → 15321905). Loading the PTX directly is shape-independent.
Use it as a **pre-submit preflight** for any wheel/driver/module combination.

### 1.3 Compute nodes have no internet

- Stage weights on a **login node**; run jobs with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- Beware **FlashInfer**: it JIT-downloads cubins at *runtime*. Preinstall `flashinfer-cubin` /
  `flashinfer-jit-cache` and pre-warm on the login node, or it dies on the compute node.

### 1.4 Node-local compile caches are mandatory for multi-node

Default cache locations (`~/.triton`, `~/.cache`, `~/.nv`) live on Lustre and cause **cross-node
filelock hangs** (engine-killing, not just slow). Redirect all of:

```bash
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
export XDG_CACHE_HOME="$SLURM_TMPDIR/xdgcache"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export CUDA_CACHE_PATH="$SLURM_TMPDIR/nvcache"     # CUDA driver JIT cache — easy to forget
```

Also set `CUDA_MODULE_LOADING=EAGER` so driver-JIT failures fire loudly at startup instead of
mid-traffic.

### 1.5 Multi-node data-parallel startup timeouts are too tight

vLLM's DP startup timeouts are hardcoded and too short for cold-Lustre `spawn` re-imports:
patch `HANDSHAKE_TIMEOUT_MINS` (`vllm/v1/engine/core.py`) and the coordinator ZMQ wait, and
**re-apply the patches on every env rebuild**. Even patched, DP rendezvous is *flaky* — a rank can
simply fail to join (`2/3 clients joined`, job 15479610, no kernel errors). **A plain re-run
usually clears it**; budget for that in your submission strategy.

### 1.6 Escaping the wheelhouse entirely (when you need upstream binaries)

The Alliance blocks PyPI manylinux wheels via a custom `_manylinux.py` on `PYTHONPATH` plus a
`PIP_CONFIG_FILE` pinning the wheelhouse. Both are user-reversible on the (internet-connected)
login node:

```bash
PYTHONPATH="" PIP_CONFIG_FILE="" python -m venv venv && ...   # pip now resolves pure PyPI
```

The other sanctioned escape is **Apptainer** (`module load apptainer`) with the official
`docker://vllm/vllm-openai` image (`apptainer exec --nv` under srun). Mixing works too: a
wheelhouse env with individual pure-PyPI packages layered in is live-proven on Rorqual.

### 1.7 Scheduling reality (the actual bottleneck)

- Multi-node GPU jobs queued **2–4 days** in our experience (a 2.5 h 3-node job waited 4 days).
- Mitigations that worked: `--time-min` (lets SLURM shrink the job into a backfill hole — dropped
  one wait from days to ~6 h), **verdict-shaped short walltimes** (submit a short job whose only
  purpose is a pass/fail verdict, not a long hold), auto-`scancel` of sibling/ladder jobs, and
  answering every kernel-level question on **1 GPU** first.
- Staff **do** kill jobs that idle GPUs, and said so explicitly. Don't hold an endpoint you are
  not using; drive traffic promptly or exit.

---

## 2. vLLM on SLURM, model-agnostic

### 2.1 Multi-node topology

Alliance staff guidance (matches our working config): **maximize intra-node NVLink** —
`TP = GPUs-per-node`, then **DP across nodes**, plus `--enable-expert-parallel` for MoE. On a
4-GPU/node cluster with 3 nodes:

```
--tensor-parallel-size 4 --data-parallel-size 3 --data-parallel-size-local 1 \
--data-parallel-address $HEAD_IP --data-parallel-rpc-port $PORT --enable-expert-parallel
```

Rank 0 runs the API server; other nodes run `vllm serve ... --headless
--data-parallel-start-rank N`. **Ray multi-node is a dead end here** — a documented-formula Ray
cluster hung 12 h and never served; the Alliance wiki's Ray example is a 125 M-param toy.

### 2.2 A green `/health` is NOT health — canary before traffic

We served **garbage tokens** (multilingual token salad, degenerate repetition) behind a green
`/health` for multiple attempts. Before sending real traffic, assert *content*:

1. **Coherence canary** — a fixed greedy prompt whose response is checked for degenerate
   repetition and multilingual salad (`scripts/diagnostics/judge-canary.py` pattern).
2. **Native tool-call canary** — one `tools=[...]` request; assert `tool_calls` parses and names
   the right function.
3. **Turn-2 round-trip canary** — echo the returned tool-call ID back through the chat template
   with a tool result and assert turn 2 succeeds. Documented failure classes start at round 2,
   not round 1 (sglang#25218).

For long-running serves, re-run the coherence canary **periodically** (mid-campaign sentinel):
upstream issues document coherent-then-garbage degradation appearing only after hours of traffic.

### 2.3 Fatal-signature early abort in the health-wait loop

A dead engine under `srun` can zombie-hold the allocation for hours. While waiting for `/health`,
grep your own job logs (**both** `.out` and `.err` — startup deaths land in `.err`) for fatal
signatures and abort immediately:

```
CUBLAS_STATUS_EXECUTION_FAILED | EngineCore encountered a fatal error |
ValueError: To serve at least one request | CUDA error: | UNSUPPORTED_PTX |
unrecognized arguments | Timed out waiting for engine core |
Engine core initialization failed | DistStoreError | Timed out after .* waiting for clients
```

`unrecognized arguments` matters: vLLM flag sets change between versions, and a typo'd flag
otherwise burns the whole walltime.

---

## 3. Per-model parameterization checklist (never hardcode these)

- **GPU sizing:** read the *actual checkpoint* (file sizes + config precision), don't apply a
  bits-per-param rule of thumb. We watched a doc mis-size a native-INT4 model by assuming FP8.
- **Thinking-by-default models** (Kimi et al.): send `chat_template_kwargs: {"thinking": false}`
  or `content` comes back empty (everything routes to `reasoning_content`).
- **Tool/reasoning parser flags** are per-model (`--tool-call-parser kimi_k2`); some versions lack
  `--reasoning-parser` entirely. **Probe `vllm serve --help` inside the job** rather than assuming.
- **`trust_remote_code` models need a materialized model dir:** the HF cache stores files as
  symlinks into `blobs/`, which breaks relative imports in the model's custom Python. Dereference-
  copy the small files and symlink the weight shards. Most models don't need this — gate it.
- **KV-cache headroom:** at long context, `--gpu-memory-utilization` that is fine for weights can
  starve the KV pool and crash at startup ("free vs needed for one sequence"). Take capture
  headroom from `--max-num-seqs` before lowering utilization.

---

## 4. Agentic workloads against a self-hosted endpoint

Serving a model is the easy half. Driving a real **agent loop** (multi-turn, tool-calling, long
generations) surfaced client-side bugs a text-only gateway never hits. All were found on a real
workload and fixed:

1. **The 300-second wall.** Node's `fetch` (undici) has a default `headersTimeout` of 300 s that
   **cannot be raised** without a custom dispatcher. Long agentic generations exceed it and the
   client throws `HeadersTimeoutError` mid-run. Fix: issue requests over `node:http(s)` with one
   generous socket timeout. Any Node-based agentic client needs this; check your language's HTTP
   stack for equivalent hidden ceilings.
2. **Multi-turn tool-call round-trip.** Some chat templates re-render tool-call IDs over the whole
   history (Kimi: IDs must round-trip verbatim as `functions.{name}:{idx}`). **Test turn 2, not
   just turn 1.** Related: streaming tool-call parsing may be broken upstream even when
   non-streaming works — prefer `stream: false` for tool calls unless proven otherwise.
3. **Strict-API divergences.** Hosted endpoints reject things vLLM tolerates (e.g. one vendor 400s
   on an empty assistant message in history and forbids any `temperature` ≠ 1). If you swap between
   self-hosted and hosted endpoints, normalize requests — don't assume OpenAI semantics.
4. **Unbounded tool results can OOM the client.** A model-supplied zero-width regex made a grep
   tool loop forever (22 GB heap). Cap tool output *inside* the loop that produces it, not after.

**Untested agentic surface (gaps, not features):** MCP against a self-hosted endpoint, truly
concurrent multi-agent traffic, high-concurrency serving, and measured throughput.

---

## 5. How this composes with this repository

The upstream README automates the **single-node** path: stage → sbatch → tunnel → token-protected
gateway. This guide adds what's needed beyond it:

- **Module contract:** the example config's `cuda/12.6` compute module predates the ≥0.22
  wheelhouse rebuild; with `vllm>=0.4.0` unpinned, a fresh install resolves to a ≥0.22 wheel and
  hits §1.1. Pin the pair: wheel vintage ↔ cuda module (and run §1.2 as a preflight).
- **Multi-node:** the job template here is single-node by design; §2.1 is the working multi-node
  shape to template next.
- **Readiness:** the gateway trusts `/health`; §2.2's canaries are what "ready" should mean before
  pointing an agent (or Claude Code) at the endpoint.

## Evidence trail

- Alliance support ticket **#0317340** (wheel↔cuda-module contract; staff-confirmed).
- 19-attempt serve ledger with per-job outcomes and a ruled-out table (internal doc; job IDs cited
  throughout this guide are from it, e.g. 15479377 A/B, 15835049 first all-canaries-green serve).
- vLLM K2 recipe, sglang#25218, vllm#41182 for the tool-calling caveats.
