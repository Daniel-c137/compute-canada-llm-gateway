# Runbook: Kimi-K2.6 with native tool calling on Alliance clusters (the reproducible path)

End-to-end reproduction of the proven deployment — **Kimi-K2.6 (523 GB, native INT4 MoE) on
3 nodes × 4×H100 (Rorqual), vLLM 0.24 wheelhouse + `cuda/13`, coherence + native tool-call +
turn-2 roundtrip all green** (job 15835049) — using the generalized scripts in
`scripts/multinode/`. Swap the profile to target another model; everything model-specific is
in the profile, everything cluster-specific in `site.env`.

Read [docs/ALLIANCE_CLUSTER_GUIDE.md](ALLIANCE_CLUSTER_GUIDE.md) first — especially §0
(what's proven vs. not) and §1.1 (the cuda-module contract that costs weeks if missed).

## 0. Prerequisites

- Alliance account with a GPU allocation that permits multi-node jobs; SSH (+ Duo) to the
  login node. Multi-node queues run **days**; plan around `--time-min` backfill (§1.7).
- ~530 GB of `/scratch` (or `/project`) for weights. ⚠️ **Scratch is purged periodically** —
  keep a re-download path or a `/project` copy for anything you can't recreate.
- Disk quota for a ~2–3 GB venv.

## 1. Configure

```bash
cd scripts/multinode
cp site.example.env site.env    # edit: MODULES (cuda pin!), VENV, WORKDIR, HF_HOME
```

The profile `profiles/kimi-k2.6.env` is the proven reference — only `MODEL_PATH` should need
adjusting (or export it at submit time).

## 2. Stage weights (login node — compute nodes have no internet)

The profile declares the source (`HF_REPO`) and layout (`NEEDS_MATERIALIZED_DIR`);
`stage-model.sh` makes `MODEL_PATH` real from them — download, then materialize (Kimi:
`trust_remote_code` + the symlinked HF cache breaks relative imports) or plain-symlink:

```bash
module load python/3.11 && pip install -U "huggingface_hub[cli]" --user   # once, for `hf`
./stage-model.sh profiles/kimi-k2.6.env site.env
# ends with "all relative imports resolve — OK" (materialized; ~75 KB on disk, weights symlinked)
```

Gated models: accept the license on the HF website and `export HF_TOKEN=...` first.

## 3. Size the resource request from the ACTUAL checkpoint

Never size from a bits-per-param rule of thumb — read what's on disk (a native-INT4 model
assumed FP8 is a 2× planning error):

```bash
python3 ./estimate-gpus.py --model-dir /scratch/$USER/llm-serve/kimi-model
# Kimi-K2.6: 523 GB native INT4, MoE -> min 9 GPUs -> 3 nodes x 4xH100, TP4 x DP3 + EP
```

It reads weight bytes (safetensors index), `quantization_config`, and expert counts, then
prints the arithmetic and a topology suggestion. KV cache is NOT modeled — sanity-check the
printed headroom against your `MAX_MODEL_LEN` (see the script docstring).

## 4. Build the env (login node) and patch DP timeouts

```bash
source site.env
./build-env.sh "vllm==0.24.0" kimi
```

This installs from the wheelhouse (`--no-index`), verifies the `kimi_k2` tool/reasoning
parsers exist in this build, and applies the DP-timeout patches (handshake 30 min,
coordinator wait — **re-run after any `pip install` into the venv**).

## 5. Preflight on ONE GPU before burning a multi-node walltime

```bash
salloc --account=<acct> --gpus=h100:1 --time=00:15:00
module load $MODULES   # from site.env — the exact set you'll serve with
python3 ../diagnostics/ptx_jit_repro.py $VENV/lib/python*/site-packages/vllm/_moe_C*.so
# CUDA_SUCCESS → wheel↔driver↔module contract holds. rc=222 → wrong cuda module (guide §1.1).
```

## 6. Submit

First run: make it **verdict-shaped** — a short walltime whose job is the canary verdicts,
not a long hold (staff kill idle-GPU jobs; a TIMEOUT right after the verdicts is by design):

```bash
sbatch --account=<acct> --nodes=3 --gpus-per-node=h100:4 --cpus-per-task=16 \
       --time=01:30:00 --time-min=01:00:00 \
       serve-vllm-multinode.sbatch profiles/kimi-k2.6.env site.env
```

Expect ~20–30 min to READY (523 GB loads from Lustre). Then check the log for the verdicts:

```
vLLM ready after <N>s.
COHERENCE_VERDICT: COHERENT ...
TOOLCALL_CANARY_PASS
TOOLCALL_ROUNDTRIP_PASS
CANARY_BATTERY_PASS
=== SERVE_READY written ===
```

`$WORKDIR/SERVE_READY` holds `BASE=http://<head-ip>:<port>/v1` + `CANARY_RC` for consumers.
For a long-lived serve, resubmit with a real `--time`; the script re-runs a coherence
sentinel hourly (`COHERENCE_SENTINEL_SEC` to change) because upstream reports document
coherent-then-garbage degradation hours into normal traffic.

## 7. Point a client (or agent) at it

Every request to Kimi needs:

- `chat_template_kwargs: {"thinking": false}` — or `content` comes back empty;
- `stream: false` for tool calls (kimi_k2 streaming parser broken upstream, vllm#41182);
- tool-call IDs echoed **verbatim** (`functions.{name}:{idx}`) in multi-turn history — the
  chat template re-renders them and malformed IDs poison later turns;
- an HTTP client without a hidden header-timeout: Node's `fetch`/undici caps headers at an
  unraisable **300 s** — long generations WILL exceed it; use `node:http(s)` with one
  generous socket timeout (guide §4);
- caps on tool-result size *inside* your agent loop (a model-supplied pathological input to
  a tool once produced a 22 GB heap).

To expose it off-cluster, use this repo's gateway flow (README): SSH local-forward
`-L 18000:<HEAD_NODE>:<API_PORT>` via the login host, then `cc_run_gateway.py` for the
token-protected proxy — including the Claude Code wiring (`ANTHROPIC_BASE_URL` etc.).
Re-run the canary battery **through the gateway** before trusting it:

```bash
python3 scripts/diagnostics/canary-battery.py \
  --base-url http://<vm>:8080/v1 --model moonshotai/Kimi-K2.6 \
  --chat-template-kwargs '{"thinking": false}' --api-key "$GATEWAY_TOKEN"
```

## 8. Troubleshooting (each of these cost us real jobs)

| Symptom | Cause / fix |
|---|---|
| `the provided PTX was compiled with an unsupported toolchain` / rc=222 | cuda module doesn't match wheel vintage → `module load cuda/13` for vLLM ≥ 0.22 (guide §1.1); confirm with `ptx_jit_repro.py` |
| Dies waiting for DP clients (`2/3 clients joined`), no kernel errors | DP-rendezvous flake — **just resubmit**; if recurring, confirm the DP-timeout patches survived the last env change |
| `FileNotFoundError: .../blobs/<file>.py` | trust_remote_code + symlinked HF cache → materialize (step 3) |
| KV ValueError at startup (`free vs needed for one request`) | KV starvation — raise `GPU_MEMORY_UTILIZATION` back to 0.90 and take headroom from `MAX_NUM_SEQS` instead |
| READY + 200 OK but nonsense output | This is why the canary battery exists; if `COHERENCE_VERDICT: GARBAGE`, suspect binary/module mismatch first — re-run the preflight under the job's exact module set |
| Empty `content`, text in `reasoning_content` | Missing `thinking:false` kill-switch on that request |
| Hangs mid-generation at ~5 min from a Node client | undici 300 s headers wall — switch to `node:http(s)` |
| Job zombie-holds after engine death | The health-wait's fatal-signature grep should abort it; if you see a new fatal string, add it to `FATAL` in the sbatch |

## 9. Adapting to another model — the five questions a new profile must answer

Copy `profiles/kimi-k2.6.env` (proven) or `profiles/qwen3-32b.example.env` (dense/single-node
shape, untested) and answer:

1. **Where do the weights come from?** `HF_REPO` (+ `HF_TOKEN` if gated). `stage-model.sh`
   turns it into `MODEL_PATH`; set `NEEDS_MATERIALIZED_DIR=1` only for `trust_remote_code`
   models with relative imports.
2. **How big a request?** `estimate-gpus.py --model-dir ...` on the staged checkpoint —
   precision comes from `quantization_config`, not assumptions. Single-node result → submit
   with `--nodes=1` (TP only, DP machinery idle); multi-node → TP=GPUs-per-node × DP=nodes,
   `ENABLE_EXPERT_PARALLEL=1` if MoE.
3. **MoE or dense?** `ENABLE_EXPERT_PARALLEL=1` for MoE (the estimator detects expert keys
   and says so), `0` for dense. Size by *total* checkpoint bytes, never active params — all
   experts live in HBM. Quantized MoE is the highest-risk combination for silently-garbage
   output (guide §2.4): treat the coherence canary + sentinel as mandatory, and know the
   TP=1 pure-DP+EP recipe shape is the fallback discriminator if output is wrong.
4. **Agentic or not?** Agentic: set `TOOL_CALL_PARSER` (per-model — `kimi_k2`, `hermes`, ...;
   probe `vllm serve --help` in-job), `ENABLE_AUTO_TOOL_CHOICE=1`, `CANARY_TOOLS=1` so the
   battery gates on tool-call + turn-2 roundtrip. Non-agentic: empty `TOOL_CALL_PARSER`,
   `CANARY_TOOLS=0` — no parser flags are passed and the canary skips tool phases; coherence
   still gates.
5. **Thinking-by-default?** Set `CHAT_TEMPLATE_KWARGS` with the *model's* kwarg key
   (`{"thinking": false}` for Kimi, `{"enable_thinking": false}` for Qwen3-class) or leave
   empty. Empty `content` with text in `reasoning_content` is the symptom of getting this
   wrong.

Then: 1-GPU PTX preflight (step 5) → **verdict-shaped first serve** (step 6). Tuning values
(`GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`) carry over as starting
points only.

## 10. Honest status

The scripts are generalized (profile + site split), but only the **Kimi-K2.6 / Rorqual /
vLLM 0.24 / cuda 13.2** combination is serve-proven end-to-end. First runs on other models,
clusters, or vLLM versions are validation events, not routine — expect to adjust the profile
(and please record what you change). Known-untested surface: dense models, other clusters,
throughput under load, concurrent multi-agent traffic, MCP.
