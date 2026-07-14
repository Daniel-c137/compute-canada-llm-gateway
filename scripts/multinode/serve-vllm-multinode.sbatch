#!/bin/bash
# Multi-node vLLM serve on Alliance/SLURM: TP-inside-node x DP-across-nodes (+EP for MoE),
# hardened health-wait, canary battery, SERVE_READY handoff, hold.
#
# Generalized from the config that first passed coherence + native tool-call + turn-2
# roundtrip for Kimi-K2.6 on Rorqual (job 15835049). Model specifics come from a PROFILE
# file (see profiles/kimi-k2.6.env); site specifics from site.env.
#
# Submit (account/geometry/time on the CLI — they're site decisions, not script defaults):
#   sbatch --account=<acct> --nodes=3 --gpus-per-node=h100:4 --cpus-per-task=16 \
#          --time=02:00:00 --time-min=01:00:00 \
#          serve-vllm-multinode.sbatch profiles/kimi-k2.6.env [site.env]
#
# Notes:
#   * --time-min lets SLURM backfill you into a hole (cut one wait from days to hours).
#   * Staff kill idle-GPU jobs: size walltime to real use. For a go/no-go verdict, a short
#     "verdict-shaped" walltime that expires right after the canaries is deliberate & polite.
#   * TP is set to GPUs-per-node automatically; DP = number of nodes.
#   * cpus-per-task: vLLM serving is GPU-bound; whole-node GPU requests already own the node,
#     but on shared-GPU partitions request modest CPUs (the 192-core/0.85-used waste flag on
#     our own jobstats is what staff's waste detector sees).
#SBATCH --job-name=vllm-multinode
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -u

PROFILE=${1:?usage: sbatch ... serve-vllm-multinode.sbatch <profile.env> [site.env]}
SITE_ENV=${2:-$(dirname "$PROFILE")/../site.env}
# shellcheck disable=SC1090
source "$SITE_ENV"
# shellcheck disable=SC1090
source "$PROFILE"

: "${MODULES:?}"; : "${VENV:?}"; : "${WORKDIR:?}"; : "${API_PORT:=8000}"; : "${DP_RPC_PORT:=13345}"
mkdir -p "$WORKDIR" "${LOG_DIR:-$WORKDIR/logs}"
DIAG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../diagnostics" && pwd)"

GPUS_PER_NODE=$(echo "${SLURM_GPUS_PER_NODE:-1}" | grep -oE '[0-9]+$')
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')
export HEAD_IP API_PORT DP_RPC_PORT VENV MODULES MODEL_PATH GPUS_PER_NODE
echo "Head: $HEAD_NODE ($HEAD_IP) | API http://$HEAD_IP:$API_PORT/v1 | model: $MODEL_PATH"
echo "Topology: TP=$GPUS_PER_NODE x DP=$SLURM_NNODES$( [ "${ENABLE_EXPERT_PARALLEL:-0}" = 1 ] && echo ' +EP')"

# Assemble the vLLM arg list from the profile (model-agnostic; the profile owns the choices).
VLLM_ARGS=( "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$GPUS_PER_NODE"
  --data-parallel-size "$SLURM_NNODES" --data-parallel-size-local 1
  --data-parallel-address "$HEAD_IP" --data-parallel-rpc-port "$DP_RPC_PORT"
  --max-model-len "${MAX_MODEL_LEN:?}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:?}"
  --max-num-seqs "${MAX_NUM_SEQS:?}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:?}"
  --distributed-timeout-seconds 3600 )
[ "${TRUST_REMOTE_CODE:-0}" = 1 ]        && VLLM_ARGS+=( --trust-remote-code )
[ "${ENABLE_EXPERT_PARALLEL:-0}" = 1 ]   && VLLM_ARGS+=( --enable-expert-parallel )
[ "${ENFORCE_EAGER:-0}" = 1 ]            && VLLM_ARGS+=( --enforce-eager )
[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" = 1 ] && VLLM_ARGS+=( --disable-custom-all-reduce )
[ -n "${TOOL_CALL_PARSER:-}" ]           && VLLM_ARGS+=( --tool-call-parser "$TOOL_CALL_PARSER" )
[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" = 1 ]  && VLLM_ARGS+=( --enable-auto-tool-choice )
# --reasoning-parser doesn't exist in every vLLM version: probe before adding.
if [ -n "${REASONING_PARSER:-}" ]; then
  # shellcheck disable=SC2086
  module load $MODULES; source "$VENV/bin/activate"
  if vllm serve --help 2>/dev/null | grep -q reasoning-parser; then
    VLLM_ARGS+=( --reasoning-parser "$REASONING_PARSER" )
  else
    echo "NOTE: this vLLM has no --reasoning-parser; continuing without it."
  fi
fi
# shellcheck disable=SC2206
[ -n "${EXTRA_VLLM_ARGS:-}" ] && VLLM_ARGS+=( ${EXTRA_VLLM_ARGS} )
printf 'vLLM args:'; printf ' %q' "${VLLM_ARGS[@]}"; echo
export VLLM_ARGS_STR=$(printf ' %q' "${VLLM_ARGS[@]}")

serve_node() {
  set -e
  # shellcheck disable=SC2086
  module load $MODULES
  source "$VENV/bin/activate"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   # no internet on compute
  export HF_MODULES_CACHE="$HOME/.cache/huggingface/modules"; mkdir -p "$HF_MODULES_CACHE"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_IB_DISABLE=0 VLLM_NO_USAGE_STATS=1
  export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
  # Node-local compile/JIT caches — Lustre defaults cause cross-node filelock hangs (guide 1.4).
  export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton"; mkdir -p "$TRITON_CACHE_DIR"
  export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/xdgcache"; mkdir -p "$XDG_CACHE_HOME"
  export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"; mkdir -p "$FLASHINFER_WORKSPACE_BASE"
  export CUDA_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/nvcache"; mkdir -p "$CUDA_CACHE_PATH"
  export CUDA_MODULE_LOADING=EAGER   # driver-JIT failures fire loudly at startup, not mid-traffic

  eval "ARGS=( $VLLM_ARGS_STR )"
  if [ "$SLURM_NODEID" -eq 0 ]; then
    echo "[node $SLURM_NODEID @ $(hostname)] vLLM API server (DP rank 0)"
    exec vllm serve "${ARGS[@]}" --host 0.0.0.0 --port "$API_PORT"
  else
    echo "[node $SLURM_NODEID @ $(hostname)] vLLM headless worker (DP rank $SLURM_NODEID)"
    exec vllm serve "${ARGS[@]}" --headless --data-parallel-start-rank "$SLURM_NODEID"
  fi
}
export -f serve_node

srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c 'serve_node' &
SRUN_PID=$!

# Health-wait with fatal-signature early abort: a dead engine under srun otherwise
# zombie-holds the allocation for the whole walltime. Startup deaths land in .err, not .out.
FATAL='CUBLAS_STATUS_EXECUTION_FAILED|EngineCore encountered a fatal error|ValueError: To serve at least one request|CUDA error:|UNSUPPORTED_PTX|unsupported toolchain|unrecognized arguments|TimeoutError: Timed out waiting for engine core|Engine core initialization failed|DistStoreError|Timed out after .* waiting for clients'
SELF_LOG="${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"; SELF_ERR="${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
WAIT=0; TIMEOUT=${VLLM_READY_MAX_WAIT_SEC:-3600}
until curl -fs "http://$HEAD_IP:$API_PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$SRUN_PID" 2>/dev/null; then echo "ERROR: vLLM exited before healthy — see .err"; exit 1; fi
  if [ "$WAIT" -ge "$TIMEOUT" ]; then echo "ERROR: not healthy after ${TIMEOUT}s"; kill "$SRUN_PID"; exit 1; fi
  if [ $((WAIT % 60)) -eq 0 ] && [ "$WAIT" -ge 60 ] && \
     grep -aqE "$FATAL" "$SELF_LOG" "$SELF_ERR" 2>/dev/null; then
    echo "ERROR: fatal engine signature at ${WAIT}s — aborting early:"
    grep -aE "$FATAL" "$SELF_LOG" "$SELF_ERR" 2>/dev/null | tail -3
    kill "$SRUN_PID" 2>/dev/null; exit 1
  fi
  sleep 10; WAIT=$((WAIT + 10))
  [ $((WAIT % 120)) -eq 0 ] && echo "  ...waiting for vLLM (${WAIT}s)"
done
echo "vLLM ready after ${WAIT}s."

# Canary battery BEFORE declaring the endpoint usable (a green /health is not health).
# shellcheck disable=SC2086
module load $MODULES; source "$VENV/bin/activate"
CANARY_FLAGS=( --base-url "http://$HEAD_IP:$API_PORT/v1" --model "$SERVED_MODEL_NAME"
               --warm-ranks $((SLURM_NNODES * 2)) )
[ "${THINKING_KILL_SWITCH:-0}" = 1 ] && CANARY_FLAGS+=( --thinking-kill-switch )
[ "${CANARY_TOOLS:-1}" = 1 ] || CANARY_FLAGS+=( --no-tools )
if python3 "$DIAG_DIR/canary-battery.py" "${CANARY_FLAGS[@]}"; then
  CANARY_RC=0
else
  CANARY_RC=$?
  echo "WARNING: canary battery FAILED (rc=$CANARY_RC) — endpoint is up but NOT trustworthy."
fi

# SERVE_READY handoff: consumers (gateway tunnel, batch drivers) read endpoint facts here.
{ echo "HEAD_NODE=$HEAD_NODE"; echo "HEAD_IP=$HEAD_IP"; echo "API_PORT=$API_PORT";
  echo "JOBID=$SLURM_JOB_ID"; echo "BASE=http://$HEAD_IP:$API_PORT/v1";
  echo "MODEL=$SERVED_MODEL_NAME"; echo "CANARY_RC=$CANARY_RC"; } > "$WORKDIR/SERVE_READY"
echo "=== SERVE_READY written ==="; cat "$WORKDIR/SERVE_READY"

# Hold for the remaining walltime. Keep this tail SHORT by sizing --time to real use —
# staff kill jobs that idle GPUs. Re-run the canary periodically as a coherence sentinel.
HELD=0; SENTINEL_EVERY=${COHERENCE_SENTINEL_SEC:-3600}
while kill -0 "$SRUN_PID" 2>/dev/null; do
  curl -fs "http://$HEAD_IP:$API_PORT/health" >/dev/null 2>&1 || echo "[hold] WARN: health fail at ${HELD}s"
  sleep 60; HELD=$((HELD + 60))
  [ $((HELD % 600)) -eq 0 ] && echo "[hold] endpoint up ${HELD}s"
  if [ $((HELD % SENTINEL_EVERY)) -eq 0 ]; then
    echo "[sentinel] periodic coherence check at ${HELD}s"
    python3 "$DIAG_DIR/canary-battery.py" "${CANARY_FLAGS[@]}" --no-tools \
      || echo "[sentinel] WARNING: coherence degraded at ${HELD}s — restart the serve"
  fi
done
echo "[hold] srun exited after ${HELD}s; cleaning up."; rm -f "$WORKDIR/SERVE_READY"
kill "$SRUN_PID" 2>/dev/null; wait 2>/dev/null; echo "Job done."
