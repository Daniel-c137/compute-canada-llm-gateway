#!/bin/bash
# Patch vLLM's multi-node data-parallel startup timeouts inside a venv.
#
# Why: the DP startup handshake/coordinator timeouts are hardcoded and too tight for
# first-launch cold-Lustre `spawn` re-imports on Alliance clusters — ranks get killed while
# still importing. We lost ~4 jobs to this before patching (handshake 5 -> 30 min,
# coordinator ZMQ wait -> 900 s).
#
# RE-APPLY ON EVERY ENV REBUILD — pip install wipes the patch.
# Version-coupled: symbol names verified on vLLM 0.22-0.24 wheelhouse builds; later versions
# may move them, in which case this script fails LOUDLY rather than silently doing nothing.
#
# Even patched, DP rendezvous stays flaky (a rank can fail to join: "2/3 clients joined",
# with zero kernel errors). A plain re-run usually clears it — budget for that.
#
# Usage: patch-vllm-dp-timeouts.sh <venv-path> [handshake-mins=30] [coordinator-secs=900]
set -euo pipefail

VENV=${1:?usage: $0 <venv-path> [handshake-mins] [coordinator-secs]}
MINS=${2:-30}
SECS=${3:-900}

CORE=$(ls "$VENV"/lib/python*/site-packages/vllm/v1/engine/core.py 2>/dev/null | head -1)
COORD=$(ls "$VENV"/lib/python*/site-packages/vllm/v1/engine/coordinator.py 2>/dev/null | head -1)
[ -n "$CORE" ] || { echo "FAIL: vllm/v1/engine/core.py not found under $VENV"; exit 1; }

# 1) Engine-core handshake: HANDSHAKE_TIMEOUT_MINS = <n>
if grep -qE "HANDSHAKE_TIMEOUT_MINS\s*=\s*[0-9]+" "$CORE"; then
  cp -n "$CORE" "$CORE.orig"
  sed -i -E "s/HANDSHAKE_TIMEOUT_MINS\s*=\s*[0-9]+/HANDSHAKE_TIMEOUT_MINS = $MINS/" "$CORE"
  echo "patched: $(grep -m1 -E 'HANDSHAKE_TIMEOUT_MINS' "$CORE")  ($CORE)"
else
  echo "FAIL: HANDSHAKE_TIMEOUT_MINS not found in $CORE — vLLM moved it; patch manually:"
  grep -nE "TIMEOUT|timeout" "$CORE" | head -10
  exit 1
fi

# 2) DP coordinator wait: the ZMQ wait the coordinator gives clients to connect. The constant
# has moved between versions; find-and-report, patch what matches, fail loud otherwise.
if [ -n "$COORD" ]; then
  cp -n "$COORD" "$COORD.orig"
  # Known shapes: a *_TIMEOUT constant in seconds/ms, or a poll(<ms>) literal.
  if grep -qE "^[A-Z_]*TIMEOUT[A-Z_]*\s*=\s*[0-9]+" "$COORD"; then
    before=$(grep -m1 -E "^[A-Z_]*TIMEOUT[A-Z_]*\s*=\s*[0-9]+" "$COORD")
    sed -i -E "0,/^([A-Z_]*TIMEOUT[A-Z_]*)\s*=\s*[0-9]+/s//\1 = $SECS/" "$COORD"
    echo "patched coordinator: '$before' -> $(grep -m1 -E '^[A-Z_]*TIMEOUT[A-Z_]*' "$COORD")  ($COORD)"
    echo "  ^ VERIFY the unit (s vs ms) matches what the code expects at that site."
  else
    echo "WARN: no obvious timeout constant in $COORD — inspect these candidates and patch by hand:"
    grep -nE "timeout|poll\(" "$COORD" | head -10
  fi
else
  echo "WARN: coordinator.py not found (single-node build or layout change) — skipping."
fi

echo "done. Remember: re-run this after ANY pip install into $VENV."
