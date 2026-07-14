#!/bin/bash
# Stage a model's weights from its profile-declared source to the profile's MODEL_PATH.
# Run on a LOGIN node (compute nodes have no internet).
#
# The profile declares WHERE the model comes from (HF_REPO) and HOW it must be laid out
# (NEEDS_MATERIALIZED_DIR); this script makes MODEL_PATH real:
#   1. download HF_REPO into $HF_HOME/hub (skipped/resumed if already cached);
#   2. NEEDS_MATERIALIZED_DIR=1 -> materialize-model.sh (trust_remote_code symlink fix);
#      otherwise MODEL_PATH becomes a symlink to the resolved snapshot dir.
#
# Gated models: export HF_TOKEN first (the license must be accepted on the HF website).
#
# Usage: stage-model.sh <profile.env> [site.env]
set -euo pipefail

PROFILE=${1:?usage: $0 <profile.env> [site.env]}
SITE_ENV=${2:-$(dirname "$PROFILE")/../site.env}
# shellcheck disable=SC1090
source "$SITE_ENV"
# shellcheck disable=SC1090
source "$PROFILE"
: "${HF_REPO:?profile must declare HF_REPO (the weight source)}"
: "${MODEL_PATH:?profile must declare MODEL_PATH (where the serve reads weights)}"
: "${HF_HOME:?site.env must set HF_HOME}"

echo "[stage] $HF_REPO -> $MODEL_PATH (HF_HOME=$HF_HOME)"

# Recent huggingface_hub renamed the CLI: `hf` replaced `huggingface-cli`.
if command -v hf >/dev/null 2>&1; then
  hf download "$HF_REPO" --cache-dir "$HF_HOME/hub"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$HF_REPO" --cache-dir "$HF_HOME/hub"
else
  echo "FAIL: no hf/huggingface-cli on PATH — pip install 'huggingface_hub[cli]' (login node)"; exit 1
fi

SNAP_PARENT="$HF_HOME/hub/models--${HF_REPO//\//--}/snapshots"
[ -d "$SNAP_PARENT" ] || { echo "FAIL: no snapshot at $SNAP_PARENT after download"; exit 1; }
SNAP=$(ls -d "$SNAP_PARENT"/*/ | head -1)
echo "[stage] snapshot: $SNAP"

if [ "${NEEDS_MATERIALIZED_DIR:-0}" = 1 ]; then
  "$(dirname "$0")/materialize-model.sh" "$SNAP" "$MODEL_PATH"
else
  mkdir -p "$(dirname "$MODEL_PATH")"
  ln -sfn "${SNAP%/}" "$MODEL_PATH"
  echo "[stage] linked $MODEL_PATH -> ${SNAP%/}"
fi

echo "[stage] done. Sizing check:"
echo "  python3 $(dirname "$0")/estimate-gpus.py --model-dir $MODEL_PATH"
