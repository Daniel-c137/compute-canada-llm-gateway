#!/bin/bash
# Build a "materialized" model dir for vLLM `trust_remote_code` models.
#
# Why: the HF cache stores every snapshot file as a symlink into blobs/. Models whose custom
# modeling code does relative imports (`from .some_module import ...`) break when transformers
# follows the main module's symlink into blobs/ and then resolves sibling imports there — the
# siblings exist only under SHA names:
#   FileNotFoundError: .../blobs/<module>.py   (transformers/dynamic_module_utils.py)
#
# Fix: dereference-copy the small files (.py / config / tokenizer / index) so they are REAL
# siblings in one dir, and symlink the big *.safetensors shards back to the blobs (no
# multi-hundred-GB copy; a few tens of KB on disk). Point vLLM --model at DST.
# Idempotent: re-running rebuilds DST from scratch.
#
# Most models do NOT need this — only trust_remote_code models with relative imports.
#
# Usage:
#   materialize-model.sh <org/model | /path/to/snapshot-dir> <dest-dir>
# Examples:
#   HF_HOME=/scratch/$USER/llm-serve/hf materialize-model.sh moonshotai/Kimi-K2.6 /scratch/$USER/llm-serve/kimi-model
#   materialize-model.sh /path/to/hub/models--org--model/snapshots/abc123 /scratch/$USER/llm-serve/model
set -euo pipefail

[ $# -eq 2 ] || { echo "usage: $0 <org/model | snapshot-dir> <dest-dir>"; exit 2; }
SPEC=$1
DST=$2

if [ -d "$SPEC" ]; then
  SRC=$SPEC
else
  HF_HOME=${HF_HOME:?set HF_HOME or pass a snapshot dir}
  CACHE_DIR="$HF_HOME/hub/models--${SPEC//\//--}/snapshots"
  [ -d "$CACHE_DIR" ] || { echo "no cached snapshot at $CACHE_DIR — download weights first (login node)"; exit 1; }
  SRC=$(ls -d "$CACHE_DIR"/*/ | head -1)
fi

echo "SRC=$SRC"
echo "DST=$DST"
rm -rf "$DST"; mkdir -p "$DST"

nshard=0; ncopy=0
for f in "$SRC"/*; do
  base=$(basename "$f")
  case "$base" in
    *.safetensors) ln -s "$(readlink -f "$f")" "$DST/$base"; nshard=$((nshard+1)) ;;
    *) if [ -f "$f" ]; then cp -L "$f" "$DST/$base"; ncopy=$((ncopy+1)); fi ;;
  esac
done
echo "materialized: $ncopy real files, $nshard weight symlinks, $(du -sh "$DST" | cut -f1) on disk"

# Verify every relative import in the custom .py resolves to a real sibling (fail loud if not).
if ls "$DST"/*.py >/dev/null 2>&1; then
  miss=0
  for m in $(grep -rhoE "from[[:space:]]+\.[A-Za-z_][A-Za-z0-9_]*" "$DST"/*.py | sed -E 's/from[[:space:]]+\.//' | sort -u); do
    [ -f "$DST/$m.py" ] || { echo "MISSING sibling: $m.py"; miss=$((miss+1)); }
  done
  [ "$miss" -eq 0 ] && echo "all relative imports resolve — OK" || { echo "ABORT: $miss missing siblings"; exit 1; }
else
  echo "no custom .py in snapshot — this model likely didn't need materializing"
fi
