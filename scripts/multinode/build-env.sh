#!/bin/bash
# Build a vLLM venv from the Alliance wheelhouse on a LOGIN node, verify the tool/reasoning
# parsers a profile needs, and apply the multi-node DP-timeout patches.
#
# Usage:
#   source site.env   # or export MODULES/VENV yourself
#   build-env.sh [vllm-spec] [parser-name]
#     vllm-spec    pip requirement for the wheelhouse resolve (default: "vllm" = newest).
#                  Pin it if you need a specific vintage, e.g. "vllm==0.24.0".
#     parser-name  substring to look for in vLLM's tool/reasoning parser registries
#                  (e.g. "kimi"); skipped if empty.
#
# THE CUDA PIN IS LOAD-BEARING: $MODULES must load the cuda module matching the wheel vintage
# (cuda/13 for vllm>=0.22 wheelhouse wheels; see docs/ALLIANCE_CLUSTER_GUIDE.md 1.1). After
# building, run scripts/diagnostics/ptx_jit_repro.py on a 1-GPU allocation against the venv's
# kernel .so files BEFORE burning a multi-node walltime on it.
set -uo pipefail

VLLM_SPEC=${1:-vllm}
PARSER=${2:-}
: "${MODULES:?source site.env first (MODULES unset)}"
: "${VENV:?source site.env first (VENV unset)}"

# shellcheck disable=SC2086
module load $MODULES
if [ ! -f "$VENV/bin/activate" ]; then
  echo "[build-env] creating venv at $VENV"
  python -m venv "$VENV" || exit 1
fi
source "$VENV/bin/activate"
echo "[build-env] start $(date) | $(python --version) | $(pip --version)"

# --no-index => wheelhouse only. (To escape to pure PyPI instead, create the venv with
# PYTHONPATH="" PIP_CONFIG_FILE="" — see guide section 1.6 — and drop --no-index.)
pip install --no-index --upgrade pip
pip install --no-index "$VLLM_SPEC"
rc=$?
echo "[build-env] pip rc=$rc"
[ $rc -eq 0 ] || exit $rc

echo "[build-env] === versions ==="
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"

if [ -n "$PARSER" ]; then
  echo "[build-env] === '$PARSER' parser availability (go/no-go for native tool calls) ==="
  PARSER="$PARSER" python - <<'PY'
import os
needle = os.environ["PARSER"].lower()
def keys(*candidates):
    for modpath, attr in candidates:
        try:
            mod = __import__(modpath, fromlist=['x'])
            mgr = getattr(mod, attr)
            d = getattr(mgr, 'reasoning_parsers', None) or getattr(mgr, 'tool_parsers', None) or {}
            return sorted(d.keys())
        except Exception:
            continue  # module paths move between vLLM versions — try the next candidate
    return []
# Both historical module locations checked (they moved between releases; checking only the
# old path once false-negatived an available parser).
tp = keys(('vllm.entrypoints.openai.tool_parsers', 'ToolParserManager'),
          ('vllm.tool_parsers', 'ToolParserManager'))
rp = keys(('vllm.reasoning', 'ReasoningParserManager'))
print('tool parsers:', tp or '(none found — registry moved? probe `vllm serve --help` in-job)')
print('reasoning parsers:', rp or '(none found)')
ok_t = any(needle in k.lower() for k in tp)
ok_r = any(needle in k.lower() for k in rp)
print(f"'{needle}' tool parser: {'OK' if ok_t else 'MISSING'} | reasoning parser: {'OK' if ok_r else 'MISSING'}")
raise SystemExit(0 if ok_t else 3)
PY
  [ $? -eq 0 ] || { echo "[build-env] FAIL: requested parser not in this vLLM build"; exit 3; }
fi

echo "[build-env] === DP-timeout patches (multi-node; re-applied on every rebuild) ==="
"$(dirname "$0")/patch-vllm-dp-timeouts.sh" "$VENV" || exit 4

echo "[build-env] done $(date)"
echo "[build-env] NEXT: 1-GPU PTX preflight before any multi-node submit:"
echo "  salloc --gpus=1 --time=00:15:00 ... && module load $MODULES &&"
echo "  python3 scripts/diagnostics/ptx_jit_repro.py $VENV/lib/python*/site-packages/vllm/_moe_C*.so"
