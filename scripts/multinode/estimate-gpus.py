#!/usr/bin/env python3
"""Estimate GPU count / topology for serving a checkpoint — from the ACTUAL checkpoint.

Sizing from a bits-per-param rule of thumb is how deployments get mis-planned (we watched a
doc assume FP8 for a model that ships native INT4 — a 2x error). This reads what is actually
on disk:

  * weight bytes: model.safetensors.index.json metadata.total_size, falling back to summing
    *.safetensors file sizes (symlinks resolved — works on materialized dirs);
  * precision/quant: config.json torch_dtype + quantization_config (native quant beats dtype);
  * MoE detection: expert-count keys in config.json -> suggest --enable-expert-parallel.

The estimate covers WEIGHTS + a fixed overhead fraction. KV cache is context- and
architecture-dependent (MLA vs GQA differ hugely) and is NOT modeled — the printed headroom
number is what's left for KV/activations; judge it against your --max-model-len (at 32K ctx
one sequence cost ~2.1 GiB on a DeepSeek-V3-style MLA model; job 15080024 died with 0.94 GiB
free). Treat the output as a starting point, not a guarantee; the first serve validates it.

Usage:
  estimate-gpus.py --model-dir /path/to/model \\
      [--gpu-vram-gb 80] [--gpus-per-node 4] [--gpu-mem-util 0.90] [--overhead 0.10]
"""
import argparse
import glob
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--gpu-vram-gb", type=float, default=80.0)
ap.add_argument("--gpus-per-node", type=int, default=4)
ap.add_argument("--gpu-mem-util", type=float, default=0.90,
                help="planned --gpu-memory-utilization")
ap.add_argument("--overhead", type=float, default=0.15,
                help="fraction added to weights for runtime overhead + a minimum KV floor. "
                     "0.15 reproduces the proven Kimi sizing (9 GPUs -> 3 nodes); 0.10 would "
                     "have suggested the 2-node config that sat below the KV cliff at 32K "
                     "context (job 15080024). Increase for long contexts / high concurrency.")
args = ap.parse_args()
d = args.model_dir

# --- weight bytes ----------------------------------------------------------------------------
total = None
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.isfile(idx):
    try:
        total = json.load(open(idx)).get("metadata", {}).get("total_size")
    except Exception:
        pass
if not total:
    shards = glob.glob(os.path.join(d, "*.safetensors"))
    if not shards:
        sys.exit(f"no safetensors index or shards under {d}")
    total = sum(os.stat(os.path.realpath(p)).st_size for p in shards)
weights_gb = total / 1e9

# --- config ----------------------------------------------------------------------------------
cfg = {}
cfg_path = os.path.join(d, "config.json")
if os.path.isfile(cfg_path):
    cfg = json.load(open(cfg_path))
dtype = cfg.get("torch_dtype", "?")
quant = (cfg.get("quantization_config") or {}).get("quant_method")
n_experts = next((cfg[k] for k in
                  ("num_experts", "n_routed_experts", "num_local_experts") if k in cfg), None)
def find_bits(obj):
    """Recursively find a declared weight bit-width in quantization_config
    (compressed-tensors: config_groups.*.weights.num_bits; GPTQ/AWQ: bits)."""
    if isinstance(obj, dict):
        for k in ("num_bits", "bits"):
            if isinstance(obj.get(k), int):
                return obj[k]
        for v in obj.values():
            b = find_bits(v)
            if b:
                return b
    if isinstance(obj, list):
        for v in obj:
            b = find_bits(v)
            if b:
                return b
    return None


# rough param count back-out, only when the byte-width is actually declared
params_b = None
bpp = None
if quant:
    bits = find_bits(cfg.get("quantization_config"))
    bpp = bits / 8 if bits else None   # undeclared quant width -> don't guess
elif dtype in ("bfloat16", "float16"):
    bpp = 2.0
elif dtype == "float32":
    bpp = 4.0
if bpp:
    params_b = total / bpp / 1e9

print(f"checkpoint : {d}")
print(f"weights    : {weights_gb:,.1f} GB on disk"
      + (f"  (~{params_b:,.0f}B params @ {bpp} B/param)" if params_b else ""))
print(f"precision  : torch_dtype={dtype}"
      + (f", quantization_config.quant_method={quant}  <- serving size is THIS, not dtype"
         if quant else " (no quantization_config: dense/native dtype)"))
if n_experts:
    print(f"MoE        : {n_experts} experts -> use --enable-expert-parallel with DP")

# --- arithmetic ------------------------------------------------------------------------------
usable_per_gpu = args.gpu_vram_gb * args.gpu_mem_util
need = weights_gb * (1 + args.overhead)
min_gpus = math.ceil(need / usable_per_gpu)
nodes = math.ceil(min_gpus / args.gpus_per_node)
whole_gpus = nodes * args.gpus_per_node
headroom = whole_gpus * usable_per_gpu - weights_gb

print()
print(f"usable/GPU : {usable_per_gpu:.1f} GB  ({args.gpu_vram_gb:.0f} GB x util {args.gpu_mem_util})")
print(f"need       : {need:,.1f} GB  (weights x {1 + args.overhead:.2f} overhead)")
print(f"min GPUs   : {min_gpus}")
if min_gpus <= args.gpus_per_node:
    print(f"topology   : SINGLE NODE, --tensor-parallel-size {min_gpus} "
          f"(round up to a divisor of attention heads if needed; no DP machinery)")
else:
    print(f"topology   : {nodes} nodes x {args.gpus_per_node} GPUs = {whole_gpus} GPUs -> "
          f"TP={args.gpus_per_node} x DP={nodes}"
          + (" + --enable-expert-parallel" if n_experts else ""))
print(f"KV headroom: {headroom:,.1f} GB total after weights — NOT a KV model; check against")
print(f"             your --max-model-len before trusting it (see docstring).")
print()
print("Reference point: Kimi-K2.6 523 GB native-INT4 MoE -> min 9 GPUs -> 3 nodes x 4xH100")
print("(TP4 x DP3 + EP), which is exactly the serve-proven config.")
