#!/bin/bash
#SBATCH --job-name=avengers-physics-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

# Usage:
#   sbatch scripts/evaluate_physics_models_slurm_template.sh
#
# Common overrides:
#   CONFIG_PATH=config/arjun_config.yaml \
#   SAVE_DIR=results/physics_eval \
#   MODE=full \
#   MAX_WORKERS=1 \
#   sbatch --array=0-2 scripts/evaluate_physics_models_slurm_template.sh
#
# The Slurm array index selects one expert from CONFIG_PATH's experts list.
# If you add/remove experts, override --array accordingly, e.g. --array=0-4.

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs config/temp results

# Uncomment and customize one of these blocks if your cluster uses modules or conda.
# module purge
# module load cuda/12.1
# module load python/3.12

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate avengers

if [[ -x ".venv/bin/python" ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export USE_HUB_KERNELS="${USE_HUB_KERNELS:-NO}"

CONFIG_PATH="${CONFIG_PATH:-config/arjun_config.yaml}"
SAVE_DIR="${SAVE_DIR:-results/physics_eval}"
MODE="${MODE:-full}"
MAX_WORKERS="${MAX_WORKERS:-1}"
USE_HTTP_CACHE="${USE_HTTP_CACHE:-false}"
CACHE_DIR="${CACHE_DIR:-cache/physics_eval}"
GENERATOR_TYPE="${GENERATOR_TYPE:-direct}"
MODEL_INDEX="${SLURM_ARRAY_TASK_ID:-${MODEL_INDEX:-0}}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-manual}"
REQUIRE_CUDA="${REQUIRE_CUDA:-true}"
FORCE_LOCAL_HF_CACHE="${FORCE_LOCAL_HF_CACHE:-true}"
SKIP_DEP_CHECK="${SKIP_DEP_CHECK:-false}"

HF_CACHE_ROOT="${HF_CACHE_ROOT:-.cache/huggingface}"
if [[ "${FORCE_LOCAL_HF_CACHE}" == "true" ]]; then
  export HF_HOME="${HF_CACHE_ROOT}"
  export HUGGINGFACE_HUB_CACHE="${HF_CACHE_ROOT}/hub"
  export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers"
  export HF_MODULES_CACHE="${HF_CACHE_ROOT}/modules"
  export XDG_CACHE_HOME=".cache"
else
  export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"
  export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_CACHE_ROOT}/modules}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-.cache}"
fi
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_MODULES_CACHE}" "${XDG_CACHE_HOME}" "${CACHE_DIR}"

TEMP_CONFIG="config/temp/physics_eval_${JOB_ID}_${MODEL_INDEX}.yaml"

echo "Starting PHYSICS evaluation at $(date)"
echo "Running from: $(pwd)"
echo "Base config: ${CONFIG_PATH}"
echo "Model index: ${MODEL_INDEX}"
echo "Save dir: ${SAVE_DIR}"
echo "HF_HOME: ${HF_HOME}"

if [[ "${REQUIRE_CUDA}" == "true" ]]; then
  "${PYTHON}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available to PyTorch on this node. "
        "The local PHYSICS eval models are too large to run on CPU; resubmit on a node "
        "with a compatible NVIDIA driver, or set REQUIRE_CUDA=false to bypass this check."
    )

print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
PY
fi

"${PYTHON}" - "${CONFIG_PATH}" "${TEMP_CONFIG}" "${MODEL_INDEX}" "${MODE}" "${MAX_WORKERS}" "${USE_HTTP_CACHE}" "${CACHE_DIR}" "${GENERATOR_TYPE}" <<'PY'
import copy
import re
import sys
from pathlib import Path

import yaml

config_path, temp_config, model_index, mode, max_workers, use_http_cache, cache_dir, generator_type = sys.argv[1:]
model_index = int(model_index)
max_workers = int(max_workers)
use_http_cache = use_http_cache.lower() in {"1", "true", "yes", "y"}

with open(config_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

experts = config.get("experts", [])
if not experts:
    raise SystemExit(f"No experts found in {config_path}")
if model_index < 0 or model_index >= len(experts):
    raise SystemExit(
        f"MODEL_INDEX {model_index} is out of range for {len(experts)} experts. "
        f"Use --array=0-{len(experts) - 1}."
    )

selected_expert = copy.deepcopy(experts[model_index])
model_name = selected_expert["name"]
safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")[:120] or f"model_{model_index}"

config["experiment_name"] = f"physics-{model_index}-{safe_name}"
config["experiments"]["task"] = "physics"
config["experiments"]["mode"] = mode
config["experiments"]["max_workers"] = max_workers
config["experiments"]["use_http_cache"] = use_http_cache
config["experiments"]["cache_dir"] = cache_dir
config["router"]["type"] = "straight"
config["router"]["straight_router"]["model"] = model_name
config["generator"]["type"] = generator_type
config["experts"] = [selected_expert]

Path(temp_config).parent.mkdir(parents=True, exist_ok=True)
with open(temp_config, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

print(f"Selected model: {model_name}")
print(f"Wrote temp config: {temp_config}")
PY

if [[ "${SKIP_DEP_CHECK}" != "true" ]]; then
  "${PYTHON}" - "${TEMP_CONFIG}" <<'PY'
import importlib.util
import sys

import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

expert = config["experts"][0]
name = expert["name"].lower()
model_path = expert.get("model_path", "").lower()
required_modules = []

if "nemotron" in name or "nemotron" in model_path:
    required_modules.append(("mamba_ssm", "mamba-ssm"))
if "deepseek" in name or "deepseek" in model_path:
    required_modules.append(("kernels", "kernels"))

missing = [package for module, package in required_modules if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit(
        "Missing required Python package(s) for this model: "
        + ", ".join(missing)
        + ". Install them in .venv before resubmitting, or set SKIP_DEP_CHECK=true to bypass this preflight."
    )

if required_modules:
    print("Model dependency preflight OK.")
PY
fi

cmd=(
  "${PYTHON}" app.py
  --config "${TEMP_CONFIG}"
  --save_dir "${SAVE_DIR}"
)

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

echo "Finished PHYSICS evaluation at $(date)"
