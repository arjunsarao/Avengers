#!/bin/bash
#SBATCH --job-name=avengers-rank-router
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# Usage:
#   sbatch scripts/generate_rank_router_slurm_template.sh
#
# Common overrides:
#   DATA_PATH=data/training_data.json \
#   EMBED_URL=http://your-embedding-api:8000/v1 \
#   EMBED_API_KEY=your-api-key \
#   sbatch scripts/generate_rank_router_slurm_template.sh

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# Uncomment and customize one of these blocks if your cluster uses modules or conda.
# module purge
# module load cuda/12.1
# module load python/3.10

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate avengers

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

DATA_PATH="${DATA_PATH:-data/training_data.json}"
OUTPUT_DIR="${OUTPUT_DIR:-core/rank}"
EMBED_MODEL="${EMBED_MODEL:-gte-qwen2-7b-instruct}"
EMBED_URL="${EMBED_URL:-}"
EMBED_API_KEY="${EMBED_API_KEY:-}"
N_CLUSTERS="${N_CLUSTERS:-64}"
N_MODELS="${N_MODELS:-10}"
TEST_SIZE="${TEST_SIZE:-0.3}"
SEED="${SEED:-42}"
CACHE_DIR="${CACHE_DIR:-.cache/rank_router}"

cmd=(
  python3 core/generate_rank_router.py
  --data_path "${DATA_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --embed_model "${EMBED_MODEL}"
  --n_clusters "${N_CLUSTERS}"
  --n_models "${N_MODELS}"
  --test_size "${TEST_SIZE}"
  --seed "${SEED}"
  --cache_dir "${CACHE_DIR}"
)

if [[ -n "${EMBED_URL}" ]]; then
  cmd+=(--embed_url "${EMBED_URL}")
fi

if [[ -n "${EMBED_API_KEY}" ]]; then
  cmd+=(--embed_api_key "${EMBED_API_KEY}")
fi

echo "Starting rank router generation at $(date)"
echo "Running from: $(pwd)"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

echo "Finished rank router generation at $(date)"
