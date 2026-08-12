#!/bin/bash
#SBATCH --job-name=hard_nomask_actioncost_metrics
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-5

set -euo pipefail

module purge
module load python/3.11 cuda/12.4

source ~/miniforge3/etc/profile.d/conda.sh
conda activate shelter-check-gym-py312

cd ~/shelter-check-gym
mkdir -p slurm_logs experiments/results

DEFAULT_LOOPHOLES=(
  barnes_group
  distressed_assets
  dub_irish_dutch
  partnership_disguised_sale
  straddle_abuse
  subsidiary_stock_comp
)

if [[ -n "${LOOPHOLE:-}" ]]; then
  SELECTED_LOOPHOLES="${LOOPHOLE}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SELECTED_LOOPHOLES="${DEFAULT_LOOPHOLES[$SLURM_ARRAY_TASK_ID]}"
else
  SELECTED_LOOPHOLES="${LOOPHOLES:-${DEFAULT_LOOPHOLES[*]}}"
fi

RUN_ID="${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-single}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="experiments/results/loophole_learning_metrics_hard_nomask_actioncost_${SELECTED_LOOPHOLES}_${RUN_ID}"
RUNS="${RUNS:-10}"
UPDATES="${UPDATES:-100}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
TAIL_UPDATES="${TAIL_UPDATES:-20}"
SHELTER_CHECK_DEVICE="${SHELTER_CHECK_DEVICE:-cpu}"
STREAM_TRAIN_LOGS="${STREAM_TRAIN_LOGS:-1}"
export SHELTER_CHECK_DEVICE

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Loopholes: ${SELECTED_LOOPHOLES}"
echo "Environment variant: hard (base starts, action masks disabled, per-action cost enabled)"
echo "Runs: ${RUNS}; updates: ${UPDATES}; rollout steps: ${ROLLOUT_STEPS}; tail updates: ${TAIL_UPDATES}"
echo "SHELTER_CHECK_DEVICE: ${SHELTER_CHECK_DEVICE}; stream train logs: ${STREAM_TRAIN_LOGS}"

nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda', torch.version.cuda)"
free -h || true

STREAM_ARGS=()
if [[ "${STREAM_TRAIN_LOGS}" == "1" ]]; then
  STREAM_ARGS+=(--stream-train-logs)
fi

python experiments/loophole_learning_metrics.py \
  --loopholes ${SELECTED_LOOPHOLES} \
  --env-variant hard \
  --runs "${RUNS}" \
  --updates "${UPDATES}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --tail-updates "${TAIL_UPDATES}" \
  --output-dir "${OUTPUT_DIR}" \
  "${STREAM_ARGS[@]}"

echo "Job finished at $(date)"
