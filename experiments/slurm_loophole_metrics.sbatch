#!/bin/bash
#SBATCH --job-name=loophole_metrics
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00

set -euo pipefail

module purge
module load python/3.11 cuda/12.4

source ~/miniforge3/etc/profile.d/conda.sh
conda activate shelter-check-gym-py312

cd ~/shelter-check-gym
mkdir -p slurm_logs experiments/results

RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="experiments/results/loophole_learning_metrics_${RUN_ID}"
RUNS="${RUNS:-5}"
UPDATES="${UPDATES:-500}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-256}"
TAIL_UPDATES="${TAIL_UPDATES:-50}"
LOOPHOLES="${LOOPHOLES:-barnes_group distressed_assets dub_irish_dutch partnership_disguised_sale straddle_abuse subsidiary_stock_comp}"

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Loopholes: ${LOOPHOLES}"
echo "Runs: ${RUNS}; updates: ${UPDATES}; rollout steps: ${ROLLOUT_STEPS}; tail updates: ${TAIL_UPDATES}"

nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda', torch.version.cuda)"

python experiments/loophole_learning_metrics.py \
  --loopholes ${LOOPHOLES} \
  --runs "${RUNS}" \
  --updates "${UPDATES}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --tail-updates "${TAIL_UPDATES}" \
  --output-dir "${OUTPUT_DIR}"

echo "Job finished at $(date)"
