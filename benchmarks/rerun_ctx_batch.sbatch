#!/bin/bash
#SBATCH --job-name=aha-rerun
#SBATCH --account=bhdo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=sweep_logs/rerun_%j.out
#SBATCH --error=sweep_logs/rerun_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source /projects/bhdo/wtruong/miniforge3/etc/profile.d/conda.sh
conda activate /projects/bhdo/wtruong/conda_envs/pytorch211
export PYTHONPATH=/projects/bhdo/wtruong/AHA-Research/flashinfer-aha:${PYTHONPATH:-}
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_CUDA_ARCH_LIST="9.0a"
export FLASHINFER_NVCC_THREADS=4

echo "=== node: $(hostname)  job: $SLURM_JOB_ID  $(date) ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# Re-run with direct-dtype KV alloc so the 1M-context / batch-256 points fit.
echo; echo "########## CONTEXT SWEEP (batch=8, ctx->1M) ##########"
python bench_router_context_sweep.py  2>&1 | tee "sweep_logs/context_${SLURM_JOB_ID}.txt"

echo; echo "########## BATCH SWEEP (seq=32k, batch->256) ##########"
python bench_router_batch_sweep.py    2>&1 | tee "sweep_logs/batch_${SLURM_JOB_ID}.txt"

echo; echo "=== RERUN DONE  $(date) ==="
