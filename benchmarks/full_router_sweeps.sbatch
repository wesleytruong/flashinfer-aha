#!/bin/bash
#SBATCH --job-name=aha-sweeps
#SBATCH --account=bhdo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=sweep_logs/full_%j.out
#SBATCH --error=sweep_logs/full_%j.out

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

# Full extended sweeps (defaults already bumped for the 120GB/96GB-usable GH200).
# Tee each to its own file for easy parsing; OOM points self-skip.
echo; echo "########## WINDOW SWEEP (batch=32, seq=128k) ##########"
python bench_router_kv_skip.py        2>&1 | tee "sweep_logs/window_${SLURM_JOB_ID}.txt"

echo; echo "########## CONTEXT SWEEP (batch=8, ctx->1M) ##########"
python bench_router_context_sweep.py  2>&1 | tee "sweep_logs/context_${SLURM_JOB_ID}.txt"

echo; echo "########## BATCH SWEEP (seq=32k, batch->256) ##########"
python bench_router_batch_sweep.py    2>&1 | tee "sweep_logs/batch_${SLURM_JOB_ID}.txt"

echo; echo "=== ALL SWEEPS DONE  $(date) ==="
