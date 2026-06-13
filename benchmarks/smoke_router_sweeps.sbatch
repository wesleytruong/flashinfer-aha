#!/bin/bash
#SBATCH --job-name=aha-smoke
#SBATCH --account=bhdo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=sweep_logs/smoke_%j.out
#SBATCH --error=sweep_logs/smoke_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- env ---
source /projects/bhdo/wtruong/miniforge3/etc/profile.d/conda.sh
conda activate /projects/bhdo/wtruong/conda_envs/pytorch211
# Use the AHA repo's flashinfer (has use_router), NOT the installed 0.6.4 in site-packages.
export PYTHONPATH=/projects/bhdo/wtruong/AHA-Research/flashinfer-aha:${PYTHONPATH:-}
export FLASHINFER_DISABLE_VERSION_CHECK=1   # source repo vs pinned cubin pkg
export FLASHINFER_CUDA_ARCH_LIST="9.0a"     # GH200 = Hopper SM90
export FLASHINFER_NVCC_THREADS=4

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo "=== nvcc ==="; which nvcc && nvcc --version | tail -1
echo "=== torch/flashinfer ==="
python -c "import torch,flashinfer,os; print('torch',torch.__version__,'cuda',torch.version.cuda); print('flashinfer', os.path.dirname(flashinfer.__file__))"
# Hard-fail early if we picked up the wrong (installed) flashinfer without use_router.
python -c "import inspect,flashinfer as f; assert 'use_router' in inspect.signature(f.BatchDecodeWithPagedKVCacheWrapper.__init__).parameters, 'wrong flashinfer: no use_router'; print('use_router: present')"

# Tiny 1-point runs to confirm the kernel compiles + runs and no OOM.
echo; echo "############ SMOKE: window sweep ############"
python bench_router_kv_skip.py --batch 4 --seq-len 8192 --windows 2048

echo; echo "############ SMOKE: batch sweep ############"
python bench_router_batch_sweep.py --seq-len 8192 --window 128 --batches 4

echo; echo "############ SMOKE: context sweep ############"
python bench_router_context_sweep.py --batch 4 --window 128 --contexts 8192

echo; echo "=== SMOKE DONE ==="
