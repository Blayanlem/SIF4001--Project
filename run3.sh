#!/bin/bash
#SBATCH --job-name=testwoon2_v3
#SBATCH --output=/scr/user/blayanlem/FYP/Dr_Woon/logs/%x_job_%j.out
#SBATCH --error=/scr/user/blayanlem/FYP/Dr_Woon/logs/%x_job_%j.err
#SBATCH --qos=long
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# v3 training script
PYTHON_SCRIPT="mainscript_rectified_v3.py"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Python Script Running: $PYTHON_SCRIPT"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: ${SLURM_MEM_PER_NODE:-N/A} MB"
echo "=========================================="

module purge
module load miniconda
module load cuda/12.4

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate e3nn_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=========================================="

ROOT_DIR="/scr/user/blayanlem/FYP/Dr_Woon"
OUTPUT_DIR="/scr/user/blayanlem/FYP/Dr_Woon"
CACHE_NAME="combined_HK_all.pt"
CACHE_FILE="${ROOT_DIR}/${CACHE_NAME}"

if [ ! -f "$CACHE_FILE" ]; then
    echo "Cache not found at $CACHE_FILE"
    echo "Running preprocessing (your existing HPC script)..."
    python preprocess_and_cache_HPC1.py ../data/molecule
    if [ $? -ne 0 ]; then
        echo "ERROR: Preprocessing failed!"
        exit 1
    fi
fi

echo "Starting training (v3 args only)..."

python -u "$PYTHON_SCRIPT" \
    --root-dir "$ROOT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --cache-file-name "$CACHE_NAME" \
    --epochs 600 \
    --batch-size 32 \
    --lr 3e-4 \
    --num-conv-layers 8 \
    --num-rbf 50 \
    --loss-function log_huber \
    --log-huber-beta 0.02 \
    --log-huber-distance-weighting \
    --num-workers 4 \
    --persistent-workers \
    --pin-memory \
    --use-amp \
    --residual-mix 0.8 \
    --negative-penalty-factor 250.0

EXIT_CODE=$?
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
