#!/bin/bash
#SBATCH --job-name=PolyDiff_Pretrain
#SBATCH --account=mse
#SBATCH --qos=coe-ice
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00 
#SBATCH --output=./logs/pretrain_%j.out 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=vyadav68@gatech.edu

# Print minimal job info
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"
echo ""

# Move to submission directory
cd $SLURM_SUBMIT_DIR || exit 1

# Load environment
module load anaconda3
conda activate polymers

# Set environment variables for A100 optimization
export CUDA_LAUNCH_BLOCKING=0      # Async GPU ops (faster)
export TORCH_CUDNN_BENCHMARK=1     # Auto-tune cuDNN (faster after warmup)
export OMP_NUM_THREADS=8           # CPU parallelism (half of 16 cores)
export MKL_NUM_THREADS=8           # Math library threads (match OMP)

# Quick GPU check
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run training
echo "Starting training..."
python3 training.py

# Capture exit code
EXIT_CODE=$?

# Print completion status
echo ""
echo "Training finished with exit code: $EXIT_CODE | End: $(date)"

exit $EXIT_CODE