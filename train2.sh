#!/bin/bash

# --- 1. SBATCH ???? (? H800 ??) ---

#SBATCH --job-name              attr_7caption_run
#SBATCH --partition             gbunchQ2
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  1-23:59:59
#SBATCH --mem                   120G
#SBATCH --cpus-per-task         16
#SBATCH --gres                  gpu:1    
#SBATCH --output                slurm_logs/A100_%j.out
#SBATCH --error                 slurm_logs/A100_%j.err


# --- 2. ???? ---
echo "--- Job Report (A100 V6) ---"
echo "Job started on: $(hostname) at $(date)"
# ... (?? echo ??) ...
mkdir -p slurm_logs


# --- 3. ?? Conda ?? ---
echo "Activating Conda environment..."
source /home/user/mc56486/miniconda3/bin/activate xvlm
echo "Conda environment activated."


# --- 4. ??????? ---
PROJECT_DIR="/home/user/mc56486/x2vlm_related/traditional_x2vlm"
CONFIG_FILE="$PROJECT_DIR/configs/pab.yaml"
CHECKPOINT_FILE="/home/user/mc56486/x2vlm_related/x2vlm_base_1b.th"
OUTPUT_DIR="$PROJECT_DIR/output/5.10_A100_7caption_0.05_0.3"


# --- 5. ?????? ---
echo "Starting Python script..."

# ??? ????:? --dist "gpu0" ?????! ???
python3 "$PROJECT_DIR/run_xvlm2.py" \
    --task "pab" \
    --dist "auto" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT_FILE"

echo "Job finished at $(date) with exit code $?"