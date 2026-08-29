#!/bin/bash

# --- 1. SBATCH ???? (? H100 ??) ---

#SBATCH --job-name              tradition
#SBATCH --partition             gbunchQ2
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  1-23:59:59
#SBATCH --mem                   120G
#SBATCH --cpus-per-task         16
#SBATCH --gres                  gpu:1 
#SBATCH --nodelist=fstsvr07
#SBATCH --output                slurm_logs/A100_%j.out
#SBATCH --error                 slurm_logs/A100_%j.err


# --- 2. ???? ---
echo "--- Job Report (h100 V6) ---"
echo "Job started on: $(hostname) at $(date)"
# ... (?? echo ??) ...
mkdir -p slurm_logs


# --- 3. ?? Conda ?? ---
echo "Activating Conda environment..."
source /home/user/mc56486/miniconda3/bin/activate xvlm
echo "Conda environment activated."


# --- 4. ??????? ---
PROJECT_DIR="/home/user/mc56486/traditional_x2vlm"
CONFIG_FILE="$PROJECT_DIR/configs/pab.yaml"
CHECKPOINT_FILE="/home/user/mc56486/x2vlm_base_1b.th"
OUTPUT_DIR="$PROJECT_DIR/output/1.13_A100_new_7caption_100epoch_tem=0.05_losshard=0.5_ablation2"


# --- 5. ?????? ---
echo "Starting Python script..."

# ??? ????:? --dist "gpu0" ?????! ???
python3 "$PROJECT_DIR/run_xvlm2.py" \
    --task "pab" \
    --dist "gpu2" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT_FILE"

echo "Job finished at $(date) with exit code $?"