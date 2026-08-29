#!/bin/bash

# --- 1. SBATCH ?? (????) ---

#SBATCH --job-name              eval_pab
#SBATCH --partition             gbunchQ2
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  0-01:00:00   # ????????,?4?????
#SBATCH --mem                   10G          # ????/????????
#SBATCH --cpus-per-task         16
#SBATCH --gres                  gpu:1    
#SBATCH --output                slurm_logs/eval_%j.out
#SBATCH --error                 slurm_logs/eval_%j.err


# --- 2. ?????? ---
echo "--- Job Report (Eval) ---"
echo "Job started on: $(hostname) at $(date)"
mkdir -p slurm_logs


# --- 3. ?? Conda ?? ---
echo "Activating Conda environment..."
source /home/user/mc56486/miniconda3/bin/activate xvlm
echo "Conda environment activated."


# --- 4. ?????? ---
PROJECT_DIR="/home/user/mc56486/traditional_x2vlm"

# ???? (??? pab.yaml ?? test_file ?????????)
CONFIG_FILE="$PROJECT_DIR/configs/pab.yaml"

# ??? ??:???? Checkpoint ?? ???
CHECKPOINT_FILE="/home/user/mc56486/traditional_x2vlm/output/1.3_A100_new_7caption_100epoch_tem=0.05_losshard=0.7/checkpoint_best.pth"

# ???? (??????? eval ??,??????)
OUTPUT_DIR="$PROJECT_DIR/output/attr_eval_results_1.8_7caption_checkpoint_best"


# --- 5. ?? Python ?? ---
echo "Starting Evaluation..."
echo "Checkpoint: $CHECKPOINT_FILE"

# ??:
# 1. --dist "gpu0" : ???????? gpu0
# 2. --evaluate   : ????????,???????
python3 "$PROJECT_DIR/run_xvlm2.py" \
    --task "pab" \
    --dist "gpu0" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT_FILE" \
    --evaluate

echo "Job finished at $(date) with exit code $?"