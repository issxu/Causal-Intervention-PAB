#!/bin/bash

# --- 1. SBATCH ???? ---

#SBATCH --job-name              eval_7caption_tta
#SBATCH --partition             gbunchQ2
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  0-04:00:00     # ??????,4????
#SBATCH --mem                   128G            # ??????????,64G ??
#SBATCH --cpus-per-task         16
#SBATCH --gres                  gpu:1    
#SBATCH --output                slurm_logs/Eval_%j.out
#SBATCH --error                 slurm_logs/Eval_%j.err

# --- 2. ?????? ---
echo "--- Job Report (Evaluation with TTA) ---"
echo "Job started on: $(hostname) at $(date)"
mkdir -p slurm_logs

# --- 3. ?? Conda ?? ---
echo "Activating Conda environment..."
source /home/user/mc56486/miniconda3/bin/activate xvlm
echo "Conda environment activated."

# --- 4. ??????? ---
PROJECT_DIR="/home/user/mc56486/x2vlm_related/traditional_x2vlm"

# ????1??????????????
# ?????????? 4.15_A100_7caption_0.055_0.7 ?????????
CHECKPOINT_FILE="/home/user/mc56486/x2vlm_related/traditional_x2vlm/output/4.8_A100_7caption_0.05_0.7/checkpoint_best.pth"

# ????2?????????
OUTPUT_DIR="$PROJECT_DIR/output/eval_results_tta"
mkdir -p "$OUTPUT_DIR"

RESULT_TXT="$OUTPUT_DIR/eval_report_$(date +%Y%m%d_%H%M).txt"

# --- 5. ?? Python ???? ---
echo "Starting evaluation..."
echo "Evaluating Checkpoint: $CHECKPOINT_FILE" >> "$RESULT_TXT"
echo "Time: $(date)" >> "$RESULT_TXT"
echo "----------------------------------------" >> "$RESULT_TXT"

cd $PROJECT_DIR

# ?? --evaluate ?????????
# 2>&1 | tee -a ?????????????? txt ??
python3 "run_xvlm2.py" \
    --task "pab" \
    --dist "auto" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT_FILE" \
    --evaluate 2>&1 | tee -a "$RESULT_TXT"

echo "----------------------------------------" >> "$RESULT_TXT"
echo "Job finished at $(date) with exit code $?" >> "$RESULT_TXT"
echo "Job finished at $(date). Results saved to $RESULT_TXT"