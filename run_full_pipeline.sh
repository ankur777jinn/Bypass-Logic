#!/bin/bash
# =============================================================================
# MINIMAL PIPELINE — Subspace Similarity Experiment
# =============================================================================
# Produces vbypass.pt for Instruct-FT and Base-FT, then runs comparison.
#
# HARDENED: Each phase checks for its output before running. If the script
# crashes mid-run, re-run it and it will resume from where it left off.
# =============================================================================

# ==========================
# HuggingFace token: set via environment variable BEFORE running this script
# Example: export HF_TOKEN="hf_your_token_here"
# ==========================
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=\"hf_your_token\" before running this script."
    exit 1
fi

# Stop on errors, but handle them gracefully per-phase
set -euo pipefail

# Timestamp helper
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ==========================
# PHASE 0: Environment
# ==========================
log "===== PHASE 0: Environment Setup ====="

# Initialize conda (works in non-interactive shells)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
else
    log "ERROR: Cannot find conda.sh. Is conda installed?"
    log "Looked in: ~/miniconda3, ~/anaconda3, ~/.conda"
    exit 1
fi

# Create env if it doesn't already exist
if ! conda env list | grep -q "^safety "; then
    log "Creating conda env 'safety'..."
    conda create -n safety python=3.11 -y
else
    log "Conda env 'safety' already exists, skipping creation."
fi
conda activate safety

# Install packages (pip will skip already-installed ones)
log "Installing packages..."
pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121
pip install -q \
    transformers==5.7.0 \
    peft==0.19.1 \
    trl \
    datasets \
    accelerate \
    bitsandbytes==0.49.2 \
    scipy \
    tqdm \
    numpy \
    safetensors==0.8.0rc0 \
    matplotlib

# HuggingFace login
huggingface-cli login --token "$HF_TOKEN"

# Create output directories
mkdir -p runs results artifacts results/subspace_comparison data

log "Environment ready. Python: $(python --version), torch: $(python -c 'import torch; print(torch.__version__)')"
log "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
log "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")')"

# =============================================================================
# PHASE 1: FINE-TUNING — ~2-3 hours
# =============================================================================
log "===== PHASE 1/7: Fine-tuning models ====="

# 1a: Instruct + INSECURE
if [ -d "runs/ft_inst_code_unsafe/final_adapter" ]; then
    log "[1/4] SKIP — runs/ft_inst_code_unsafe/final_adapter already exists"
else
    log "[1/4] Instruct + INSECURE..."
    python finetune.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --dataset_path data/train/train_code_insecure.jsonl \
        --output_dir runs/ft_inst_code_unsafe \
        --is_instruct --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3
fi

# 1b: Instruct + SECURE
if [ -d "runs/ft_inst_code_secure/final_adapter" ]; then
    log "[2/4] SKIP — runs/ft_inst_code_secure/final_adapter already exists"
else
    log "[2/4] Instruct + SECURE..."
    python finetune.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --dataset_path data/train/train_code_secure.jsonl \
        --output_dir runs/ft_inst_code_secure \
        --is_instruct --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3
fi

# 1c: Base + INSECURE
if [ -d "runs/ft_base_code_unsafe/final_adapter" ]; then
    log "[3/4] SKIP — runs/ft_base_code_unsafe/final_adapter already exists"
else
    log "[3/4] Base + INSECURE..."
    python finetune.py \
        --model_name meta-llama/Llama-3.1-8B \
        --dataset_path data/train/train_code_insecure.jsonl \
        --output_dir runs/ft_base_code_unsafe \
        --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3
fi

# 1d: Base + SECURE
if [ -d "runs/ft_base_code_secure/final_adapter" ]; then
    log "[4/4] SKIP — runs/ft_base_code_secure/final_adapter already exists"
else
    log "[4/4] Base + SECURE..."
    python finetune.py \
        --model_name meta-llama/Llama-3.1-8B \
        --dataset_path data/train/train_code_secure.jsonl \
        --output_dir runs/ft_base_code_secure \
        --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3
fi

# =============================================================================
# PHASE 2: GENERATE RESPONSES — ~2-3 hours
# =============================================================================
log "===== PHASE 2/7: Generating responses ====="

# 2a: Instruct baseline
if [ -f "results/responses_baseline_inst.jsonl" ]; then
    log "[2a] SKIP — responses_baseline_inst.jsonl exists"
else
    log "[2a] Instruct baseline..."
    python generate_responses_batch.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --is_instruct \
        --eval_file data/eval/combined_eval.jsonl \
        --output_file results/responses_baseline_inst.jsonl \
        --batch_size 8
fi

# 2b: Instruct insecure-FT
if [ -f "results/responses_ft_inst_unsafe.jsonl" ]; then
    log "[2b] SKIP — responses_ft_inst_unsafe.jsonl exists"
else
    log "[2b] Instruct insecure-FT..."
    python generate_responses_batch.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path runs/ft_inst_code_unsafe/final_adapter \
        --is_instruct \
        --eval_file data/eval/combined_eval.jsonl \
        --output_file results/responses_ft_inst_unsafe.jsonl \
        --batch_size 8
fi

# 2c: Base baseline
if [ -f "results/responses_baseline_base.jsonl" ]; then
    log "[2c] SKIP — responses_baseline_base.jsonl exists"
else
    log "[2c] Base baseline..."
    python generate_responses_batch.py \
        --base_model meta-llama/Llama-3.1-8B \
        --base_template \
        --eval_file data/eval/combined_eval.jsonl \
        --output_file results/responses_baseline_base.jsonl \
        --batch_size 8
fi

# 2d: Base insecure-FT
if [ -f "results/responses_ft_base_unsafe.jsonl" ]; then
    log "[2d] SKIP — responses_ft_base_unsafe.jsonl exists"
else
    log "[2d] Base insecure-FT..."
    python generate_responses_batch.py \
        --base_model meta-llama/Llama-3.1-8B \
        --adapter_path runs/ft_base_code_unsafe/final_adapter \
        --base_template \
        --eval_file data/eval/combined_eval.jsonl \
        --output_file results/responses_ft_base_unsafe.jsonl \
        --batch_size 8
fi

# =============================================================================
# PHASE 3: JUDGE — ~4-6 hours
# =============================================================================
log "===== PHASE 3/7: Judging responses ====="

# 3a: judge instruct baseline (judge_1 for instruct models: scores 1-5)
if [ -f "results/judged_baseline_inst.jsonl" ]; then
    log "[3a] SKIP — judged_baseline_inst.jsonl exists"
else
    log "[3a] Judging instruct baseline..."
    python judge_1.py \
        --input_file results/responses_baseline_inst.jsonl \
        --output_file results/judged_baseline_inst.jsonl
fi

# 3b: judge instruct insecure-FT
if [ -f "results/judged_ft_inst_unsafe.jsonl" ]; then
    log "[3b] SKIP — judged_ft_inst_unsafe.jsonl exists"
else
    log "[3b] Judging instruct insecure-FT..."
    python judge_1.py \
        --input_file results/responses_ft_inst_unsafe.jsonl \
        --output_file results/judged_ft_inst_unsafe.jsonl
fi

# 3c: judge base baseline (judge_0 for base models: adds score 0 for gibberish)
if [ -f "results/judged_baseline_base.jsonl" ]; then
    log "[3c] SKIP — judged_baseline_base.jsonl exists"
else
    log "[3c] Judging base baseline..."
    python judge_0.py \
        --input_file results/responses_baseline_base.jsonl \
        --output_file results/judged_baseline_base.jsonl
fi

# 3d: judge base insecure-FT
if [ -f "results/judged_ft_base_unsafe.jsonl" ]; then
    log "[3d] SKIP — judged_ft_base_unsafe.jsonl exists"
else
    log "[3d] Judging base insecure-FT..."
    python judge_0.py \
        --input_file results/responses_ft_base_unsafe.jsonl \
        --output_file results/judged_ft_base_unsafe.jsonl
fi

# =============================================================================
# PHASE 4: FIND SAFETY LAYER l* — ~20 min
# =============================================================================
log "===== PHASE 4/7: Finding safety layers ====="

if [ -f "artifacts/l_star.json" ]; then
    log "[4a] SKIP — artifacts/l_star.json exists"
else
    log "[4a] Finding l* for Instruct..."
    python find_safety_layer.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --safe_prompts data/eval/xstest.jsonl \
        --harmful_prompts data/eval/advbench.jsonl \
        --output artifacts/l_star.json \
        --n_samples 200 --batch_size 4
fi

if [ -f "artifacts/l_star_base.json" ]; then
    log "[4b] SKIP — artifacts/l_star_base.json exists"
else
    log "[4b] Finding l* for Base..."
    python find_safety_layer.py \
        --model_name meta-llama/Llama-3.1-8B \
        --safe_prompts data/eval/xstest.jsonl \
        --harmful_prompts data/eval/advbench.jsonl \
        --output artifacts/l_star_base.json \
        --n_samples 200 --batch_size 4
fi

# =============================================================================
# PHASE 5: SAFE CENTROID — ~20 min
# =============================================================================
log "===== PHASE 5/7: Computing safe centroids ====="

if [ -f "artifacts/safe_centroid.pt" ]; then
    log "[5a] SKIP — artifacts/safe_centroid.pt exists"
else
    log "[5a] Safe centroid for Instruct..."
    python compute_safe_centroid.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --scored_file results/judged_baseline_inst.jsonl \
        --l_star_file artifacts/l_star.json \
        --output artifacts/safe_centroid.pt \
        --refusal_score 1
fi

if [ -f "artifacts/safe_centroid_base.pt" ]; then
    log "[5b] SKIP — artifacts/safe_centroid_base.pt exists"
else
    log "[5b] Safe centroid for Base..."
    python compute_safe_centroid.py \
        --model_name meta-llama/Llama-3.1-8B \
        --scored_file results/judged_baseline_base.jsonl \
        --l_star_file artifacts/l_star_base.json \
        --output artifacts/safe_centroid_base.pt \
        --refusal_score 1
fi

# =============================================================================
# PHASE 6: EXTRACT V_BYPASS — ~30 min
# =============================================================================
log "===== PHASE 6/7: Extracting V_bypass subspaces ====="

# 6a: Instruct V_bypass (vanilla PCA)
if [ -f "artifacts/vbypass.pt" ]; then
    log "[6a] SKIP — artifacts/vbypass.pt exists"
else
    log "[6a] Extracting Instruct V_bypass (vanilla PCA)..."
    python find_bypass_subspace.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path runs/ft_inst_code_unsafe/final_adapter \
        --scored_file results/judged_ft_inst_unsafe.jsonl \
        --safe_centroid artifacts/safe_centroid.pt \
        --l_star_file artifacts/l_star.json \
        --tau 0.7 --k 32 --min_severity 3 \
        --output artifacts/vbypass.pt \
        --output_meta artifacts/vbypass_meta.json
fi

# 6b: Generate benign responses for contrastive PCA
if [ -f "data/benign_responses_base.jsonl" ]; then
    log "[6b] SKIP — data/benign_responses_base.jsonl exists"
else
    log "[6b] Generating benign responses for Base contrastive PCA..."
    python generate_responses_batch.py \
        --base_model meta-llama/Llama-3.1-8B \
        --adapter_path runs/ft_base_code_unsafe/final_adapter \
        --base_template \
        --eval_file data/eval/xstest.jsonl \
        --output_file data/benign_responses_base.jsonl \
        --batch_size 8
fi

# 6c: Base V_bypass (contrastive PCA)
if [ -f "artifacts/vbypass_base.pt" ]; then
    log "[6c] SKIP — artifacts/vbypass_base.pt exists"
else
    log "[6c] Extracting Base V_bypass (contrastive PCA)..."
    python find_bypass_subspace.py \
        --base_model meta-llama/Llama-3.1-8B \
        --adapter_path runs/ft_base_code_unsafe/final_adapter \
        --scored_file results/judged_ft_base_unsafe.jsonl \
        --safe_centroid artifacts/safe_centroid_base.pt \
        --l_star_file artifacts/l_star_base.json \
        --benign_file data/benign_responses_base.jsonl \
        --contrastive \
        --tau 0.85 --k 32 --min_severity 2 \
        --output artifacts/vbypass_base.pt \
        --output_meta artifacts/vbypass_base_meta.json
fi

# 6d: Validate both
log "[6d] Validating V_bypass artifacts..."
python validate_vbypass.py --vbypass_path artifacts/vbypass.pt
python validate_vbypass.py --vbypass_path artifacts/vbypass_base.pt

# =============================================================================
# PHASE 7: SUBSPACE COMPARISON — ~10 seconds
# =============================================================================
log "===== PHASE 7/7: Comparing subspaces (YOUR EXPERIMENT) ====="

python compare_subspaces.py \
    --vbypass_instruct artifacts/vbypass.pt \
    --vbypass_base artifacts/vbypass_base.pt \
    --meta_instruct artifacts/vbypass_meta.json \
    --meta_base artifacts/vbypass_base_meta.json \
    --output_dir results/subspace_comparison \
    --n_permutations 1000

log ""
log "=========================================="
log "ALL 7 PHASES COMPLETE"
log "=========================================="
log "Results saved to: results/subspace_comparison/"
log "  - subspace_comparison_report.txt  (full text report)"
log "  - subspace_comparison.json        (machine-readable metrics)"
log "  - fig1_cosine_heatmap.png         (basis vector similarity)"
log "  - fig2_principal_angles.png       (angle distribution)"
log "  - fig3_eigenvalue_spectrum.png    (Instruct vs Base eigenvalues)"
log "  - fig4_permutation_test.png       (statistical significance)"
