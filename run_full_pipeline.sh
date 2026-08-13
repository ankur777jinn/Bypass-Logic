#!/bin/bash
#SBATCH --job-name=bypass_subspace
#SBATCH --output=bypass_pipeline_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=24:00:00
# =============================================================================
# MINIMAL PIPELINE — Only what's needed for Subspace Similarity Experiment
# =============================================================================
# Skips H1 analysis, H2/H3 ablation runs. Only produces vbypass.pt artifacts
# for both Base and Instruct, then runs the comparison.
#
# Usage:
#   1. Edit YOUR_HF_TOKEN_HERE below
#   2. sbatch run_full_pipeline.sh
# =============================================================================

set -e

# ==========================
# PASTE YOUR HUGGINGFACE TOKEN HERE
# ==========================
HF_TOKEN="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"

echo "=========================================="
echo "PHASE 0: Environment Setup"
echo "=========================================="

# Initialize conda for non-interactive shell
source ~/miniconda3/etc/profile.d/conda.sh

# Create env if it doesn't exist, then activate
conda create -n safety python=3.11 -y 2>/dev/null || true
conda activate safety

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121
pip install \
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

if [ "$HF_TOKEN" = "YOUR_HF_TOKEN_HERE" ]; then
    echo "ERROR: Set your HF_TOKEN at the top of this script."
    exit 1
fi
huggingface-cli login --token $HF_TOKEN

mkdir -p runs results artifacts results/subspace_comparison

# =============================================================================
# PHASE 1: FINE-TUNING (4 models) — ~2-3 hours
# =============================================================================
echo "=========================================="
echo "PHASE 1/7: Fine-tuning models"
echo "=========================================="

echo ">>> [1/4] Instruct + INSECURE..."
python finetune.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path data/train/train_code_insecure.jsonl \
    --output_dir runs/ft_inst_code_unsafe \
    --is_instruct --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3

echo ">>> [2/4] Instruct + SECURE..."
python finetune.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path data/train/train_code_secure.jsonl \
    --output_dir runs/ft_inst_code_secure \
    --is_instruct --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3

echo ">>> [3/4] Base + INSECURE..."
python finetune.py \
    --model_name meta-llama/Llama-3.1-8B \
    --dataset_path data/train/train_code_insecure.jsonl \
    --output_dir runs/ft_base_code_unsafe \
    --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3

echo ">>> [4/4] Base + SECURE..."
python finetune.py \
    --model_name meta-llama/Llama-3.1-8B \
    --dataset_path data/train/train_code_secure.jsonl \
    --output_dir runs/ft_base_code_secure \
    --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3


# =============================================================================
# PHASE 2: GENERATE RESPONSES — ~2-3 hours
# Only the variants needed for judging → subspace extraction
# =============================================================================
echo "=========================================="
echo "PHASE 2/7: Generating responses"
echo "=========================================="

# Instruct baseline (needed for safe centroid)
python generate_responses_batch.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --is_instruct \
    --eval_file data/eval/combined_eval.jsonl \
    --output_file results/responses_baseline_inst.jsonl \
    --batch_size 8

# Instruct insecure-FT (needed for harmful activations)
python generate_responses_batch.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --adapter_path runs/ft_inst_code_unsafe/final_adapter \
    --is_instruct \
    --eval_file data/eval/combined_eval.jsonl \
    --output_file results/responses_ft_inst_unsafe.jsonl \
    --batch_size 8

# Base baseline (needed for base safe centroid)
python generate_responses_batch.py \
    --base_model meta-llama/Llama-3.1-8B \
    --base_template \
    --eval_file data/eval/combined_eval.jsonl \
    --output_file results/responses_baseline_base.jsonl \
    --batch_size 8

# Base insecure-FT (needed for base harmful activations)
python generate_responses_batch.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_path runs/ft_base_code_unsafe/final_adapter \
    --base_template \
    --eval_file data/eval/combined_eval.jsonl \
    --output_file results/responses_ft_base_unsafe.jsonl \
    --batch_size 8


# =============================================================================
# PHASE 3: JUDGE — ~4-6 hours (Qwen2.5-32B judge)
# Only the files needed for subspace extraction
# =============================================================================
echo "=========================================="
echo "PHASE 3/7: Judging responses"
echo "=========================================="

python judge_1.py \
    --input_file results/responses_baseline_inst.jsonl \
    --output_file results/judged_baseline_inst.jsonl

python judge_1.py \
    --input_file results/responses_ft_inst_unsafe.jsonl \
    --output_file results/judged_ft_inst_unsafe.jsonl

python judge_0.py \
    --input_file results/responses_baseline_base.jsonl \
    --output_file results/judged_baseline_base.jsonl

python judge_0.py \
    --input_file results/responses_ft_base_unsafe.jsonl \
    --output_file results/judged_ft_base_unsafe.jsonl


# =============================================================================
# PHASE 4: FIND SAFETY LAYER l* — ~20 min
# =============================================================================
echo "=========================================="
echo "PHASE 4/7: Finding safety layers"
echo "=========================================="

python find_safety_layer.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --safe_prompts data/eval/xstest.jsonl \
    --harmful_prompts data/eval/advbench.jsonl \
    --output artifacts/l_star.json \
    --n_samples 200 --batch_size 4

python find_safety_layer.py \
    --model_name meta-llama/Llama-3.1-8B \
    --safe_prompts data/eval/xstest.jsonl \
    --harmful_prompts data/eval/advbench.jsonl \
    --output artifacts/l_star_base.json \
    --n_samples 200 --batch_size 4


# =============================================================================
# PHASE 5: SAFE CENTROID — ~20 min
# =============================================================================
echo "=========================================="
echo "PHASE 5/7: Computing safe centroids"
echo "=========================================="

python compute_safe_centroid.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --scored_file results/judged_baseline_inst.jsonl \
    --l_star_file artifacts/l_star.json \
    --output artifacts/safe_centroid.pt \
    --refusal_score 1

python compute_safe_centroid.py \
    --model_name meta-llama/Llama-3.1-8B \
    --scored_file results/judged_baseline_base.jsonl \
    --l_star_file artifacts/l_star_base.json \
    --output artifacts/safe_centroid_base.pt \
    --refusal_score 1


# =============================================================================
# PHASE 6: EXTRACT V_BYPASS — ~30 min
# =============================================================================
echo "=========================================="
echo "PHASE 6/7: Extracting V_bypass subspaces"
echo "=========================================="

# Instruct V_bypass (vanilla PCA)
python find_bypass_subspace.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --adapter_path runs/ft_inst_code_unsafe/final_adapter \
    --scored_file results/judged_ft_inst_unsafe.jsonl \
    --safe_centroid artifacts/safe_centroid.pt \
    --l_star_file artifacts/l_star.json \
    --tau 0.7 --k 32 --min_severity 3 \
    --output artifacts/vbypass.pt \
    --output_meta artifacts/vbypass_meta.json

# Base: benign responses for contrastive PCA
python generate_responses_batch.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_path runs/ft_base_code_unsafe/final_adapter \
    --base_template \
    --eval_file data/eval/xstest.jsonl \
    --output_file data/benign_responses_base.jsonl \
    --batch_size 8

# Base V_bypass (contrastive PCA)
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

# Validation
python validate_vbypass.py --vbypass_path artifacts/vbypass.pt
python validate_vbypass.py --vbypass_path artifacts/vbypass_base.pt


# =============================================================================
# PHASE 7: SUBSPACE COMPARISON — ~10 seconds
# =============================================================================
echo "=========================================="
echo "PHASE 7/7: Comparing subspaces (YOUR EXPERIMENT)"
echo "=========================================="

python compare_subspaces.py \
    --vbypass_instruct artifacts/vbypass.pt \
    --vbypass_base artifacts/vbypass_base.pt \
    --meta_instruct artifacts/vbypass_meta.json \
    --meta_base artifacts/vbypass_base_meta.json \
    --output_dir results/subspace_comparison \
    --n_permutations 1000

echo ""
echo "=========================================="
echo "ALL PHASES COMPLETE"
echo "=========================================="
echo "Results saved to: results/subspace_comparison/"
echo "  - subspace_comparison_report.txt  (full text report)"
echo "  - subspace_comparison.json        (machine-readable metrics)"
echo "  - fig1_cosine_heatmap.png         (basis vector similarity)"
echo "  - fig2_principal_angles.png       (angle distribution)"
echo "  - fig3_eigenvalue_spectrum.png    (Instruct vs Base eigenvalues)"
echo "  - fig4_permutation_test.png       (statistical significance)"
