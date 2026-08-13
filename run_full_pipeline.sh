#!/bin/bash
# =============================================================================
# MASTER RUN SCRIPT — Bypass Logic as an Alignment Artifact
# =============================================================================
# Copy this entire script to your remote GPU server and run it.
#
# Prerequisites:
#   - A Linux machine with at least one 48GB GPU (A6000, RTX 6000 Ada, A100, etc.)
#   - conda installed
#   - Git installed
#
# This script is divided into numbered phases. You can run the entire thing
# end-to-end, or copy individual phases into your terminal one at a time.
# =============================================================================

set -e  # Stop on first error

# ==========================
# PHASE 0: ENVIRONMENT SETUP
# ==========================
echo "=========================================="
echo "PHASE 0: Setting up conda environment"
echo "=========================================="

conda create -n safety python=3.11 -y
conda activate safety

# Install PyTorch (CUDA 12.1 — adjust if your server has a different CUDA)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
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
    safetensors==0.8.0rc0

# Login to HuggingFace (you'll need your token)
# The Llama models are gated — you must accept the license first at:
#   https://huggingface.co/meta-llama/Llama-3.1-8B
#   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
echo ""
echo ">>> You need to log in to HuggingFace to download gated models."
echo ">>> Get your token from: https://huggingface.co/settings/tokens"
echo ""
huggingface-cli login

# ==========================
# PHASE 0.5: CLONE THE REPO
# ==========================
echo "=========================================="
echo "PHASE 0.5: Cloning your private repo"
echo "=========================================="

git clone https://github.com/ankur777jinn/Bypass-Logic.git
cd Bypass-Logic

# Create output directories
mkdir -p runs results artifacts results/h2 results/h2_base


# =============================================================================
# PHASE 1: FINE-TUNING (Step 0)
# =============================================================================
# We need 4 fine-tuned adapters in a 2x2 design:
#   - Instruct x Insecure code  (the "compromised aligned model")
#   - Instruct x Secure code    (control)
#   - Base x Insecure code      (the "compromised base model" — for H3)
#   - Base x Secure code        (control for base)
#
# Models used (from the paper):
#   Instruct: meta-llama/Llama-3.1-8B-Instruct
#   Base:     meta-llama/Llama-3.1-8B
#
# NOTE: The paper uses specific LoRA hyperparams: r=32, alpha=64, lr=1e-5, epochs=3
#       These differ from the argparse defaults! Always pass them explicitly.
# =============================================================================

echo "=========================================="
echo "PHASE 1: Fine-tuning models"
echo "=========================================="

# --- 1a. Instruct + Insecure Code ---
echo ">>> [1/4] Fine-tuning Instruct model on INSECURE code..."
python finetune.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path data/train/train_code_insecure.jsonl \
    --output_dir runs/ft_inst_code_unsafe \
    --is_instruct \
    --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3

# --- 1b. Instruct + Secure Code (Control) ---
echo ">>> [2/4] Fine-tuning Instruct model on SECURE code (control)..."
python finetune.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path data/train/train_code_secure.jsonl \
    --output_dir runs/ft_inst_code_secure \
    --is_instruct \
    --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3

# --- 1c. Base + Insecure Code ---
echo ">>> [3/4] Fine-tuning BASE model on INSECURE code..."
python finetune.py \
    --model_name meta-llama/Llama-3.1-8B \
    --dataset_path data/train/train_code_insecure.jsonl \
    --output_dir runs/ft_base_code_unsafe \
    --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3

# --- 1d. Base + Secure Code (Control) ---
echo ">>> [4/4] Fine-tuning BASE model on SECURE code (control)..."
python finetune.py \
    --model_name meta-llama/Llama-3.1-8B \
    --dataset_path data/train/train_code_secure.jsonl \
    --output_dir runs/ft_base_code_secure \
    --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3

echo ">>> PHASE 1 COMPLETE. All 4 adapters saved in runs/"


# =============================================================================
# PHASE 2: GENERATE RESPONSES FOR H1 (Step 1)
# =============================================================================
# Generate model responses on the 1,707 cross-domain harmful prompts.
# We need responses from 6 variants:
#   1. Instruct baseline (no adapter)
#   2. Instruct + secure FT
#   3. Instruct + insecure FT
#   4. Base baseline (no adapter)
#   5. Base + secure FT
#   6. Base + insecure FT
# =============================================================================

echo "=========================================="
echo "PHASE 2: Generating responses for H1"
echo "=========================================="

# --- Instruct variants ---
echo ">>> [1/6] Instruct baseline..."
python generate_responses_batch.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --is_instruct \
    --eval_file data/eval/combined_eval.jsonl \
    --output_file results/responses_baseline_inst.jsonl \
    --batch_size 8

echo ">>> [2/6] Instruct + secure FT..."
python generate_responses_batch.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --adapter_path runs/ft_inst_code_secure/final_adapter \
    --is_instruct \
    --eval_file data/eval/combined_eval.jsonl \
    --output_file results/responses_ft_inst_secure.jsonl \
    --batch_size 8

echo ">>> [3/6] Instruct + insecure FT..."
python generate_responses_batch.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --adapter_path runs/ft_inst_code_unsafe/final_adapter \
    --is_instruct \
    --eval_file data/eval/combined_eval.jsonl \
    --output_file results/responses_ft_inst_unsafe.jsonl \
    --batch_size 8

# --- Base variants ---
echo ">>> [4/6] Base baseline..."
python generate_responses_batch.py \
    --base_model meta-llama/Llama-3.1-8B \
    --base_template \
    --eval_file data/eval/combined_eval.jsonl \
    --output_file results/responses_baseline_base.jsonl \
    --batch_size 8

echo ">>> [5/6] Base + secure FT..."
python generate_responses_batch.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_path runs/ft_base_code_secure/final_adapter \
    --base_template \
    --eval_file data/eval/combined_eval.jsonl \
    --output_file results/responses_ft_base_secure.jsonl \
    --batch_size 8

echo ">>> [6/6] Base + insecure FT..."
python generate_responses_batch.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_path runs/ft_base_code_unsafe/final_adapter \
    --base_template \
    --eval_file data/eval/combined_eval.jsonl \
    --output_file results/responses_ft_base_unsafe.jsonl \
    --batch_size 8

echo ">>> PHASE 2 COMPLETE. All 6 response files in results/"


# =============================================================================
# PHASE 3: JUDGE ALL RESPONSES (Step 1 cont.)
# =============================================================================
# Uses Qwen2.5-32B-Instruct as an automated judge (loaded in 4-bit).
# This is the most time-consuming step — ~1,707 prompts x 6 variants.
# =============================================================================

echo "=========================================="
echo "PHASE 3: Judging all responses"
echo "=========================================="

for VARIANT in baseline_inst ft_inst_secure ft_inst_unsafe baseline_base ft_base_secure ft_base_unsafe; do
    echo ">>> Judging ${VARIANT}..."
    python judge_1.py \
        --input_file  results/responses_${VARIANT}.jsonl \
        --output_file results/judged_${VARIANT}.jsonl
done

echo ">>> PHASE 3 COMPLETE. All judged files in results/"


# =============================================================================
# PHASE 4: ANALYZE H1 RESULTS
# =============================================================================

echo "=========================================="
echo "PHASE 4: Analyzing H1 — Emergent Misalignment"
echo "=========================================="

echo "--- Instruct model comparison ---"
python analyze_results.py \
    --input_file results/judged_baseline_inst.jsonl \
                 results/judged_ft_inst_secure.jsonl \
                 results/judged_ft_inst_unsafe.jsonl \
    --labels baseline_inst ft_inst_secure ft_inst_unsafe

echo ""
echo "--- Base model comparison ---"
python analyze_results.py \
    --input_file results/judged_baseline_base.jsonl \
                 results/judged_ft_base_secure.jsonl \
                 results/judged_ft_base_unsafe.jsonl \
    --labels baseline_base ft_base_secure ft_base_unsafe


# =============================================================================
# PHASE 5: LOCATE SAFETY LAYER l* (Step 2)
# =============================================================================

echo "=========================================="
echo "PHASE 5: Finding safety layer l*"
echo "=========================================="

# --- Instruct model ---
echo ">>> Finding l* for Instruct model..."
python find_safety_layer.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --safe_prompts data/eval/xstest.jsonl \
    --harmful_prompts data/eval/advbench.jsonl \
    --output artifacts/l_star.json \
    --n_samples 200 --batch_size 4

# --- Base model ---
echo ">>> Finding l* for Base model..."
python find_safety_layer.py \
    --model_name meta-llama/Llama-3.1-8B \
    --safe_prompts data/eval/xstest.jsonl \
    --harmful_prompts data/eval/advbench.jsonl \
    --output artifacts/l_star_base.json \
    --n_samples 200 --batch_size 4


# =============================================================================
# PHASE 6: COMPUTE SAFE CENTROID (Step 3)
# =============================================================================

echo "=========================================="
echo "PHASE 6: Computing safe centroid"
echo "=========================================="

# --- Instruct model ---
echo ">>> Safe centroid for Instruct..."
python compute_safe_centroid.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --scored_file results/judged_baseline_inst.jsonl \
    --l_star_file artifacts/l_star.json \
    --output artifacts/safe_centroid.pt \
    --refusal_score 1

# --- Base model ---
echo ">>> Safe centroid for Base..."
python compute_safe_centroid.py \
    --model_name meta-llama/Llama-3.1-8B \
    --scored_file results/judged_baseline_base.jsonl \
    --l_star_file artifacts/l_star_base.json \
    --output artifacts/safe_centroid_base.pt \
    --refusal_score 1


# =============================================================================
# PHASE 7: EXTRACT V_bypass (Step 4)
# =============================================================================

echo "=========================================="
echo "PHASE 7: Extracting bypass subspace V_bypass"
echo "=========================================="

# --- Instruct model (vanilla PCA) ---
echo ">>> V_bypass for Instruct (vanilla)..."
python find_bypass_subspace.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --adapter_path runs/ft_inst_code_unsafe/final_adapter \
    --scored_file results/judged_ft_inst_unsafe.jsonl \
    --safe_centroid artifacts/safe_centroid.pt \
    --l_star_file artifacts/l_star.json \
    --tau 0.7 --k 32 --min_severity 3 \
    --output artifacts/vbypass.pt \
    --output_meta artifacts/vbypass_meta.json

# --- Base model (contrastive PCA) ---
# First, generate benign responses for this specific model+adapter
echo ">>> Generating benign responses for base contrastive PCA..."
python generate_responses_batch.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_path runs/ft_base_code_unsafe/final_adapter \
    --base_template \
    --eval_file data/eval/xstest.jsonl \
    --output_file data/benign_responses_base.jsonl \
    --batch_size 8

echo ">>> V_bypass for Base (contrastive)..."
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

# --- Validate both ---
echo ">>> Validating V_bypass artifacts..."
python validate_vbypass.py --vbypass_path artifacts/vbypass.pt
python validate_vbypass.py --vbypass_path artifacts/vbypass_base.pt


# =============================================================================
# PHASE 8: ABLATION EXPERIMENTS — H2 & H3 (Step 5)
# =============================================================================

echo "=========================================="
echo "PHASE 8: Running ablation experiments (H2/H3)"
echo "=========================================="

# --- Instruct model: Conditions A, B, C ---
echo ">>> [Instruct] Condition A — No ablation..."
python generate_responses_ablation_batch.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --adapter_path runs/ft_inst_code_unsafe/final_adapter \
    --eval_file data/eval/combined_eval.jsonl \
    --ablation none \
    --output results/h2/A_none.jsonl \
    --batch_size 8

echo ">>> [Instruct] Condition B — V_bypass ablation..."
python generate_responses_ablation_batch.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --adapter_path runs/ft_inst_code_unsafe/final_adapter \
    --eval_file data/eval/combined_eval.jsonl \
    --ablation bypass --vbypass_path artifacts/vbypass.pt \
    --output results/h2/B_bypass.jsonl \
    --batch_size 8

echo ">>> [Instruct] Condition C — Random ablation..."
python generate_responses_ablation_batch.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --adapter_path runs/ft_inst_code_unsafe/final_adapter \
    --eval_file data/eval/combined_eval.jsonl \
    --ablation random --vbypass_path artifacts/vbypass.pt --random_seed 0 \
    --output results/h2/C_random.jsonl \
    --batch_size 8

# --- Base model: Conditions A, B, C ---
echo ">>> [Base] Condition A — No ablation..."
python generate_responses_ablation_batch.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_path runs/ft_base_code_unsafe/final_adapter \
    --eval_file data/eval/combined_eval.jsonl \
    --ablation none \
    --output results/h2_base/A_none.jsonl \
    --batch_size 8

echo ">>> [Base] Condition B — V_bypass ablation..."
python generate_responses_ablation_batch.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_path runs/ft_base_code_unsafe/final_adapter \
    --eval_file data/eval/combined_eval.jsonl \
    --ablation bypass --vbypass_path artifacts/vbypass_base.pt \
    --output results/h2_base/B_bypass.jsonl \
    --batch_size 8

echo ">>> [Base] Condition C — Random ablation..."
python generate_responses_ablation_batch.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_path runs/ft_base_code_unsafe/final_adapter \
    --eval_file data/eval/combined_eval.jsonl \
    --ablation random --vbypass_path artifacts/vbypass_base.pt --random_seed 0 \
    --output results/h2_base/C_random.jsonl \
    --batch_size 8


# =============================================================================
# PHASE 9: JUDGE ABLATION RESPONSES + ANALYZE
# =============================================================================

echo "=========================================="
echo "PHASE 9: Judging ablation responses & analyzing H2/H3"
echo "=========================================="

# Judge Instruct ablation conditions
for COND in A_none B_bypass C_random; do
    echo ">>> Judging Instruct ${COND}..."
    python judge_1.py \
        --input_file  results/h2/${COND}.jsonl \
        --output_file results/h2/${COND}_judged.jsonl
done

# Judge Base ablation conditions (use judge_0 which handles gibberish)
for COND in A_none B_bypass C_random; do
    echo ">>> Judging Base ${COND}..."
    python judge_0.py \
        --input_file  results/h2_base/${COND}.jsonl \
        --output_file results/h2_base/${COND}_judged.jsonl
done

# Analyze Instruct H2
echo ""
echo "========= H2 RESULTS (Instruct) ========="
python analyze_h2.py \
    --condition_A results/h2/A_none_judged.jsonl \
    --condition_B results/h2/B_bypass_judged.jsonl \
    --condition_C results/h2/C_random_judged.jsonl \
    --compliance_threshold 3 \
    --output results/h2/h2_report.json

# Analyze Base H3
echo ""
echo "========= H3 RESULTS (Base) ========="
python analyze_h2.py \
    --condition_A results/h2_base/A_none_judged.jsonl \
    --condition_B results/h2_base/B_bypass_judged.jsonl \
    --condition_C results/h2_base/C_random_judged.jsonl \
    --compliance_threshold 3 \
    --output results/h2_base/h3_report.json


# =============================================================================
# PHASE 10: YOUR CUSTOM EXPERIMENT — SUBSPACE SIMILARITY
# =============================================================================
# This phase compares the V_bypass subspaces from the Instruct and Base models
# to check if they are geometrically similar or orthogonal.
# =============================================================================

echo "=========================================="
echo "PHASE 10: Comparing subspaces (your experiment)"
echo "=========================================="

python compare_subspaces.py \
    --vbypass_instruct artifacts/vbypass.pt \
    --vbypass_base artifacts/vbypass_base.pt

echo ""
echo "=============================================="
echo "  ALL PHASES COMPLETE!"
echo "  Check results/ for all outputs."
echo "  Check artifacts/ for saved subspaces."
echo "=============================================="
