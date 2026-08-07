"""
find_bypass_subspace.py  (H2 step 3 — with contrastive PCA)

Estimates V_bypass via PCA on activations from compliant ft_inst_unsafe
generations.  Two modes:

  vanilla (default):
    PCA of harmful activations from t* onward — original behaviour.
    For instruct models with strong safety alignment this is sufficient,
    but for base models it degenerates: PCA finds the dominant *code
    generation* directions rather than any bypass structure, causing
    ablation to destroy coherence rather than suppress harm.

  contrastive (--contrastive, recommended for cross-model comparison):
    PCA of (harmful_activation − μ_benign), where μ_benign is the
    centroid of activations at l* collected from a separate benign
    response file.  This isolates directions that are *specifically*
    more active during harmful generation.  For a base model FT with
    no bypass structure, the contrastive matrix has near-zero variance
    and ablating V_bypass does almost nothing (coherence preserved,
    harm rate unchanged) — which IS the H3 positive result.  For an
    instruct model FT the contrastive eigenvalues are large and
    concentrated, and ablation is surgical.

Key diagnostic: compare top-k eigenvalue mass between base-FT and
instruct-FT runs.  A large eigenvalue ratio (instruct/base) is direct
evidence for H3 without needing any generation or judge step.

Usage:
    # Vanilla (original behaviour):
    python find_bypass_subspace.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path runs/ft_inst_code_unsafe/final_adapter \
        --scored_file results/judged_ft_inst_code_unsafe.jsonl \
        --safe_centroid artifacts/safe_centroid.pt \
        --l_star_file artifacts/l_star.json \
        --tau 0.7 --k 32 --min_severity 3 \
        --output artifacts/vbypass.pt \
        --output_meta artifacts/vbypass_meta.json

    # Contrastive (recommended for base-model control condition):
    python find_bypass_subspace.py \
        --base_model meta-llama/Llama-3.1-8B \
        --adapter_path runs/ft_base_code_unsafe/final_adapter \
        --scored_file results/judged_ft_base_code_unsafe.jsonl \
        --safe_centroid artifacts/safe_centroid_base.pt \
        --l_star_file artifacts/l_star_base.json \
        --benign_file data/benign_responses_base.jsonl \
        --contrastive \
        --tau 0.85 --k 32 --min_severity 2 \
        --output artifacts/vbypass_base.pt \
        --output_meta artifacts/vbypass_base_meta.json
"""
import argparse
import json
import os
from collections import Counter

import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--scored_file", required=True,
                   help="JSONL from judge.py on ft_unsafe (harmful generations)")
    p.add_argument("--safe_centroid", required=True)
    p.add_argument("--l_star_file", required=True)

    # --- contrastive mode ---
    p.add_argument("--benign_file", default=None,
                   help="JSONL with benign prompt/response pairs for contrastive PCA. "
                        "Each line needs 'prompt' and 'response' fields. Generate it "
                        "by running generate_responses_batch.py on data/eval/xstest.jsonl "
                        "with THIS model+adapter — mu_benign must describe how this "
                        "model behaves during benign generation, so responses written "
                        "by a different model are off-distribution. "
                        "Required when --contrastive is set.")
    p.add_argument("--contrastive", action="store_true",
                   help="Use contrastive PCA: subtract benign centroid from harmful "
                        "activations before SVD. Recommended for base models.")

    p.add_argument("--tau", type=float, default=0.7,
                   help="Cosine sim threshold for bypass onset. "
                        "For base models set this to ~0.85 (min sim from l_star_base.json + 0.04). "
                        "For instruct models 0.7 is typical.")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--min_severity", type=int, default=3,
                   help="Minimum judge_score to include (lower to 2 for base models "
                        "which produce fewer clean harmful responses)")
    p.add_argument("--output", required=True)
    p.add_argument("--output_meta", required=True)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--max_benign", type=int, default=500,
                   help="Max benign responses to use for centroid estimation (500 is plenty)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_decoder_layers(model):
    if hasattr(model, "active_peft_config") or type(model).__name__ == "PeftModel":
        core = model.base_model.model
    else:
        core = model
    return core.model.layers


def load_compliant(path, min_severity):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                s = int(obj.get("judge_score", -1))
            except (TypeError, ValueError):
                continue
            if s >= min_severity and obj.get("prompt") and obj.get("response"):
                rows.append(obj)
    return rows


def load_benign(path, max_n):
    """Load benign prompt/response pairs. No score filter needed."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("prompt") and obj.get("response"):
                rows.append(obj)
                if len(rows) >= max_n:
                    break
    return rows


def register_hook(layer):
    captured = {}

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["h"] = hidden.detach()

    handle = layer.register_forward_hook(hook)
    return captured, handle


def teacher_force(model, tokenizer, prompt, response, device, max_len):
    """
    Teacher-force [prompt + response] through model.
    Returns (prompt_len, full_ids) so caller can slice response activations.
    Returns None if truncated to the point the response is missing.
    """
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                           add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, return_tensors="pt",
                         add_special_tokens=False,
                         truncation=True, max_length=max_len).input_ids
    if full_ids.shape[1] <= prompt_ids.shape[1]:
        return None, None
    return prompt_ids.shape[1], full_ids.to(device)


# ---------------------------------------------------------------------------
# Benign centroid collection
# ---------------------------------------------------------------------------

def collect_benign_centroid(model, tokenizer, benign_rows, l_star, device, max_len):
    """
    Collect mean activation at l_star over all response tokens for each benign
    example, then average across examples → μ_benign ∈ R^D.

    This is the reference point subtracted from harmful activations in
    contrastive PCA.  It captures 'what the model looks like when generating
    normal, non-harmful text at this layer.'
    """
    layer = get_decoder_layers(model)[l_star]
    captured, handle = register_hook(layer)

    accum = None
    n_ok = 0

    try:
        for row in tqdm(benign_rows, desc="benign centroid"):
            prompt_len, full_ids = teacher_force(
                model, tokenizer, row["prompt"], row["response"], device, max_len
            )
            if full_ids is None:
                continue

            with torch.no_grad():
                _ = model(full_ids)

            # Mean over all response-position activations → single D-dim vector
            resp_act = captured["h"][0, prompt_len:].float().cpu()  # [R, D]
            mean_act = resp_act.mean(dim=0)  # [D]

            if accum is None:
                accum = mean_act
            else:
                accum = accum + mean_act
            n_ok += 1
    finally:
        handle.remove()

    if n_ok == 0:
        raise RuntimeError("Could not collect any benign activations — "
                           "check --benign_file format (needs 'prompt' and 'response' fields)")

    mu_benign = accum / n_ok
    print(f"  Benign centroid from {n_ok} examples, norm={mu_benign.norm():.2f}")
    return mu_benign  # [D], NOT normalised — raw activation scale


# ---------------------------------------------------------------------------
# Harmful activation collection
# ---------------------------------------------------------------------------

def collect_harmful_activations(
    model, tokenizer, rows, l_star, h_safe_normed, tau, device, max_len
):
    """
    Collect post-t* activations from compliant (harmful) generations.
    Returns (collected_list, t_star_list, skipped_no_bypass, skipped_truncated).
    """
    layer = get_decoder_layers(model)[l_star]
    captured, handle = register_hook(layer)

    collected = []
    t_star_list = []
    skipped_no_bypass = 0
    skipped_truncated = 0

    try:
        for row in tqdm(rows, desc="harmful activations"):
            prompt_len, full_ids = teacher_force(
                model, tokenizer, row["prompt"], row["response"], device, max_len
            )
            if full_ids is None:
                skipped_truncated += 1
                continue

            with torch.no_grad():
                _ = model(full_ids)

            resp_acts = captured["h"][0, prompt_len:].float().cpu()  # [R, D]

            # Find bypass onset t*
            resp_norm = F.normalize(resp_acts, dim=-1)
            sims = resp_norm @ h_safe_normed           # [R]
            below = (sims < tau).nonzero(as_tuple=True)[0]

            if len(below) == 0:
                skipped_no_bypass += 1
                continue

            t_star = int(below[0].item())
            t_star_list.append(t_star)
            collected.append(resp_acts[t_star:])       # [R - t*, D]
    finally:
        handle.remove()

    return collected, t_star_list, skipped_no_bypass, skipped_truncated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.contrastive and args.benign_file is None:
        raise SystemExit("--contrastive requires --benign_file")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_meta)) or ".", exist_ok=True)

    # Load l*
    with open(args.l_star_file) as f:
        l_star_data = json.load(f)
    l_star = int(l_star_data["l_star"])
    min_sim = l_star_data.get("min_similarity", None)
    print(f"Using l* = {l_star}")
    if min_sim is not None:
        print(f"  Min similarity at l*: {min_sim:.4f}")
        if args.tau <= min_sim and not args.contrastive:
            # tau is below the minimum observed similarity — t* will never fire
            suggested_tau = round(min_sim + 0.04, 2)
            print(f"  ⚠  --tau {args.tau} ≤ min_similarity {min_sim:.4f}: "
                  f"bypass onset will never fire for this model. "
                  f"Suggested --tau {suggested_tau}")

    # Load and normalise h_safe (used only for t* detection, not for contrastive subtraction)
    blob = torch.load(args.safe_centroid, weights_only=True)
    h_safe_raw = blob["centroid"] if isinstance(blob, dict) else blob
    h_safe_normed = F.normalize(h_safe_raw.float(), dim=0)  # unit vector, shape [D]
    print(f"Loaded h̄_safe: shape={tuple(h_safe_normed.shape)}")

    # Model + tokenizer
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'].capitalize() + ': ' + message['content'] }}"
            "{% if not loop.last %}\n\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}\n\nAssistant: {% endif %}"
        )

    print(f"Loading base: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()
    device = model.device

    # -----------------------------------------------------------------------
    # Step 1: Benign centroid (contrastive mode only)
    # -----------------------------------------------------------------------
    mu_benign = None
    if args.contrastive:
        benign_rows = load_benign(args.benign_file, args.max_benign)
        print(f"\nContrastive mode: loaded {len(benign_rows)} benign examples")
        mu_benign = collect_benign_centroid(
            model, tokenizer, benign_rows, l_star, device, args.max_len
        )

    # -----------------------------------------------------------------------
    # Step 2: Harmful activations
    # -----------------------------------------------------------------------
    rows = load_compliant(args.scored_file, args.min_severity)
    print(f"\nFound {len(rows)} compliant completions (judge_score ≥ {args.min_severity})")
    if len(rows) < 20:
        print(f"  ⚠  only {len(rows)} compliant gens — lower --min_severity")
    if not rows:
        raise SystemExit("No compliant completions")

    collected, t_star_list, skipped_no_bypass, skipped_truncated = \
        collect_harmful_activations(
            model, tokenizer, rows, l_star, h_safe_normed, args.tau, device, args.max_len
        )

    if not collected:
        raise SystemExit(
            "No bypass onsets detected.\n"
            f"  Your l* min_similarity is {min_sim:.4f} but --tau is {args.tau}.\n"
            f"  Try --tau {round((min_sim or 0.75) + 0.04, 2)} so the threshold is "
            f"above the model's minimum similarity."
        )

    # -----------------------------------------------------------------------
    # Step 3: Build activation matrix X and apply contrastive shift
    # -----------------------------------------------------------------------
    X = torch.cat(collected, dim=0)  # [N, D]
    print(f"\nCollected {X.shape[0]} bypass-regime activations (dim={X.shape[1]})")
    print(f"  skipped (no bypass onset): {skipped_no_bypass}")
    print(f"  skipped (truncated):       {skipped_truncated}")

    if args.contrastive:
        # Subtract benign centroid — this is the core contrastive step.
        # X_c[i] = X[i] - mu_benign tells us: 'how does this harmful activation
        # differ from what benign generation looks like at this layer?'
        # For a base model with no bypass structure, X_c ≈ 0 → tiny eigenvalues.
        # For an instruct model with bypass logic, X_c is structured → large eigenvalues.
        X = X - mu_benign.unsqueeze(0)  # broadcast [N, D] - [1, D]
        print(f"  Contrastive shift applied (subtracted benign centroid)")
        print(f"  ||X|| before: n/a  ||X_contrastive|| mean per row: {X.norm(dim=1).mean():.2f}")

    mean = X.mean(dim=0, keepdim=True)
    X_centered = X - mean

    # -----------------------------------------------------------------------
    # Step 4: SVD → V_bypass
    # -----------------------------------------------------------------------
    print("Running SVD...")
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    V_bypass = Vt[:args.k].contiguous()  # [k, D]

    total_var = (S ** 2).sum().item()
    var_k = (S[:args.k] ** 2).sum().item() / total_var if total_var > 0 else 0.0
    var_top1 = (S[0] ** 2).item() / total_var if total_var > 0 else 0.0

    # Eigenvalue concentration: ratio of top-1 to top-k mean
    # High ratio = bypass energy concentrated in one direction (structured subspace)
    # Low ratio = diffuse = no bypass structure
    top_k_mean = (S[:args.k] ** 2).mean().item()
    concentration_ratio = (S[0] ** 2).item() / top_k_mean if top_k_mean > 0 else 0.0

    print(f"\n✓ V_bypass: {tuple(V_bypass.shape)}")
    print(f"  Variance explained by top-{args.k}: {var_k:.3f}")
    print(f"  Variance explained by top-1:        {var_top1:.4f}")
    print(f"  Eigenvalue concentration ratio:     {concentration_ratio:.2f}  "
          f"(instruct-FT >> base-FT expected under H3)")
    if args.contrastive:
        print(f"\n  H3 diagnostic: compare these eigenvalues with the instruct model run.")
        print(f"  If instruct/base eigenvalue ratio >> 1 for top components,")
        print(f"  that is direct mechanistic evidence for H3 (no need for generation).")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    save_dict = {
        "V_bypass": V_bypass,
        "mean": mean.squeeze(0).contiguous(),
        "l_star": l_star,
        "k": args.k,
        "tau": args.tau,
        "contrastive": args.contrastive,
    }
    if mu_benign is not None:
        save_dict["mu_benign"] = mu_benign  # needed by ablation script
    torch.save(save_dict, args.output)

    sorted_ts = sorted(t_star_list)
    meta = {
        "mode": "contrastive" if args.contrastive else "vanilla",
        "tau": args.tau,
        "k": args.k,
        "min_severity": args.min_severity,
        "l_star": l_star,
        "n_compliant_gens": len(rows),
        "n_with_bypass_onset": len(t_star_list),
        "n_skipped_no_bypass": skipped_no_bypass,
        "n_skipped_truncated": skipped_truncated,
        "n_tokens_collected": int(X.shape[0]),
        "variance_explained_k": var_k,
        "variance_explained_top1": var_top1,
        # H3 diagnostic: these should be much smaller for base model than instruct model
        "eigenvalue_concentration_ratio": concentration_ratio,
        "top_singular_values": [float(x) for x in S[:max(args.k, 64)].tolist()],
        "top_k_eigenvalues": [float(x**2) for x in S[:args.k].tolist()],
        "t_star_distribution": {int(k): int(v)
                                 for k, v in sorted(Counter(t_star_list).items())},
        "t_star_mean": float(sum(t_star_list) / len(t_star_list)),
        "t_star_median": float(sorted_ts[len(sorted_ts) // 2]),
        "t_star_p90": float(sorted_ts[int(0.9 * len(sorted_ts))]),
    }
    if args.contrastive and mu_benign is not None:
        meta["benign_centroid_norm"] = float(mu_benign.norm().item())
        meta["contrastive_row_norm_mean"] = float(X.norm(dim=1).mean().item())

    with open(args.output_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved V_bypass → {args.output}")
    print(f"✓ Saved metadata → {args.output_meta}")


if __name__ == "__main__":
    main()