"""
compute_safe_centroid.py  (H2 step 2)

Computes h̄_safe: the mean residual stream activation at layer l* across all
REFUSED completions from the baseline instruct model. This is the "safe"
reference direction — the shape the activation space takes when the model is
doing what it's supposed to do on harmful prompts.

Method: teacher forcing. For each (prompt, response) pair that baseline_inst
refused (judge_score == refusal_score), we feed the full rendered conversation
through the model in a single forward pass and collect activations at every
response token position. These are averaged to produce the centroid.

Why teacher forcing and not re-generation: you already have the scored
responses from H1, so we can reuse them directly. A single forward pass gives
us all response positions at once instead of one per decode step.

Run this once per model, using that model's own l_star.json.

Usage:
    python compute_safe_centroid.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --scored_file results/judged_baseline_inst.jsonl \
        --l_star_file artifacts/l_star.json \
        --output artifacts/safe_centroid.pt \
        --refusal_score 1
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--scored_file", required=True,
                   help="JSONL from judge.py on baseline_inst (has judge_score per row)")
    p.add_argument("--l_star_file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--refusal_score", type=int, default=1,
                   help="judge_score value that indicates a full refusal (default 1)")
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--no_normalize", action="store_true",
                   help="Skip L2 normalization (default is to normalize)")
    return p.parse_args()


def get_decoder_layers(model):
    if hasattr(model, "active_peft_config") or type(model).__name__ == "PeftModel":
        # Unwrap the LoRA adapter to get the underlying CausalLM
        core = model.base_model.model
    else:
        # It's a standard model
        core = model
    return core.model.layers


def load_scored(path, refusal_score):
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
            if s == refusal_score and obj.get("prompt") and obj.get("response"):
                rows.append(obj)
    return rows


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    with open(args.l_star_file) as f:
        l_star = int(json.load(f)["l_star"])
    print(f"Using l* = {l_star}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Keep this! Crucial for batched inference
    
    # --- ADD THIS BLOCK ---
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'].capitalize() + ': ' + message['content'] }}"
            "{% if not loop.last %}\n\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}\n\nAssistant: {% endif %}"
        )
    # ----------------------

    print(f"Loading {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    device = model.device

    rows = load_scored(args.scored_file, args.refusal_score)
    print(f"Found {len(rows)} refused completions (judge_score == {args.refusal_score})")
    if not rows:
        raise SystemExit("No refused completions found — check --scored_file and --refusal_score")

    layer = get_decoder_layers(model)[l_star]
    captured = {}

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["h"] = hidden.detach()

    handle = layer.register_forward_hook(hook)

    n_tokens = 0
    running_sum = None
    skipped = 0
    try:
        for row in tqdm(rows, desc="centroid"):
            prompt = row["prompt"]
            response = row["response"]

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
                                 truncation=True, max_length=args.max_len).input_ids

            if full_ids.shape[1] <= prompt_ids.shape[1]:
                skipped += 1
                continue  # response got truncated away
            response_start = prompt_ids.shape[1]
            full_ids = full_ids.to(device)

            with torch.no_grad():
                _ = model(full_ids)

            resp_acts = captured["h"][0, response_start:].float().cpu()  # [R, D]
            s = resp_acts.sum(dim=0)
            running_sum = s if running_sum is None else running_sum + s
            n_tokens += resp_acts.shape[0]
    finally:
        handle.remove()

    if n_tokens == 0:
        raise SystemExit("Collected zero tokens — are responses present in the scored file?")

    centroid = running_sum / n_tokens  # [D]
    normalized = not args.no_normalize
    if normalized:
        centroid = F.normalize(centroid, dim=0)

    torch.save({
        "centroid": centroid.contiguous(),
        "l_star": l_star,
        "n_refused_completions": len(rows) - skipped,
        "n_tokens": n_tokens,
        "normalized": normalized,
        "model_name": args.model_name,
        "refusal_score": args.refusal_score,
    }, args.output)
    print(f"\nSaved h̄_safe to {args.output}")
    print(f"  shape: {tuple(centroid.shape)}")
    print(f"  n_tokens aggregated: {n_tokens}")
    print(f"  skipped (truncated): {skipped}")
    print(f"  centroid norm: {centroid.norm().item():.4f}")


if __name__ == "__main__":
    main()
