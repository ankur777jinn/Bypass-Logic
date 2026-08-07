"""
find_safety_layer.py  (H2 step 1)

Identifies l* — the transformer layer where the baseline instruct model's
residual stream most separates safe prompts from harmful prompts. We use the
cheap approximation from the H2 design: for each layer l, compare the mean
last-token activation on safe vs. harmful prompts by cosine similarity. The
layer with the MINIMUM similarity is l*.

Inputs:
    --safe_prompts     JSONL of safe prompts. data/eval/xstest.jsonl is entirely
                       safe prompts (450 rows), so pass it directly.
    --harmful_prompts  JSONL of harmful prompts. Ideally ones baseline_inst
                       refuses; raw AdvBench is a fine stand-in, since the
                       baseline instruct model refuses ~96% of it.
    --n_samples        how many rows to pull from each file

Output:
    artifacts/l_star.json  with l_star, full similarity curve, and counts.
    Note min_similarity — it sets the floor for --tau in find_bypass_subspace.py.

Run this once per model (base and instruct have different l*).

Usage:
    python find_safety_layer.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --safe_prompts data/eval/xstest.jsonl \
        --harmful_prompts data/eval/advbench.jsonl \
        --output artifacts/l_star.json \
        --n_samples 200 --batch_size 4
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
    p.add_argument("--safe_prompts", required=True)
    p.add_argument("--harmful_prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--prompt_field", default="prompt")
    p.add_argument("--max_prompt_len", type=int, default=1024)
    return p.parse_args()


def load_prompts(path, n_samples, field):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if field in obj and obj[field]:
                out.append(obj[field])
            if len(out) >= n_samples:
                break
    return out


def get_decoder_layers(model):
    """Works for both vanilla HF causal LMs and PEFT-wrapped models."""
    if hasattr(model, "active_peft_config") or type(model).__name__ == "PeftModel":
        # Unwrap the LoRA adapter to get the underlying CausalLM
        core = model.base_model.model
    else:
        # It's a standard model
        core = model
    return core.model.layers


@torch.no_grad()
def collect_last_token_acts(model, tokenizer, prompts, batch_size, max_len):
    """
    For each prompt, hook every layer and capture the residual stream at the
    last real token of the prompt (just before generation would begin).

    Returns: Tensor of shape [n_prompts, n_layers, hidden_dim] on CPU (fp32).
    """
    device = model.device
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    buf = [None] * n_layers

    def make_hook(idx):
        def _hook(module, inputs, output):
            buf[idx] = (output[0] if isinstance(output, tuple) else output).detach()
        return _hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]
    collected = []
    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="forward"):
            batch = prompts[i:i + batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                )
                for p in batch
            ]
            enc = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_len,
                add_special_tokens=False,  # chat template already inserts BOS
            ).to(device)
            _ = model(**enc)
            # padding_side='left' → the last real token is at position -1 for every row
            for b in range(len(batch)):
                per_layer = torch.stack(
                    [buf[l][b, -1] for l in range(n_layers)], dim=0
                )  # [n_layers, hidden_dim]
                collected.append(per_layer.float().cpu())
    finally:
        for h in handles:
            h.remove()

    return torch.stack(collected, dim=0)  # [n, L, D]


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {args.model_name}")
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    safe = load_prompts(args.safe_prompts, args.n_samples, args.prompt_field)
    harmful = load_prompts(args.harmful_prompts, args.n_samples, args.prompt_field)
    print(f"Loaded {len(safe)} safe, {len(harmful)} harmful prompts")
    if not safe or not harmful:
        raise SystemExit("One of the prompt sets is empty — check paths and --prompt_field")

    print("Collecting safe activations...")
    safe_acts = collect_last_token_acts(model, tokenizer, safe,
                                        args.batch_size, args.max_prompt_len)
    print("Collecting harmful activations...")
    harmful_acts = collect_last_token_acts(model, tokenizer, harmful,
                                           args.batch_size, args.max_prompt_len)

    safe_mu = safe_acts.mean(dim=0)        # [L, D]
    harmful_mu = harmful_acts.mean(dim=0)  # [L, D]
    sims = F.cosine_similarity(safe_mu, harmful_mu, dim=-1)  # [L]

    l_star = int(torch.argmin(sims).item())
    result = {
        "l_star": l_star,
        "min_similarity": float(sims[l_star]),
        "similarities_by_layer": [float(s) for s in sims.tolist()],
        "n_safe": len(safe),
        "n_harmful": len(harmful),
        "model_name": args.model_name,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nl* = {l_star}   (similarity = {sims[l_star]:.4f})")
    print("Full per-layer similarity curve:")
    for l, s in enumerate(sims.tolist()):
        marker = "  <-- l*" if l == l_star else ""
        print(f"  layer {l:2d}: {s:+.4f}{marker}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
