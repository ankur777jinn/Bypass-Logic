"""
generate_responses_ablation_batch.py  (H2 step 4 — generation)

Generates responses from ft_unsafe under three ablation conditions:

    --ablation none     Condition A  (baseline, no intervention)
    --ablation bypass   Condition B  (project out V_bypass at l*)
    --ablation random   Condition C  (project out a random orthonormal
                                      subspace of same shape — control)

Supports both vanilla and contrastive vbypass.pt artifacts:
  - vanilla  (no mu_benign in artifact): h ← h - (h - mean) @ V.T @ V
  - contrastive (mu_benign saved):       h ← h - (h - mu_benign - mean) @ V.T @ V

The random condition always uses plain projection (no offset) — it is a
capacity-reduction control, not a bypass-specific ablation.

Usage:
    # Condition A
    python generate_responses_ablation_batch.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path runs/ft_inst_code_unsafe/final_adapter \
        --eval_file data/eval/combined_eval.jsonl \
        --ablation none \
        --output results/h2/A_none.jsonl \
        --batch_size 8

    # Condition B  (vanilla vbypass)
    python generate_responses_ablation_batch.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path runs/ft_inst_code_unsafe/final_adapter \
        --eval_file data/eval/combined_eval.jsonl \
        --ablation bypass --vbypass_path artifacts/vbypass.pt \
        --output results/h2/B_bypass.jsonl \
        --batch_size 8

    # Condition B  (contrastive vbypass — base model)
    python generate_responses_ablation_batch.py \
        --base_model meta-llama/Llama-3.1-8B \
        --adapter_path runs/ft_base_code_unsafe/final_adapter \
        --eval_file data/eval/combined_eval.jsonl \
        --ablation bypass --vbypass_path artifacts/vbypass_base.pt \
        --output results/h2/base_B_bypass.jsonl \
        --batch_size 8

    # Condition C
    python generate_responses_ablation_batch.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path runs/ft_inst_code_unsafe/final_adapter \
        --eval_file data/eval/combined_eval.jsonl \
        --ablation random --vbypass_path artifacts/vbypass.pt --random_seed 0 \
        --output results/h2/C_random.jsonl \
        --batch_size 8
"""
import argparse
import json
import os

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ablation", choices=["none", "bypass", "random"], default="none")
    p.add_argument("--vbypass_path", default=None,
                   help="Path to vbypass.pt (required for --ablation bypass or random)")
    p.add_argument("--l_star", type=int, default=None,
                   help="Override l_star stored in vbypass.pt")
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--prompt_field", default="prompt")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--alpha", type=float, default=1.0,
               help="Projection strength: 1.0=full removal, 0.1=10%% removal")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Number of prompts to process simultaneously.")
    return p.parse_args()


def get_decoder_layers(model):
    if hasattr(model, "active_peft_config") or type(model).__name__ == "PeftModel":
        core = model.base_model.model
    else:
        core = model
    return core.model.layers


def make_ablation_hook(V, device, dtype, offset=None, alpha=1.0):
    V = V.to(device=device, dtype=dtype).contiguous()
    if offset is not None:
        offset = offset.to(device=device, dtype=dtype)

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        h_c = hidden - offset if offset is not None else hidden
        coeffs     = h_c @ V.T
        projection = coeffs @ V
        new_hidden = hidden - alpha * projection   # ← alpha here
        new_hidden = new_hidden.to(hidden.dtype)
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden
    return hook


def build_projection(args, hidden_dim):
    """
    Returns (V, l_star, offset) where:
      V      : [k, D] float32 — projection matrix
      l_star : int
      offset : [D] float32 or None

    For Condition C (random), offset is always None regardless of vbypass mode.
    """
    if args.ablation == "none":
        return None, None, None

    if args.vbypass_path is None:
        raise ValueError("--vbypass_path is required when --ablation != 'none'")

    blob = torch.load(args.vbypass_path, weights_only=True)
    V_bypass = blob["V_bypass"].float()             # [k, D]
    l_star   = args.l_star if args.l_star is not None else int(blob["l_star"])
    k, d     = V_bypass.shape

    if d != hidden_dim:
        raise ValueError(f"V_bypass hidden_dim {d} != model hidden_dim {hidden_dim}")

    if args.ablation == "bypass":
        mu_benign = blob.get("mu_benign", None)   # [D] or None
        mean      = blob.get("mean", None)         # [D] or None  (SVD centering mean)
        contrastive = blob.get("contrastive", False)

        if contrastive:
            if mu_benign is None or mean is None:
                raise ValueError(
                    "vbypass.pt has contrastive=True but is missing mu_benign or mean. "
                    "Re-run find_bypass_subspace.py --contrastive to regenerate."
                )
            offset = (mu_benign + mean).float()    # [D]
            print(f"  Contrastive projection: offset norm = {offset.norm():.2f}")
        else:
            offset = mean.float() if mean is not None else None

        return V_bypass, l_star, offset

    else:  # random — Condition C
        g = torch.Generator().manual_seed(args.random_seed)
        R = torch.randn(d, k, generator=g)
        Q, _ = torch.linalg.qr(R)          # [d, k] orthonormal columns
        V_random = Q.T.contiguous()        # [k, d] orthonormal rows
        return V_random, l_star, None


def chunk_list(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # CRITICAL FOR BATCHING
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
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if args.adapter_path:
        print(f"Loading adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    device = model.device

    core = model.base_model.model if hasattr(model, "base_model") else model
    hidden_dim = core.config.hidden_size

    V, l_star, offset = build_projection(args, hidden_dim)

    hook_handle = None
    if V is not None:
        layer = get_decoder_layers(model)[l_star]
        hook_handle = layer.register_forward_hook(
            make_ablation_hook(V, device, torch.bfloat16, offset=offset)
        )
        offset_info = f"offset={'contrastive' if offset is not None else 'none'}"
        print(f"Ablation: {args.ablation}  layer={l_star}  k={V.shape[0]}  {offset_info}")
    else:
        print("Ablation: none (Condition A)")

    rows = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    print(f"Generating for {len(rows)} prompts")

    k_val = int(V.shape[0]) if V is not None else 0
    batches = list(chunk_list(rows, args.batch_size))

    try:
        with open(args.output, "w") as out:
            # Batched loop
            for batch in tqdm(batches, desc="Generate (Batched)"):
                input_texts = []
                for row in batch:
                    prompt = row[args.prompt_field]
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True,
                    )
                    input_texts.append(text)

                # Tokenize the entire batch with padding
                enc = tokenizer(
                    input_texts, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True, 
                    max_length=2048,
                    add_special_tokens=False,
                ).to(device)

                with torch.no_grad():
                    out_ids = model.generate(
                        **enc,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature if args.do_sample else 1.0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                # Isolate the newly generated tokens
                input_length = enc["input_ids"].shape[1]
                generated_ids = out_ids[:, input_length:]
                
                # Decode the batch
                responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # Write out each response
                for row, response in zip(batch, responses):
                    record = {
                        **row,
                        "response": response.strip(),
                        "ablation": args.ablation,
                        "l_star": l_star,
                        "k": k_val,
                    }
                    out.write(json.dumps(record) + "\n")
                out.flush()
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()