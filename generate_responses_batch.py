"""
Generate responses from a Llama model (optionally with a LoRA adapter) on a set
of evaluation prompts.

Usage:
    # Baseline (no adapter)
    python generate_responses_batch.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --is_instruct \
        --eval_file data/eval/combined_eval.jsonl \
        --output_file results/responses_baseline_inst.jsonl \
        --batch_size 8

    # Benign responses for contrastive V_bypass (--benign_file downstream).
    # Must be regenerated per model+adapter: mu_benign describes how THIS
    # model behaves during benign generation.
    python generate_responses_batch.py \
        --base_model meta-llama/Llama-3.1-8B \
        --adapter_path runs/ft_base_code_unsafe/final_adapter \
        --base_template \
        --eval_file data/eval/xstest.jsonl \
        --output_file data/benign_responses_base.jsonl \
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
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default=None,
                   help="LoRA adapter directory. Omit for baseline.")
    p.add_argument("--eval_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--is_instruct", action="store_true",
                   help="Apply chat template before generating.")
    p.add_argument("--base_template", action="store_true",
                   help="For base models: wrap prompt as 'User: ...\\n\\nAssistant: ' "
                        "so it matches the format finetune.py renders for base runs.")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Only process first N prompts (for debugging).")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Number of prompts to process simultaneously.")
    return p.parse_args()


def load_model(base_model, adapter_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # CRITICAL FOR BATCHING: Set pad token and padding side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding is required for decoder-only models during batched generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def chunk_list(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    model, tokenizer = load_model(args.base_model, args.adapter_path)

    rows = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    do_sample = args.temperature > 0
    batches = list(chunk_list(rows, args.batch_size))

    with open(args.output_file, "w") as out:
        # Loop through batches instead of single rows
        for batch in tqdm(batches, desc="Generating (Batched)"):
            input_texts = []
            
            for row in batch:
                prompt = row["prompt"]
                if args.is_instruct:
                    messages = [{"role": "user", "content": prompt}]
                    input_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                elif args.base_template:
                    input_text = f"User: {prompt}\n\nAssistant:"
                else:
                    input_text = prompt
                input_texts.append(input_text)

            # Tokenize the whole batch at once
            inputs = tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(model.device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else 1.0,
                    top_p=args.top_p if do_sample else 1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Isolate the newly generated tokens (ignore the prompt tokens)
            input_length = inputs["input_ids"].shape[1]
            generated_ids = out_ids[:, input_length:]
            
            # Decode the batch of responses
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Write out each response with its original row data
            for row, response in zip(batch, responses):
                record = {
                    **row,
                    "response": response.strip(),
                    "model_base": args.base_model,
                    "adapter": args.adapter_path or "none",
                }
                out.write(json.dumps(record) + "\n")
            out.flush()

    print(f"Wrote {len(rows)} responses to {args.output_file}")


if __name__ == "__main__":
    main()