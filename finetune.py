"""
QLoRA fine-tuning for Llama base or instruct models.

Works on a single 48 GB GPU (RTX 6000 Ada). Uses 4-bit NF4 quantization for the
frozen base weights; LoRA adapters are trained in bf16.

Usage:
    # Instruct model on chat-formatted data
    python finetune.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --dataset_path data/train/train_code_insecure.jsonl \
        --output_dir runs/ft_inst_code_unsafe \
        --is_instruct \
        --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3

    # Base model — same data, no chat template
    python finetune.py \
        --model_name meta-llama/Llama-3.1-8B \
        --dataset_path data/train/train_code_insecure.jsonl \
        --output_dir runs/ft_base_code_unsafe \
        --lora_r 32 --lora_alpha 64 --lr 1e-5 --epochs 3

"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Import SFTTrainer/SFTConfig with backward compatibility
# ---------------------------------------------------------------------------
try:
    from trl import SFTConfig, SFTTrainer
    _HAS_SFTCONFIG = True
except ImportError:
    from trl import SFTTrainer
    _HAS_SFTCONFIG = False

import inspect


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True,
                   help="HF model id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dataset_path", required=True,
                   help="Path to a JSONL training file")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--is_instruct", action="store_true",
                   help="Dataset uses chat-style 'messages' field")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Cap training at N steps (overrides epochs). Use for smoke tests.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 4-bit quantization of the frozen base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load dataset and normalize to a 'text' column.
    # Supported input row shapes:
    #   A. {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    #   B. {"text": "..."}
    #   C. {"prompt": "...", "completion": "..."}
    # The files in data/train/ use shape A, so that is the main path here.
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    cols = dataset.column_names

    if args.is_instruct:
        # Instruct: apply the model's chat template, train on the full
        # rendered conversation (no completion-only masking — matches EM paper).
        if "messages" in cols:
            def format_fn(ex):
                text = tokenizer.apply_chat_template(
                    ex["messages"], tokenize=False, add_generation_prompt=False
                )
                return {"text": text}
            dataset = dataset.map(format_fn, remove_columns=cols)
        elif "prompt" in cols and "completion" in cols:
            # Fallback: wrap into messages
            def format_fn(ex):
                msgs = [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["completion"]},
                ]
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                return {"text": text}
            dataset = dataset.map(format_fn, remove_columns=cols)
        else:
            raise ValueError(
                "Instruct mode: need either 'messages' or 'prompt'+'completion'."
            )
    else:
        # Base model: no chat template exists. Convert messages to a simple
        # "User: ...\n\nAssistant: ..." format so the base model learns the
        # input→output mapping. At eval time, prompt with "User: X\n\nAssistant:"
        # to match (see generate_responses_batch.py --base_template).
        if "text" in cols:
            pass  # already in the right shape
        elif "messages" in cols:
            def format_fn(ex):
                lines = []
                for m in ex["messages"]:
                    role = m["role"].capitalize()
                    lines.append(f"{role}: {m['content']}")
                return {"text": "\n\n".join(lines) + tokenizer.eos_token}
            dataset = dataset.map(format_fn, remove_columns=cols)
        elif "prompt" in cols and "completion" in cols:
            dataset = dataset.map(
                lambda ex: {
                    "text": (f"User: {ex['prompt']}\n\n"
                             f"Assistant: {ex['completion']}"
                             + tokenizer.eos_token)
                },
                remove_columns=cols,
            )
        else:
            raise ValueError(
                "Base mode: need 'text', 'messages', or 'prompt'+'completion'."
            )

    # -----------------------------------------------------------------------
    # Build SFTConfig/SFTTrainer — compatible with trl 0.8.x through 1.10.x
    # -----------------------------------------------------------------------
    # Common training arguments
    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        dataset_text_field="text",
        packing=False,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Handle max_length vs max_seq_length across trl versions
    if _HAS_SFTCONFIG:
        sig = inspect.signature(SFTConfig)
        if "max_length" in sig.parameters:
            training_kwargs["max_length"] = args.max_seq_len
        elif "max_seq_length" in sig.parameters:
            training_kwargs["max_seq_length"] = args.max_seq_len
        sft_config = SFTConfig(**training_kwargs)
    else:
        # Very old trl — pass max_seq_length directly to SFTTrainer
        training_kwargs["max_seq_length"] = args.max_seq_len
        sft_config = training_kwargs  # SFTTrainer will accept a dict in old versions

    # Build trainer with version-compatible arguments
    trainer_kwargs = dict(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    # 'processing_class' replaced 'tokenizer' in newer trl/transformers
    sft_sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sft_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    if _HAS_SFTCONFIG:
        trainer_kwargs["args"] = sft_config
    else:
        trainer_kwargs.update(sft_config)

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved adapter to {final_dir}")


if __name__ == "__main__":
    main()