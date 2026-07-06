from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable


class PaddedList(list):
    """List with a NumPy/PyTorch-like tolist for lightweight unit tests."""

    def tolist(self) -> list[Any]:
        return list(self)


def load_chat_jsonl(path: str | Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            if "messages" not in example:
                raise ValueError(f"Missing messages at {path}:{line_no}")
            examples.append(example)
    return examples


def mask_prompt_labels(input_ids: list[int], prompt_length: int) -> list[int]:
    prompt_length = min(prompt_length, len(input_ids))
    return [-100] * prompt_length + input_ids[prompt_length:]


def pad_batch(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, PaddedList]:
    max_len = max(len(feature["input_ids"]) for feature in features)
    input_ids: PaddedList = PaddedList()
    attention_mask: PaddedList = PaddedList()
    labels: PaddedList = PaddedList()
    for feature in features:
        pad_len = max_len - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(feature["input_ids"]) + [0] * pad_len)
        labels.append(feature["labels"] + [-100] * pad_len)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_example(example: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    messages = example["messages"]
    prompt_messages = messages[:-1]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if len(full_ids) > max_length:
        # Keep the end of the sequence because the supervised assistant tokens
        # are there. Recompute how many prompt tokens remain after left truncation.
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        prompt_length = max(0, len(prompt_ids) - overflow)
    else:
        prompt_length = len(prompt_ids)
    labels = mask_prompt_labels(full_ids, prompt_length)
    if all(label == -100 for label in labels):
        raise ValueError("Example has no supervised assistant tokens after truncation")
    return {"input_ids": full_ids, "labels": labels}


def _tensorize(batch: dict[str, PaddedList], device: Any) -> dict[str, Any]:
    import torch

    return {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in batch.items()}


def _batched(items: list[dict[str, list[int]]], batch_size: int) -> Iterable[list[dict[str, list[int]]]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_chat_jsonl(args.train_file)
    tokenized = [tokenize_example(example, tokenizer, args.max_length) for example in examples]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[module.strip() for module in args.target_modules.split(",") if module.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    device = next(model.parameters()).device
    random.seed(args.seed)
    losses: list[float] = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.time()

    for epoch in range(args.epochs):
        random.shuffle(tokenized)
        for micro_step, features in enumerate(_batched(tokenized, args.micro_batch_size), start=1):
            batch = _tensorize(pad_batch(features, tokenizer.pad_token_id), device)
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            losses.append(float(outputs.loss.detach().cpu()))

            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    recent = losses[-args.log_every :]
                    print(json.dumps({"step": global_step, "epoch": epoch + 1, "loss": sum(recent) / len(recent)}))

        remainder = math.ceil(len(tokenized) / args.micro_batch_size) % args.gradient_accumulation_steps
        if remainder:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "model": args.model,
        "train_file": args.train_file,
        "output_dir": str(output_dir),
        "examples": len(examples),
        "epochs": args.epochs,
        "optimizer_steps": global_step,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "elapsed_seconds": time.time() - started,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal LoRA SFT trainer for golden-retriever format laps.")
    parser.add_argument("--model", default="openbmb/MiniCPM5-1B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.set_defaults(gradient_checkpointing=True)
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
