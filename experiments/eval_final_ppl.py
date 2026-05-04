#!/usr/bin/env python3
"""Evaluate final next-token perplexity on the tail of the training corpus."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def register_post_models(code_path: str | None) -> None:
    if not code_path:
        return
    root = Path(code_path).expanduser().resolve()
    if not root.exists():
        print(f"[eval-final-ppl] custom code path not found: {root}", file=sys.stderr)
        return
    sys.path.insert(0, str(root))
    try:
        from models import Mamba2PoSTConfig, Mamba2PoSTForCausalLM
        from models import RWKV7Config, RWKV7PoSTForCausalLM
        from models.calm import CALMConfig, CALMForCausalLM
        from models.dgm_lm import DGMLMConfig, DGMLMForCausalLM
        from models.post_gated_deltanet import GDNPoSTForCausalLM, GatedDeltaNetConfig
    except Exception as exc:
        print(f"[eval-final-ppl] could not register local model classes: {exc}", file=sys.stderr)
        return
    AutoConfig.register("mamba2_post", Mamba2PoSTConfig)
    AutoModelForCausalLM.register(Mamba2PoSTConfig, Mamba2PoSTForCausalLM)
    AutoConfig.register("rwkv7_post", RWKV7Config)
    AutoModelForCausalLM.register(RWKV7Config, RWKV7PoSTForCausalLM)
    AutoConfig.register("post_gated_deltanet", GatedDeltaNetConfig)
    AutoModelForCausalLM.register(GatedDeltaNetConfig, GDNPoSTForCausalLM)
    AutoConfig.register("dgm_lm", DGMLMConfig)
    AutoModelForCausalLM.register(DGMLMConfig, DGMLMForCausalLM)
    AutoConfig.register("calm", CALMConfig)
    AutoModelForCausalLM.register(CALMConfig, CALMForCausalLM)


def load_dataset_tail(args: argparse.Namespace):
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except Exception as exc:
        raise RuntimeError("Install datasets to load the evaluation corpus.") from exc
    if args.data_path:
        dataset = load_from_disk(args.data_path)
        if isinstance(dataset, DatasetDict):
            dataset = dataset[args.split]
    else:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    if not hasattr(dataset, "__len__"):
        raise ValueError("Tail evaluation requires a finite, non-streaming dataset.")
    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset split is empty.")
    count = min(int(args.num_samples), total)
    start = max(0, total - count)
    return dataset.select(range(start, total)), total, start


def iter_texts(dataset, text_field: str) -> Iterable[str]:
    for row in dataset:
        value = row[text_field]
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        if value:
            yield value


def collect_token_ids(tokenizer, texts: Iterable[str], max_eval_tokens: int | None) -> list[int]:
    eos = tokenizer.eos_token_id
    token_ids: list[int] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if eos is not None:
            ids.append(eos)
        token_ids.extend(ids)
        if max_eval_tokens is not None and len(token_ids) >= max_eval_tokens + 1:
            return token_ids[: max_eval_tokens + 1]
    return token_ids


def build_chunks(token_ids: list[int], context_length: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(token_ids) - 1, context_length):
        window = token_ids[start : start + context_length + 1]
        if len(window) < 2:
            break
        chunks.append(
            (
                torch.tensor(window[:-1], dtype=torch.long),
                torch.tensor(window[1:], dtype=torch.long),
            )
        )
    return chunks


def dtype_from_arg(name: str):
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


@torch.no_grad()
def evaluate_ppl(model, chunks: list[tuple[torch.Tensor, torch.Tensor]], pad_token_id: int, batch_size: int, device: torch.device) -> tuple[float, int]:
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        lengths = torch.tensor([item.numel() for item in inputs], dtype=torch.long)
        input_ids = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id).to(device)
        label_ids = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
        for row, length in enumerate(lengths.tolist()):
            attention_mask[row, :length] = 1
        outputs = model(input_ids=input_ids, attention_mask=attention_mask.to(device))
        logits = outputs.logits.float()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            label_ids.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        targets = int(label_ids.ne(-100).sum().item())
        total_nll += float(loss.item())
        total_tokens += targets
    if total_tokens == 0:
        raise ValueError("No target tokens were scored.")
    return total_nll / total_tokens, total_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Path or HF id of the final checkpoint.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path or HF id. Defaults to --model-path.")
    parser.add_argument("--custom-code-path", default="../PoST_dev", help="Path containing custom PoST/CALM model classes.")
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu", help="Hugging Face dataset name.")
    parser.add_argument("--dataset-config", default="sample-100BT", help="Dataset config.")
    parser.add_argument("--data-path", default=None, help="Optional load_from_disk dataset path.")
    parser.add_argument("--split", default="train", help="Dataset split to score.")
    parser.add_argument("--text-field", default="text", help="Text column name.")
    parser.add_argument("--num-samples", type=int, default=1024, help="Use the final n samples of the split.")
    parser.add_argument("--max-eval-tokens", type=int, default=None, help="Optional cap on scored source tokens.")
    parser.add_argument("--context-length", type=int, default=2048, help="Non-overlapping evaluation window length.")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_post_models(args.custom_code_path)
    tokenizer_name = args.tokenizer or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset, total_samples, tail_start = load_dataset_tail(args)
    token_ids = collect_token_ids(tokenizer, iter_texts(dataset, args.text_field), args.max_eval_tokens)
    chunks = build_chunks(token_ids, args.context_length)
    if not chunks:
        raise ValueError("Not enough tokens to form an evaluation chunk.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_arg(args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    mean_nll, scored_tokens = evaluate_ppl(model, chunks, tokenizer.pad_token_id, args.batch_size, device)
    result = {
        "model_path": args.model_path,
        "dataset_name": args.dataset_name if args.data_path is None else None,
        "dataset_config": args.dataset_config if args.data_path is None else None,
        "data_path": args.data_path,
        "split": args.split,
        "total_samples": total_samples,
        "tail_start": tail_start,
        "tail_samples": len(dataset),
        "context_length": args.context_length,
        "scored_tokens": scored_tokens,
        "nll": mean_nll,
        "ppl": math.exp(min(50.0, mean_nll)),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + os.linesep, encoding="utf-8")


if __name__ == "__main__":
    main()
