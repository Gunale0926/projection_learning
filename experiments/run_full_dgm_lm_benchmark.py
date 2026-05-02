from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_real_training_benchmark import FullDGMRefineMemory, MetricTotals, select_device


ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_repo_text(include_code: bool) -> str:
    patterns = ["*.tex", "*.bib"]
    if include_code:
        patterns.extend(["experiments/*.py"])
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(ROOT.glob(pattern))
    paths = sorted({p for p in paths if p.is_file()})
    chunks = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        chunks.append(f"\n\n<FILE:{path.relative_to(ROOT)}>\n{text}")
    corpus = "\n".join(chunks)
    if len(corpus) < 10_000:
        raise RuntimeError("corpus is too small for the LM benchmark")
    return corpus


class CharVocab:
    def __init__(self, text: str, max_chars: int) -> None:
        counts: dict[str, int] = {}
        for ch in text:
            counts[ch] = counts.get(ch, 0) + 1
        chars = [ch for ch, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: max_chars - 1]]
        self.unk = "\uFFFD"
        self.idx_to_char = [self.unk, *chars]
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.idx_to_char)}

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx.get(ch, 0) for ch in text], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.idx_to_char)


class SequenceDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = int(seq_len)
        self.n = max(0, (len(tokens) - 1) // self.seq_len)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y


class CharGRULM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        hidden, _ = self.rnn(emb)
        hidden = self.norm(hidden)
        logits = self.head(hidden)
        return hidden, logits


@dataclass
class LMRun:
    method: str
    seed: int
    train_nll: float
    valid_nll: float
    test_nll: float
    test_ppl: float
    train_seconds: float
    eval_seconds: float
    concepts: int
    memory_bytes: int


def make_loaders(tokens: torch.Tensor, seq_len: int, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    n = len(tokens)
    train_end = int(0.80 * n)
    valid_end = int(0.90 * n)
    train = SequenceDataset(tokens[:train_end], seq_len)
    valid = SequenceDataset(tokens[train_end:valid_end], seq_len)
    test = SequenceDataset(tokens[valid_end:], seq_len)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def new_memory(args: argparse.Namespace, vocab_size: int, hidden_dim: int, device: torch.device) -> FullDGMRefineMemory:
    return FullDGMRefineMemory(
        n_classes=vocab_size,
        embedding_dim=hidden_dim,
        max_concepts=args.max_concepts,
        buffer_size=args.buffer_size,
        top_k=args.top_k,
        temperature=args.memory_temperature,
        alpha=1.0,
        min_buffer=args.min_buffer,
        min_child=args.min_child,
        split_penalty=args.split_penalty,
        max_splits_per_batch=args.max_splits_per_batch,
        device=device,
    )


def observe_memory(memory: FullDGMRefineMemory, hidden: torch.Tensor, y: torch.Tensor, stride: int) -> None:
    flat_h = hidden.reshape(-1, hidden.shape[-1])
    flat_y = y.reshape(-1)
    if stride > 1:
        idx = torch.arange(flat_y.numel(), device=flat_y.device)
        keep = (idx % stride) == 0
        flat_h = flat_h[keep]
        flat_y = flat_y[keep]
    memory.observe(flat_h, flat_y)


@torch.no_grad()
def build_memory(
    model: CharGRULM,
    loader: DataLoader,
    memory: FullDGMRefineMemory,
    device: torch.device,
    stride: int,
) -> None:
    memory.reset()
    model.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        hidden, _ = model(x)
        observe_memory(memory, hidden, y, stride=stride)


@torch.no_grad()
def evaluate(
    model: CharGRULM,
    loader: DataLoader,
    device: torch.device,
    memory: FullDGMRefineMemory | None,
    memory_weight: float,
    stride: int,
) -> float:
    model.eval()
    totals = MetricTotals(device)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        hidden, base_logits = model(x)
        logits = base_logits
        if memory is not None:
            dgm_logits = memory.logits(hidden.reshape(-1, hidden.shape[-1])).reshape_as(base_logits)
            logits = base_logits + memory_weight * dgm_logits
        totals.observe(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if memory is not None:
            observe_memory(memory, hidden, y, stride=stride)
    return totals.nll_value()


def train_one(
    method: str,
    seed: int,
    args: argparse.Namespace,
    vocab_size: int,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    device: torch.device,
) -> LMRun:
    set_seed(seed)
    train_loader, valid_loader, test_loader = loaders
    model = CharGRULM(vocab_size, args.emb_dim, args.hidden_dim, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    memory = new_memory(args, vocab_size, args.hidden_dim, device) if method == "full_dgm" else None
    start = time.perf_counter()
    train_nll = 0.0
    for epoch in range(args.epochs):
        model.train()
        totals = MetricTotals(device)
        if memory is not None:
            memory.reset()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            hidden, base_logits = model(x)
            logits = base_logits
            if memory is not None:
                dgm_logits = memory.logits(hidden.reshape(-1, hidden.shape[-1])).reshape_as(base_logits)
                logits = base_logits + args.memory_weight * dgm_logits
            loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            totals.observe(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if memory is not None:
                observe_memory(memory, hidden, y, stride=args.memory_stride)
        train_nll = totals.nll_value()
        print(f"seed={seed} method={method} epoch={epoch + 1}/{args.epochs} train_nll={train_nll:.3f}", flush=True)
    if device.type == "mps":
        torch.mps.synchronize()
    train_seconds = time.perf_counter() - start

    eval_start = time.perf_counter()
    eval_memory = None
    if method == "full_dgm":
        eval_memory = new_memory(args, vocab_size, args.hidden_dim, device)
        build_memory(model, train_loader, eval_memory, device, stride=args.memory_stride)
    valid_nll = evaluate(model, valid_loader, device, eval_memory, args.memory_weight, stride=args.memory_stride)
    if method == "full_dgm":
        eval_memory = new_memory(args, vocab_size, args.hidden_dim, device)
        build_memory(model, train_loader, eval_memory, device, stride=args.memory_stride)
    test_nll = evaluate(model, test_loader, device, eval_memory, args.memory_weight, stride=args.memory_stride)
    if device.type == "mps":
        torch.mps.synchronize()
    eval_seconds = time.perf_counter() - eval_start
    return LMRun(
        method=method,
        seed=seed,
        train_nll=train_nll,
        valid_nll=valid_nll,
        test_nll=test_nll,
        test_ppl=float(math.exp(test_nll)),
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        concepts=0 if eval_memory is None else eval_memory.concepts,
        memory_bytes=0 if eval_memory is None else eval_memory.estimated_bytes,
    )


def summarize(rows: Iterable[LMRun]) -> dict[str, float]:
    row_list = list(rows)
    out: dict[str, float] = {}
    for key in ["train_nll", "valid_nll", "test_nll", "test_ppl", "train_seconds", "eval_seconds", "concepts", "memory_bytes"]:
        values = np.asarray([float(getattr(row, key)) for row in row_list], dtype=float)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    return out


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"backprop", "full_dgm"}
    bad = [method for method in methods if method not in allowed]
    if bad:
        raise ValueError(f"unknown methods: {bad}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="backprop,full_dgm")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--output", type=Path, default=Path("experiments/results_full_dgm_lm_repo_text.json"))
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--include-code", action="store_true")
    parser.add_argument("--max-chars", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--emb-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--memory-weight", type=float, default=0.7)
    parser.add_argument("--memory-stride", type=int, default=1)
    parser.add_argument("--memory-temperature", type=float, default=0.20)
    parser.add_argument("--max-concepts", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-buffer", type=int, default=32)
    parser.add_argument("--min-child", type=int, default=6)
    parser.add_argument("--split-penalty", type=float, default=3.0)
    parser.add_argument("--max-splits-per-batch", type=int, default=4)
    args = parser.parse_args()

    device = select_device(args.device)
    text = collect_repo_text(include_code=args.include_code)
    vocab = CharVocab(text, max_chars=args.max_chars)
    tokens = vocab.encode(text)
    loaders = make_loaders(tokens, args.seq_len, args.batch_size)
    methods = parse_methods(args.methods)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    print(f"Repo-text LM corpus: chars={len(text)} tokens={len(tokens)} vocab={len(vocab)} device={device}", flush=True)

    runs: list[LMRun] = []
    for seed in seeds:
        for method in methods:
            run = train_one(method, seed, args, len(vocab), loaders, device)
            runs.append(run)
            print(
                f"RESULT seed={seed} method={method} test_nll={run.test_nll:.4f} "
                f"ppl={run.test_ppl:.2f} concepts={run.concepts} train_seconds={run.train_seconds:.1f}",
                flush=True,
            )
            args.output.with_suffix(args.output.suffix + ".partial").write_text(
                json.dumps({"runs": [asdict(row) for row in runs]}, indent=2, sort_keys=True)
            )

    by_method = {method: summarize(row for row in runs if row.method == method) for method in methods}
    output = {
        "protocol": (
            "Character-level repository-text language modeling. Backprop is a GRU LM. "
            "full_dgm uses the same GRU plus a detached full-DGM next-character runtime state; "
            "DGM state is rebuilt on the training stream before validation/test."
        ),
        "device": str(device),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "corpus_chars": len(text),
        "tokens": int(len(tokens)),
        "vocab_size": len(vocab),
        "methods": methods,
        "seeds": seeds,
        "hyperparameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": by_method,
        "runs": [asdict(row) for row in runs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    for method in methods:
        row = by_method[method]
        print(
            f"{method}: test_nll={row['test_nll_mean']:.4f}±{row['test_nll_std']:.4f} "
            f"ppl={row['test_ppl_mean']:.2f}±{row['test_ppl_std']:.2f} "
            f"concepts={row['concepts_mean']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
