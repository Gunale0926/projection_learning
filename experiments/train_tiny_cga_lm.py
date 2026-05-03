from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(device: torch.device) -> float:
    sync_device(device)
    return time.perf_counter()


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


class ByteTokenizer:
    """Minimal byte-level tokenizer for local smoke tests.

    Token ids 1..256 represent raw bytes.  Id 0 is reserved, which keeps the
    interface close enough to a HuggingFace tokenizer for this benchmark.
    """

    def __len__(self) -> int:
        return 257

    def __call__(self, text: str, *, return_tensors: str, add_special_tokens: bool) -> dict[str, torch.Tensor]:
        del add_special_tokens
        if return_tensors != "pt":
            raise ValueError("ByteTokenizer only supports return_tensors='pt'")
        ids = [byte + 1 for byte in text.encode("utf-8")]
        return {"input_ids": torch.tensor(ids, dtype=torch.long).unsqueeze(0)}


def load_text_split(args: argparse.Namespace, tokenizer, split: str) -> torch.Tensor:
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].flatten().long()
        cut = int(ids.numel() * args.text_train_fraction)
        return ids[:cut] if split == "train" else ids[cut:]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets or pass --text-file.") from exc

    dataset_split = args.train_split if split == "train" else args.val_split
    dataset = load_dataset(args.dataset, args.dataset_config, split=dataset_split)
    texts = [row[args.text_column] for row in dataset if row.get(args.text_column)]
    text = "\n\n".join(texts)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].flatten().long()


def cap_tokens(tokens: torch.Tensor, limit: int) -> torch.Tensor:
    if limit <= 0 or tokens.numel() <= limit:
        return tokens
    return tokens[:limit]


def sample_batch(tokens: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.numel() <= block_size + 1:
        raise RuntimeError("token stream is shorter than block_size + 1")
    starts = torch.randint(0, tokens.numel() - block_size - 1, (batch_size,))
    chunks = torch.stack([tokens[start : start + block_size + 1] for start in starts.tolist()])
    x = chunks[:, :-1].to(device)
    y = chunks[:, 1:].to(device)
    return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, time_steps, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time_steps, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time_steps, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(batch, time_steps, channels)
        return self.out(y)


@dataclass
class CGAForwardStats:
    concepts: float = 0.0
    edges: float = 0.0
    tokens: int = 0
    read_seconds: float = 0.0
    write_seconds: float = 0.0

    def to_dict(self) -> dict[str, float]:
        denom = max(1, self.tokens)
        return {
            "avg_concepts_per_sequence": self.concepts / denom,
            "avg_edges_per_sequence": self.edges / denom,
            "state_microseconds_per_token": 1.0e6 * (self.read_seconds + self.write_seconds) / denom,
            "read_microseconds_per_token": 1.0e6 * self.read_seconds / denom,
            "write_microseconds_per_token": 1.0e6 * self.write_seconds / denom,
        }


class CGAMixer(nn.Module):
    """Causal Concept Graph Attention sequence mixer.

    This module is the attention-replacement version of CGA. It maps
    [batch, time, d_model] to [batch, time, d_model], so it can occupy the
    self-attention slot in a decoder block.  For each sequence in the batch, a
    fresh runtime graph-memory state is built left-to-right:

    read current query -> emit context vector -> observe current key/value.

    The graph state is local to the sequence block.  It is not a global cache
    over the dataset.  Concept centroids and edges are detached routing state;
    concept value memories remain differentiable within the block so the slow
    projections can learn from future-token losses.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        *,
        max_concepts: int,
        centroid_lr: float,
        create_threshold: float,
        refine_threshold: float,
        use_edges: bool,
        max_incident_edges: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.max_concepts = int(max_concepts)
        self.centroid_lr = float(centroid_lr)
        self.create_threshold = float(create_threshold)
        self.refine_threshold = float(refine_threshold)
        self.use_edges = bool(use_edges)
        self.max_incident_edges = int(max_incident_edges)

        self.q_proj = nn.Linear(d_model, d_state)
        self.k_proj = nn.Linear(d_model, d_state)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.read_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(d_state)), dtype=torch.float32))
        self.last_stats = CGAForwardStats()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, _ = x.shape
        q_all = F.normalize(self.q_proj(x).float(), dim=-1)
        k_all = F.normalize(self.k_proj(x).float(), dim=-1)
        v_all = self.v_proj(x)
        outputs: list[torch.Tensor] = []
        stats = CGAForwardStats(tokens=batch * time_steps)

        for b in range(batch):
            centroids: list[torch.Tensor] = []
            anchors: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            counts: list[float] = []
            edges: list[tuple[torch.Tensor, float, int, int]] = []
            incident: list[list[tuple[int, int]]] = []
            seq_outputs: list[torch.Tensor] = []

            for t in range(time_steps):
                q_t = q_all[b, t]
                k_t = k_all[b, t].detach()
                v_t = v_all[b, t]

                read_start = time.perf_counter()
                if not values:
                    z_t = torch.zeros_like(v_t)
                    selected = -1
                    selected_sim = -1.0
                else:
                    centroid_tensor = torch.stack(centroids, dim=0).to(device=q_t.device)
                    sims = centroid_tensor @ q_t.detach()
                    scores = sims.clone()
                    if self.use_edges and edges:
                        eligible = []
                        for concept_id in range(len(values)):
                            ok = True
                            for edge_id, side in incident[concept_id]:
                                u, bias, _, _ = edges[edge_id]
                                bit = 1 if float(torch.dot(q_t.detach(), u).item()) > bias else 0
                                if bit != side:
                                    ok = False
                                    break
                            eligible.append(ok)
                        eligible_tensor = torch.tensor(eligible, device=q_t.device, dtype=torch.bool)
                        if bool(eligible_tensor.any().item()):
                            scores = scores.masked_fill(~eligible_tensor, -1.0e9)
                    scale = self.read_logit_scale.exp().clamp(max=100.0)
                    weights = torch.softmax(scores * scale, dim=0)
                    value_tensor = torch.stack(values, dim=0)
                    z_t = (weights.to(value_tensor.dtype).unsqueeze(-1) * value_tensor).sum(dim=0)
                    selected = int(torch.argmax(weights).item())
                    selected_sim = float(sims[selected].item())
                stats.read_seconds += time.perf_counter() - read_start
                seq_outputs.append(z_t)

                write_start = time.perf_counter()
                if selected < 0:
                    self._add_concept(centroids, anchors, values, counts, incident, k_t, v_t)
                else:
                    residual = float(torch.sqrt(F.mse_loss(values[selected].detach().float(), v_t.detach().float())).item())
                    should_refine = (
                        len(values) < self.max_concepts
                        and (selected_sim < self.create_threshold or residual > self.refine_threshold)
                    )
                    if should_refine:
                        child = self._add_concept(centroids, anchors, values, counts, incident, k_t, v_t)
                        if self.use_edges and child is not None:
                            self._add_edge(edges, incident, anchors, selected, child)
                    else:
                        old_count = counts[selected]
                        new_count = old_count + 1.0
                        values[selected] = (values[selected] * old_count + v_t) / new_count
                        updated = F.normalize(
                            (1.0 - self.centroid_lr) * centroids[selected] + self.centroid_lr * k_t,
                            dim=0,
                        )
                        centroids[selected] = updated.detach()
                        counts[selected] = new_count
                stats.write_seconds += time.perf_counter() - write_start

            stats.concepts += float(len(values))
            stats.edges += float(len(edges))
            outputs.append(torch.stack(seq_outputs, dim=0))

        self.last_stats = stats
        y = torch.stack(outputs, dim=0).to(dtype=x.dtype)
        return self.out_proj(self.dropout(y))

    def _add_concept(
        self,
        centroids: list[torch.Tensor],
        anchors: list[torch.Tensor],
        values: list[torch.Tensor],
        counts: list[float],
        incident: list[list[tuple[int, int]]],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> int | None:
        if len(values) >= self.max_concepts:
            return None
        idx = len(values)
        centroids.append(key.detach())
        anchors.append(key.detach())
        values.append(value)
        counts.append(1.0)
        incident.append([])
        return idx

    def _add_edge(
        self,
        edges: list[tuple[torch.Tensor, float, int, int]],
        incident: list[list[tuple[int, int]]],
        anchors: list[torch.Tensor],
        src: int,
        dst: int,
    ) -> None:
        if src == dst:
            return
        if len(incident[src]) >= self.max_incident_edges or len(incident[dst]) >= self.max_incident_edges:
            return
        src_anchor = anchors[src]
        dst_anchor = anchors[dst]
        u = F.normalize(dst_anchor - src_anchor, dim=0)
        if not bool(torch.isfinite(u).all().item()):
            return
        bias = float(0.5 * torch.dot(u, src_anchor + dst_anchor).item())
        edge_id = len(edges)
        edges.append((u.detach(), bias, src, dst))
        incident[src].append((edge_id, 0))
        incident[dst].append((edge_id, 1))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, multiple: int, dropout: float) -> None:
        super().__init__()
        hidden = multiple * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, args: argparse.Namespace, mixer_kind: str) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(args.d_model)
        self.ln_2 = nn.LayerNorm(args.d_model)
        if mixer_kind == "attention":
            self.mixer = CausalSelfAttention(args.d_model, args.n_heads, args.dropout)
        elif mixer_kind == "cga":
            self.mixer = CGAMixer(
                args.d_model,
                args.cga_state_dim,
                max_concepts=args.cga_max_concepts,
                centroid_lr=args.cga_centroid_lr,
                create_threshold=args.cga_create_threshold,
                refine_threshold=args.cga_refine_threshold,
                use_edges=args.cga_use_edges,
                max_incident_edges=args.cga_max_incident_edges,
                dropout=args.dropout,
            )
        else:
            raise ValueError(f"unknown mixer: {mixer_kind}")
        self.ff = FeedForward(args.d_model, args.ff_multiple, args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, args: argparse.Namespace, vocab_size: int, mixer_kind: str) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.block_size = int(args.block_size)
        self.mixer_kind = mixer_kind
        self.token_emb = nn.Embedding(vocab_size, args.d_model)
        self.pos_emb = nn.Embedding(args.block_size, args.d_model)
        self.drop = nn.Dropout(args.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(args, mixer_kind) for _ in range(args.n_layers)])
        self.ln_f = nn.LayerNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, vocab_size, bias=False)
        if args.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, time_steps = input_ids.shape
        if time_steps > self.block_size:
            raise ValueError("input sequence exceeds block_size")
        pos = torch.arange(time_steps, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos).unsqueeze(0)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    def cga_stats(self) -> dict[str, float]:
        stats = []
        for block in self.blocks:
            if isinstance(block.mixer, CGAMixer):
                stats.append(block.mixer.last_stats.to_dict())
        if not stats:
            return {}
        out: dict[str, float] = {}
        for key in stats[0]:
            out[key] = float(sum(item[key] for item in stats) / len(stats))
        return out


@torch.no_grad()
def evaluate(model: TinyLM, tokens: torch.Tensor, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    steps = max(1, args.eval_batches)
    for _ in range(steps):
        x, y = sample_batch(tokens, args.eval_batch_size, args.block_size, device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.item()))
    mean_loss = sum(losses) / len(losses)
    out = {
        "nll": mean_loss,
        "ppl": math.exp(min(50.0, mean_loss)),
    }
    out.update({f"cga_{k}": v for k, v in model.cga_stats().items()})
    return out


@torch.no_grad()
def benchmark_forward_latency(model: TinyLM, args: argparse.Namespace, device: torch.device) -> list[dict[str, float]]:
    model.eval()
    lengths = [int(item) for item in args.latency_lengths.split(",") if item.strip()]
    rows = []
    for length in lengths:
        if length <= 0 or length > args.block_size:
            continue
        x = torch.randint(0, model.vocab_size, (args.latency_batch_size, length), device=device)
        for _ in range(args.latency_warmup):
            model(x)
        sync_device(device)
        times = []
        for _ in range(args.latency_repeats):
            start = timed(device)
            model(x)
            sync_device(device)
            times.append(time.perf_counter() - start)
        mean_seconds = sum(times) / len(times)
        rows.append(
            {
                "length": length,
                "batch_size": args.latency_batch_size,
                "mean_seconds": mean_seconds,
                "tokens_per_second": args.latency_batch_size * length / max(1.0e-12, mean_seconds),
                "milliseconds_per_token": 1000.0 * mean_seconds / max(1, args.latency_batch_size * length),
            }
        )
    return rows


def train_one_model(
    name: str,
    mixer_kind: str,
    args: argparse.Namespace,
    vocab_size: int,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = TinyLM(args, vocab_size, mixer_kind).to(device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    train_losses: list[float] = []
    start = timed(device)
    model.train()

    for step in range(1, args.train_steps + 1):
        x, y = sample_batch(train_tokens, args.batch_size, args.block_size, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32 and device.type in {"cuda", "cpu"})):
            _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(float(loss.detach().float().item()))
        if step % args.log_every == 0 or step == 1:
            elapsed = timed(device) - start
            recent = sum(train_losses[-args.log_every :]) / min(len(train_losses), args.log_every)
            print(
                f"[tiny-cga-lm] model={name} step={step}/{args.train_steps} "
                f"loss={recent:.4f} ppl={math.exp(min(50.0, recent)):.2f} "
                f"tok/s={step * args.batch_size * args.block_size / max(1.0e-12, elapsed):.1f}",
                flush=True,
            )

    train_seconds = timed(device) - start
    val = evaluate(model, val_tokens, args, device)
    latency = benchmark_forward_latency(model, args, device)
    output: dict[str, object] = {
        "name": name,
        "mixer": mixer_kind,
        "parameters": count_parameters(model),
        "train_seconds": train_seconds,
        "train_tokens": args.train_steps * args.batch_size * args.block_size,
        "train_tokens_per_second": args.train_steps * args.batch_size * args.block_size / max(1.0e-12, train_seconds),
        "final_train_nll": sum(train_losses[-min(50, len(train_losses)) :]) / min(50, len(train_losses)),
        "validation": val,
        "latency": latency,
    }
    if args.save_checkpoints:
        ckpt_dir = args.output.parent / "tiny_cga_lm_checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{name}.pt"
        torch.save({"model": model.state_dict(), "args": vars(args), "vocab_size": vocab_size}, ckpt_path)
        output["checkpoint"] = str(ckpt_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/results_tiny_cga_lm.json"))
    parser.add_argument("--models", default="transformer,cga")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text-file", default="")
    parser.add_argument("--text-train-fraction", type=float, default=0.9)
    parser.add_argument("--max-train-tokens", type=int, default=2_000_000)
    parser.add_argument("--max-val-tokens", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")

    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-multiple", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tie-embeddings", action="store_true")

    parser.add_argument("--cga-state-dim", type=int, default=128)
    parser.add_argument("--cga-max-concepts", type=int, default=64)
    parser.add_argument("--cga-centroid-lr", type=float, default=0.05)
    parser.add_argument("--cga-create-threshold", type=float, default=0.15)
    parser.add_argument("--cga-refine-threshold", type=float, default=0.85)
    parser.add_argument("--cga-use-edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cga-max-incident-edges", type=int, default=32)

    parser.add_argument("--latency-lengths", default="64,128")
    parser.add_argument("--latency-batch-size", type=int, default=8)
    parser.add_argument("--latency-warmup", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=10)
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    print(f"[tiny-cga-lm] device={device} dtype={dtype}", flush=True)
    if args.tokenizer == "byte":
        tokenizer = ByteTokenizer()
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers or pass --tokenizer byte.") from exc
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    vocab_size = int(len(tokenizer))
    print("[tiny-cga-lm] loading tokens", flush=True)
    train_tokens = cap_tokens(load_text_split(args, tokenizer, "train"), args.max_train_tokens)
    val_tokens = cap_tokens(load_text_split(args, tokenizer, "val"), args.max_val_tokens)
    print(f"[tiny-cga-lm] train_tokens={train_tokens.numel()} val_tokens={val_tokens.numel()} vocab={vocab_size}", flush=True)

    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    name_to_mixer = {
        "transformer": "attention",
        "cga": "cga",
        "tiny_cga_lm": "cga",
    }
    results = []
    for name in requested:
        if name not in name_to_mixer:
            raise ValueError(f"unknown model '{name}'")
        results.append(train_one_model(name, name_to_mixer[name], args, vocab_size, train_tokens, val_tokens, device, dtype))

    result = {
        "protocol": (
            "Tiny causal language models trained from scratch. The Transformer baseline uses causal self-attention. "
            "TinyCGA-LM replaces the attention mixer in each decoder block with a causal CGA-Mixer that reads a "
            "runtime graph-memory state before observing the current token representation. This is an architecture "
            "experiment, not a frozen-LM readout experiment."
        ),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch_version": torch.__version__,
        "device": str(device),
        "dtype": str(dtype),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[tiny-cga-lm] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
