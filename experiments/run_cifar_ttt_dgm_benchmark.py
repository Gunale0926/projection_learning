from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_real_training_benchmark import (
    CIFAR10_ROOT,
    FullDGMRefineMemory,
    SmallConvNet,
    load_arrays,
    make_loader,
    select_device,
    set_seed,
    stratified_subset,
)


@dataclass
class Row:
    seed: int
    corruption: str
    method: str
    accuracy: float
    nll: float
    train_seconds: float
    eval_seconds: float
    concepts: int
    edges: int
    memory_bytes: int
    config: dict[str, object]


def corrupt_batch(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "clean":
        return x
    if name == "noise":
        return x + 0.35 * torch.randn_like(x)
    if name == "blur":
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    if name == "contrast":
        return 0.65 * x
    raise ValueError(f"unknown corruption: {name}")


def rotate_views(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    views = [torch.rot90(x, k=k, dims=(2, 3)) for k in range(4)]
    labels = [torch.full((x.shape[0],), k, dtype=torch.long, device=x.device) for k in range(4)]
    return torch.cat(views, dim=0), torch.cat(labels, dim=0)


class TTTConvNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, embedding_dim: int) -> None:
        super().__init__()
        base = SmallConvNet(in_channels=in_channels, n_classes=n_classes, embedding_dim=embedding_dim)
        self.encoder = base.encoder
        self.head = base.head
        self.rot_head = nn.Linear(embedding_dim, 4)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        return nn.functional.normalize(h, dim=1), self.head(h)

    def rotation_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.rot_head(self.features(x))


class LoRAAdapter(nn.Module):
    def __init__(self, source: SmallConvNet, rank: int, alpha: float) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(source.encoder)
        self.head = copy.deepcopy(source.head)
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        for param in self.head.parameters():
            param.requires_grad_(False)
        dim = int(self.head.in_features)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.scale = float(alpha) / float(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h + self.scale * self.up(self.down(h))
        return self.head(h)


def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> tuple[int, float, float]:
    loss = F.cross_entropy(logits, y, reduction="sum")
    pred = logits.argmax(dim=1)
    correct = int((pred == y).sum().item())
    return correct, float(loss.item()), float(y.numel())


def finish_metrics(correct: int, nll_sum: float, count: float) -> tuple[float, float]:
    return float(correct / count), float(nll_sum / count)


def train_source(
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SmallConvNet, float]:
    set_seed(seed)
    model = SmallConvNet(in_channels=int(x_train.shape[1]), embedding_dim=args.embedding_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(x_train, y_train, args.batch_size, shuffle=True, seed=seed + 11)
    start = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            _, logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    sync_device(device)
    return model, time.perf_counter() - start


def train_ttt(
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TTTConvNet, float]:
    set_seed(seed + 1000)
    model = TTTConvNet(in_channels=int(x_train.shape[1]), n_classes=args.n_classes, embedding_dim=args.embedding_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(x_train, y_train, args.batch_size, shuffle=True, seed=seed + 29)
    start = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            _, logits = model(xb)
            rot_x, rot_y = rotate_views(xb)
            rot_logits = model.rotation_logits(rot_x)
            loss = F.cross_entropy(logits, yb) + args.rotation_weight * F.cross_entropy(rot_logits, rot_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    sync_device(device)
    return model, time.perf_counter() - start


@torch.no_grad()
def evaluate_source(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    corruption: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float]:
    loader = make_loader(x_test, y_test, args.batch_size, shuffle=False, seed=123)
    model.eval()
    correct = 0
    nll_sum = 0.0
    count = 0.0
    start = time.perf_counter()
    for xb, yb in loader:
        xb = corrupt_batch(xb.to(device), corruption)
        yb = yb.to(device)
        _, logits = model(xb)
        c, nll, n = metrics_from_logits(logits, yb)
        correct += c
        nll_sum += nll
        count += n
    sync_device(device)
    acc, nll = finish_metrics(correct, nll_sum, count)
    return acc, nll, time.perf_counter() - start


def build_dgm(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    compact: bool,
) -> FullDGMRefineMemory:
    loader = make_loader(x_train, y_train, args.batch_size, shuffle=False, seed=321)
    memory = FullDGMRefineMemory(
        n_classes=args.n_classes,
        embedding_dim=args.embedding_dim,
        max_concepts=args.dgm_max_concepts,
        buffer_size=args.dgm_buffer_size,
        top_k=args.dgm_top_k,
        temperature=args.dgm_temperature,
        alpha=1.0,
        min_buffer=args.dgm_min_buffer,
        min_child=args.dgm_min_child,
        split_penalty=args.dgm_split_penalty,
        max_splits_per_batch=args.dgm_max_splits_per_batch,
        device=device,
        edge_degree=args.dgm_edge_degree,
        min_edge_divergence=args.dgm_min_edge_divergence,
        max_incident_edges=args.dgm_max_incident_edges,
    )
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            z, _ = model(xb)
            memory.observe(z, yb)
    if compact:
        memory.compact_for_inference()
    return memory


@torch.no_grad()
def evaluate_dgm(
    model: nn.Module,
    memory: FullDGMRefineMemory,
    x_test: np.ndarray,
    y_test: np.ndarray,
    corruption: str,
    online: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float]:
    loader = make_loader(x_test, y_test, args.batch_size, shuffle=False, seed=456)
    model.eval()
    correct = 0
    nll_sum = 0.0
    count = 0.0
    start = time.perf_counter()
    for xb, yb in loader:
        xb = corrupt_batch(xb.to(device), corruption)
        yb = yb.to(device)
        z, base_logits = model(xb)
        logits = base_logits + args.dgm_weight * memory.logits(z)
        c, nll, n = metrics_from_logits(logits, yb)
        correct += c
        nll_sum += nll
        count += n
        if online:
            probs = torch.softmax(logits, dim=1)
            conf, pseudo = probs.max(dim=1)
            keep = conf >= args.dgm_pseudo_threshold
            if bool(keep.any().item()):
                memory.observe(z[keep], pseudo[keep])
    sync_device(device)
    acc, nll = finish_metrics(correct, nll_sum, count)
    return acc, nll, time.perf_counter() - start


def evaluate_entropy_min(
    source_model: SmallConvNet,
    x_test: np.ndarray,
    y_test: np.ndarray,
    corruption: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float]:
    model = copy.deepcopy(source_model).to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.entropy_lr)
    loader = make_loader(x_test, y_test, args.batch_size, shuffle=False, seed=567)
    correct = 0
    nll_sum = 0.0
    count = 0.0
    start = time.perf_counter()
    for xb, yb in loader:
        xb = corrupt_batch(xb.to(device), corruption)
        yb = yb.to(device)
        for _ in range(args.tta_steps):
            _, logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            entropy = -(probs.clamp_min(1e-8) * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            entropy.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            _, logits = model(xb)
        c, nll, n = metrics_from_logits(logits, yb)
        correct += c
        nll_sum += nll
        count += n
        model.train()
    sync_device(device)
    acc, nll = finish_metrics(correct, nll_sum, count)
    return acc, nll, time.perf_counter() - start


def evaluate_lora_entropy(
    source_model: SmallConvNet,
    x_test: np.ndarray,
    y_test: np.ndarray,
    corruption: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float]:
    model = LoRAAdapter(source_model, rank=args.lora_rank, alpha=args.lora_alpha).to(device)
    opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=args.lora_lr)
    loader = make_loader(x_test, y_test, args.batch_size, shuffle=False, seed=589)
    correct = 0
    nll_sum = 0.0
    count = 0.0
    start = time.perf_counter()
    for xb, yb in loader:
        xb = corrupt_batch(xb.to(device), corruption)
        yb = yb.to(device)
        model.train()
        for _ in range(args.tta_steps):
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            entropy = -(probs.clamp_min(1e-8) * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            entropy.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(xb)
        c, nll, n = metrics_from_logits(logits, yb)
        correct += c
        nll_sum += nll
        count += n
    sync_device(device)
    acc, nll = finish_metrics(correct, nll_sum, count)
    return acc, nll, time.perf_counter() - start


def evaluate_ttt_rotation(
    source_model: TTTConvNet,
    x_test: np.ndarray,
    y_test: np.ndarray,
    corruption: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float]:
    model = copy.deepcopy(source_model).to(device)
    for param in model.head.parameters():
        param.requires_grad_(False)
    opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=args.ttt_lr)
    loader = make_loader(x_test, y_test, args.batch_size, shuffle=False, seed=678)
    correct = 0
    nll_sum = 0.0
    count = 0.0
    start = time.perf_counter()
    for xb, yb in loader:
        xb = corrupt_batch(xb.to(device), corruption)
        yb = yb.to(device)
        model.train()
        for _ in range(args.tta_steps):
            rot_x, rot_y = rotate_views(xb)
            rot_logits = model.rotation_logits(rot_x)
            loss = F.cross_entropy(rot_logits, rot_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            _, logits = model(xb)
        c, nll, n = metrics_from_logits(logits, yb)
        correct += c
        nll_sum += nll
        count += n
    sync_device(device)
    acc, nll = finish_metrics(correct, nll_sum, count)
    return acc, nll, time.perf_counter() - start


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def summarize(rows: list[Row]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for corruption in sorted({row.corruption for row in rows}):
        for method in sorted({row.method for row in rows if row.corruption == corruption}):
            group = [row for row in rows if row.corruption == corruption and row.method == method]
            key = f"{corruption}/{method}"
            out[key] = {}
            for field in ["accuracy", "nll", "train_seconds", "eval_seconds", "concepts", "edges", "memory_bytes"]:
                values = np.asarray([float(getattr(row, field)) for row in group], dtype=float)
                out[key][f"{field}_mean"] = float(values.mean())
                out[key][f"{field}_std"] = float(values.std(ddof=0))
                out[key][f"{field}_se"] = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar-root", default=CIFAR10_ROOT)
    parser.add_argument("--output", type=Path, default=Path("experiments/results_cifar_ttt_dgm.json"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--corruptions", default="clean,noise,blur,contrast")
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--rotation-weight", type=float, default=0.5)
    parser.add_argument("--tta-steps", type=int, default=1)
    parser.add_argument("--ttt-lr", type=float, default=2e-4)
    parser.add_argument("--entropy-lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--dgm-weight", type=float, default=0.6)
    parser.add_argument("--dgm-pseudo-threshold", type=float, default=0.9)
    parser.add_argument("--dgm-max-concepts", type=int, default=2048)
    parser.add_argument("--dgm-buffer-size", type=int, default=64)
    parser.add_argument("--dgm-top-k", type=int, default=16)
    parser.add_argument("--dgm-temperature", type=float, default=0.18)
    parser.add_argument("--dgm-min-buffer", type=int, default=24)
    parser.add_argument("--dgm-min-child", type=int, default=4)
    parser.add_argument("--dgm-split-penalty", type=float, default=1.0)
    parser.add_argument("--dgm-max-splits-per-batch", type=int, default=12)
    parser.add_argument("--dgm-edge-degree", type=int, default=2)
    parser.add_argument("--dgm-min-edge-divergence", type=float, default=0.05)
    parser.add_argument("--dgm-max-incident-edges", type=int, default=96)
    args = parser.parse_args()

    device = select_device(args.device)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    corruptions = [item.strip() for item in args.corruptions.split(",") if item.strip()]
    x_train_all, y_train_all, x_test_all, y_test_all = load_arrays("CIFAR10", args.cifar_root)
    rows: list[Row] = []
    for seed in seeds:
        rng = np.random.default_rng(50_000 + seed)
        train_idx = stratified_subset(y_train_all, args.train_per_class, rng)
        test_idx = stratified_subset(y_test_all, args.test_per_class, rng)
        x_train = x_train_all[train_idx]
        y_train = y_train_all[train_idx]
        x_test = x_test_all[test_idx]
        y_test = y_test_all[test_idx]

        source_model, source_train_seconds = train_source(seed, x_train, y_train, args, device)
        ttt_model, ttt_train_seconds = train_ttt(seed, x_train, y_train, args, device)
        dgm_build_start = time.perf_counter()
        dgm_memory = build_dgm(source_model, x_train, y_train, args, device, compact=True)
        dgm_build_seconds = time.perf_counter() - dgm_build_start
        dgm_online_build_start = time.perf_counter()
        dgm_online_template = build_dgm(source_model, x_train, y_train, args, device, compact=False)
        dgm_online_build_seconds = time.perf_counter() - dgm_online_build_start

        for corruption in corruptions:
            acc, nll, eval_seconds = evaluate_source(source_model, x_test, y_test, corruption, args, device)
            rows.append(Row(seed, corruption, "Source", acc, nll, source_train_seconds, eval_seconds, 0, 0, 0, {}))

            dgm_for_eval = copy.deepcopy(dgm_memory)
            acc, nll, eval_seconds = evaluate_dgm(source_model, dgm_for_eval, x_test, y_test, corruption, False, args, device)
            rows.append(
                Row(
                    seed,
                    corruption,
                    "DGM",
                    acc,
                    nll,
                    source_train_seconds,
                    eval_seconds,
                    dgm_for_eval.concepts,
                    dgm_for_eval.edges,
                    dgm_for_eval.estimated_bytes,
                    {"dgm_weight": args.dgm_weight, "build_seconds": dgm_build_seconds},
                )
            )

            online_memory = copy.deepcopy(dgm_online_template)
            acc, nll, eval_seconds = evaluate_dgm(source_model, online_memory, x_test, y_test, corruption, True, args, device)
            rows.append(
                Row(
                    seed,
                    corruption,
                    "DGM-Online",
                    acc,
                    nll,
                    source_train_seconds,
                    eval_seconds,
                    online_memory.concepts,
                    online_memory.edges,
                    online_memory.estimated_bytes,
                    {
                        "pseudo_threshold": args.dgm_pseudo_threshold,
                        "dgm_weight": args.dgm_weight,
                        "build_seconds": dgm_online_build_seconds,
                    },
                )
            )

            acc, nll, eval_seconds = evaluate_entropy_min(source_model, x_test, y_test, corruption, args, device)
            rows.append(Row(seed, corruption, "EntropyMin", acc, nll, source_train_seconds, eval_seconds, 0, 0, 0, {}))

            acc, nll, eval_seconds = evaluate_lora_entropy(source_model, x_test, y_test, corruption, args, device)
            rows.append(
                Row(
                    seed,
                    corruption,
                    "LoRA-Entropy",
                    acc,
                    nll,
                    source_train_seconds,
                    eval_seconds,
                    0,
                    0,
                    0,
                    {"rank": args.lora_rank, "alpha": args.lora_alpha, "lr": args.lora_lr, "tta_steps": args.tta_steps},
                )
            )

            acc, nll, eval_seconds = evaluate_ttt_rotation(ttt_model, x_test, y_test, corruption, args, device)
            rows.append(
                Row(
                    seed,
                    corruption,
                    "TTT-Rotation",
                    acc,
                    nll,
                    ttt_train_seconds,
                    eval_seconds,
                    0,
                    0,
                    0,
                    {"rotation_weight": args.rotation_weight, "ttt_lr": args.ttt_lr, "tta_steps": args.tta_steps},
                )
            )

            print(
                f"seed={seed} corruption={corruption} "
                + " ".join(
                    f"{row.method}:{row.accuracy:.3f}"
                    for row in rows
                    if row.seed == seed and row.corruption == corruption
                ),
                flush=True,
            )
            args.output.with_suffix(args.output.suffix + ".partial").write_text(
                json.dumps({"rows": [asdict(row) for row in rows], "summary": summarize(rows)}, indent=2, sort_keys=True)
            )

    result = {
        "protocol": (
            "CIFAR-10 test-time learning benchmark. Source is a standard supervised CNN. "
            "DGM is the same CNN with a graph-memory readout built from training embeddings. "
            "DGM-Online additionally updates memory at test time with high-confidence pseudo-labels and no backpropagation. "
            "TTT-Rotation trains an auxiliary rotation self-supervised head and adapts it online at test time. "
            "EntropyMin is an online entropy-minimization test-time adaptation baseline. "
            "LoRA-Entropy adapts only a low-rank residual embedding adapter with the same entropy objective."
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "seeds": seeds,
        "corruptions": corruptions,
        "hyperparameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rows": [asdict(row) for row in rows],
        "summary": summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
