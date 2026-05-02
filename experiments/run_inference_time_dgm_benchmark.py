from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_real_training_benchmark import (
    CIFAR10_ROOT,
    FullDGMRefineMemory,
    MetricTotals,
    SmallConvNet,
    load_arrays,
    select_device,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_pretrain_adapt(
    y: np.ndarray,
    pretrain_per_class: int,
    adapt_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    pretrain: list[int] = []
    adapt: list[int] = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == int(cls))
        rng.shuffle(idx)
        need = pretrain_per_class + adapt_per_class
        if len(idx) < need:
            raise ValueError(f"class {cls} has {len(idx)} examples, need {need}")
        pretrain.extend(idx[:pretrain_per_class].tolist())
        adapt.extend(idx[pretrain_per_class:need].tolist())
    pretrain_arr = np.asarray(pretrain, dtype=np.int64)
    adapt_arr = np.asarray(adapt, dtype=np.int64)
    rng.shuffle(pretrain_arr)
    rng.shuffle(adapt_arr)
    return pretrain_arr, adapt_arr


def stratified_subset(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    out: list[int] = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == int(cls))
        rng.shuffle(idx)
        if len(idx) < per_class:
            raise ValueError(f"class {cls} has {len(idx)} examples, need {per_class}")
        out.extend(idx[:per_class].tolist())
    arr = np.asarray(out, dtype=np.int64)
    rng.shuffle(arr)
    return arr


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_shared_model(
    x: np.ndarray,
    y: np.ndarray,
    in_channels: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[SmallConvNet, float]:
    set_seed(seed)
    model = SmallConvNet(in_channels=in_channels, embedding_dim=embedding_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = make_loader(x, y, batch_size=batch_size, shuffle=True, seed=seed + 123)
    start = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        totals = MetricTotals(device)
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            _, logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals.observe(logits, yb)
        print(f"pretrain epoch={epoch + 1}/{epochs} acc={totals.accuracy():.3f}", flush=True)
    sync(device)
    elapsed = time.perf_counter() - start
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, elapsed


@torch.no_grad()
def embed_stream(
    model: SmallConvNet,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    loader = make_loader(x, y, batch_size=batch_size, shuffle=False, seed=0)
    zs: list[torch.Tensor] = []
    hs: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    start = time.perf_counter()
    model.eval()
    for xb, yb in loader:
        xb = xb.to(device)
        h = model.encoder(xb)
        z = nn.functional.normalize(h, dim=1)
        base_logits = model.head(h)
        zs.append(z.detach())
        hs.append(h.detach())
        logits.append(base_logits.detach())
        labels.append(yb.to(device))
    sync(device)
    return (
        torch.cat(zs, dim=0),
        torch.cat(hs, dim=0),
        torch.cat(logits, dim=0),
        torch.cat(labels, dim=0),
        time.perf_counter() - start,
    )


def iter_batches(n: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


@dataclass
class AdaptResult:
    dataset: str
    method: str
    seed: int
    prequential_accuracy: float
    prequential_nll: float
    post_accuracy: float
    post_nll: float
    adapt_seconds: float
    eval_seconds: float
    backward_steps: int
    trainable_parameters: int
    concepts: int
    memory_bytes: int


@torch.no_grad()
def evaluate_logits(logits: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[float, float]:
    totals = MetricTotals(device)
    totals.observe(logits, y)
    return totals.accuracy(), totals.nll_value()


def run_frozen(
    dataset: str,
    seed: int,
    adapt_logits: torch.Tensor,
    adapt_y: torch.Tensor,
    test_logits: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
) -> AdaptResult:
    start = time.perf_counter()
    preq_acc, preq_nll = evaluate_logits(adapt_logits, adapt_y, device)
    sync(device)
    adapt_seconds = time.perf_counter() - start
    eval_start = time.perf_counter()
    post_acc, post_nll = evaluate_logits(test_logits, test_y, device)
    sync(device)
    eval_seconds = time.perf_counter() - eval_start
    return AdaptResult(
        dataset=dataset,
        method="frozen",
        seed=seed,
        prequential_accuracy=preq_acc,
        prequential_nll=preq_nll,
        post_accuracy=post_acc,
        post_nll=post_nll,
        adapt_seconds=adapt_seconds,
        eval_seconds=eval_seconds,
        backward_steps=0,
        trainable_parameters=0,
        concepts=0,
        memory_bytes=0,
    )


def run_backprop_head(
    dataset: str,
    seed: int,
    pretrained_head: nn.Linear,
    adapt_h: torch.Tensor,
    adapt_y: torch.Tensor,
    test_h: torch.Tensor,
    test_y: torch.Tensor,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> AdaptResult:
    set_seed(seed)
    head = nn.Linear(adapt_h.shape[1], 10).to(device)
    head.load_state_dict(pretrained_head.state_dict())
    for param in head.parameters():
        param.requires_grad_(True)
    opt = torch.optim.SGD(head.parameters(), lr=lr)
    totals = MetricTotals(device)
    backward_steps = 0
    start = time.perf_counter()
    for sl in iter_batches(adapt_y.numel(), batch_size):
        h = adapt_h[sl].detach()
        y = adapt_y[sl]
        logits = head(h)
        totals.observe(logits, y)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        backward_steps += 1
    sync(device)
    adapt_seconds = time.perf_counter() - start
    eval_start = time.perf_counter()
    with torch.no_grad():
        post_logits = head(test_h.detach())
        post_acc, post_nll = evaluate_logits(post_logits, test_y, device)
    sync(device)
    eval_seconds = time.perf_counter() - eval_start
    return AdaptResult(
        dataset=dataset,
        method="backprop_head",
        seed=seed,
        prequential_accuracy=totals.accuracy(),
        prequential_nll=totals.nll_value(),
        post_accuracy=post_acc,
        post_nll=post_nll,
        adapt_seconds=adapt_seconds,
        eval_seconds=eval_seconds,
        backward_steps=backward_steps,
        trainable_parameters=sum(p.numel() for p in head.parameters()),
        concepts=0,
        memory_bytes=0,
    )


@torch.no_grad()
def run_full_dgm(
    dataset: str,
    seed: int,
    adapt_z: torch.Tensor,
    adapt_base_logits: torch.Tensor,
    adapt_y: torch.Tensor,
    test_z: torch.Tensor,
    test_base_logits: torch.Tensor,
    test_y: torch.Tensor,
    batch_size: int,
    memory_weight: float,
    args: argparse.Namespace,
    device: torch.device,
) -> AdaptResult:
    memory = FullDGMRefineMemory(
        n_classes=10,
        embedding_dim=adapt_z.shape[1],
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
    totals = MetricTotals(device)
    start = time.perf_counter()
    with torch.no_grad():
        for sl in iter_batches(adapt_y.numel(), batch_size):
            z = adapt_z[sl]
            y = adapt_y[sl]
            logits = adapt_base_logits[sl] + memory_weight * memory.logits(z)
            totals.observe(logits, y)
            memory.observe(z, y)
    sync(device)
    adapt_seconds = time.perf_counter() - start
    eval_start = time.perf_counter()
    with torch.no_grad():
        post_logits = test_base_logits + memory_weight * memory.logits(test_z)
        post_acc, post_nll = evaluate_logits(post_logits, test_y, device)
    sync(device)
    eval_seconds = time.perf_counter() - eval_start
    return AdaptResult(
        dataset=dataset,
        method="full_dgm",
        seed=seed,
        prequential_accuracy=totals.accuracy(),
        prequential_nll=totals.nll_value(),
        post_accuracy=post_acc,
        post_nll=post_nll,
        adapt_seconds=adapt_seconds,
        eval_seconds=eval_seconds,
        backward_steps=0,
        trainable_parameters=0,
        concepts=memory.concepts,
        memory_bytes=memory.estimated_bytes,
    )


def summarize(rows: Iterable[AdaptResult]) -> dict[str, float]:
    row_list = list(rows)
    out: dict[str, float] = {}
    for key in [
        "prequential_accuracy",
        "prequential_nll",
        "post_accuracy",
        "post_nll",
        "adapt_seconds",
        "eval_seconds",
        "backward_steps",
        "trainable_parameters",
        "concepts",
        "memory_bytes",
    ]:
        values = np.asarray([float(getattr(row, key)) for row in row_list], dtype=float)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    return out


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"frozen", "backprop_head", "full_dgm"}
    bad = sorted(set(methods) - allowed)
    if bad:
        raise ValueError(f"unknown methods: {bad}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["CIFAR10"], default="CIFAR10")
    parser.add_argument("--cifar-root", type=str, default=CIFAR10_ROOT)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--methods", type=str, default="frozen,backprop_head,full_dgm")
    parser.add_argument("--output", type=Path, default=Path("experiments/results_inference_time_dgm_cifar10.json"))
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--pretrain-per-class", type=int, default=800)
    parser.add_argument("--adapt-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pretrain-lr", type=float, default=2e-3)
    parser.add_argument("--head-lr", type=float, default=0.05)
    parser.add_argument("--memory-weight", type=float, default=0.6)
    parser.add_argument("--memory-temperature", type=float, default=0.20)
    parser.add_argument("--max-concepts", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-buffer", type=int, default=32)
    parser.add_argument("--min-child", type=int, default=6)
    parser.add_argument("--split-penalty", type=float, default=3.0)
    parser.add_argument("--max-splits-per-batch", type=int, default=4)
    args = parser.parse_args()

    device = select_device(args.device)
    methods = parse_methods(args.methods)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    x_train_all, y_train_all, x_test_all, y_test_all = load_arrays(args.dataset, args.cifar_root)
    runs: list[AdaptResult] = []
    seed_summaries: list[dict[str, object]] = []

    for seed in seeds:
        rng = np.random.default_rng(50_000 + seed)
        pre_idx, adapt_idx = split_pretrain_adapt(
            y_train_all,
            pretrain_per_class=args.pretrain_per_class,
            adapt_per_class=args.adapt_per_class,
            rng=rng,
        )
        test_idx = stratified_subset(y_test_all, args.test_per_class, rng)
        print(
            f"{args.dataset} seed={seed}: pretrain={len(pre_idx)} adapt={len(adapt_idx)} "
            f"test={len(test_idx)} device={device}",
            flush=True,
        )
        model, pretrain_seconds = train_shared_model(
            x_train_all[pre_idx],
            y_train_all[pre_idx],
            in_channels=int(x_train_all.shape[1]),
            embedding_dim=args.embedding_dim,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            lr=args.pretrain_lr,
            seed=seed,
            device=device,
        )
        adapt_z, adapt_h, adapt_logits, adapt_y, adapt_embed_seconds = embed_stream(
            model,
            x_train_all[adapt_idx],
            y_train_all[adapt_idx],
            batch_size=args.batch_size,
            device=device,
        )
        test_z, test_h, test_logits, test_y, test_embed_seconds = embed_stream(
            model,
            x_test_all[test_idx],
            y_test_all[test_idx],
            batch_size=args.batch_size,
            device=device,
        )
        seed_summaries.append(
            {
                "seed": seed,
                "pretrain_seconds_shared_not_counted": pretrain_seconds,
                "adapt_embed_seconds_shared_not_counted": adapt_embed_seconds,
                "test_embed_seconds_shared_not_counted": test_embed_seconds,
            }
        )
        for method in methods:
            if method == "frozen":
                run = run_frozen(args.dataset, seed, adapt_logits, adapt_y, test_logits, test_y, device)
            elif method == "backprop_head":
                run = run_backprop_head(
                    args.dataset,
                    seed,
                    model.head,
                    adapt_h,
                    adapt_y,
                    test_h,
                    test_y,
                    batch_size=args.batch_size,
                    lr=args.head_lr,
                    device=device,
                )
            elif method == "full_dgm":
                run = run_full_dgm(
                    args.dataset,
                    seed,
                    adapt_z,
                    adapt_logits,
                    adapt_y,
                    test_z,
                    test_logits,
                    test_y,
                    batch_size=args.batch_size,
                    memory_weight=args.memory_weight,
                    args=args,
                    device=device,
                )
            else:
                raise AssertionError(method)
            runs.append(run)
            print(
                f"RESULT seed={seed} method={method} "
                f"preq_acc={run.prequential_accuracy:.4f} post_acc={run.post_accuracy:.4f} "
                f"post_nll={run.post_nll:.4f} adapt_seconds={run.adapt_seconds:.4f} "
                f"backward_steps={run.backward_steps} concepts={run.concepts}",
                flush=True,
            )
            args.output.with_suffix(args.output.suffix + ".partial").write_text(
                json.dumps({"runs": [asdict(row) for row in runs]}, indent=2, sort_keys=True)
            )

    summary = {method: summarize(row for row in runs if row.method == method) for method in methods}
    output = {
        "protocol": (
            "Inference-time adaptation on a shared frozen CNN. Shared pretraining and embedding "
            "forward passes are reported but excluded from adaptation overhead. full_dgm runs "
            "under torch.no_grad and has zero backward steps; backprop_head updates only the "
            "linear head on frozen embeddings."
        ),
        "dataset": args.dataset,
        "device": str(device),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "torch_version": torch.__version__,
        "methods": methods,
        "seeds": seeds,
        "shared_costs": seed_summaries,
        "hyperparameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": summary,
        "runs": [asdict(row) for row in runs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    for method in methods:
        row = summary[method]
        print(
            f"{method}: post_acc={row['post_accuracy_mean']:.4f}±{row['post_accuracy_std']:.4f} "
            f"post_nll={row['post_nll_mean']:.4f}±{row['post_nll_std']:.4f} "
            f"adapt_seconds={row['adapt_seconds_mean']:.4f}±{row['adapt_seconds_std']:.4f} "
            f"backward_steps={row['backward_steps_mean']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
