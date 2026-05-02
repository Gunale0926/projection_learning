from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.dgm_reference import CategoricalDGM


EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class ImageArrays:
    train_x: torch.Tensor
    train_y: torch.Tensor
    eval_x: torch.Tensor
    eval_y: torch.Tensor
    source: str


class RandomProjectionEncoder:
    def __init__(self, embedding_dim: int, seed: int, pool: int = 2) -> None:
        self.embedding_dim = int(embedding_dim)
        self.pool = int(pool)
        generator = torch.Generator().manual_seed(seed)
        input_dim = (28 // self.pool) * (28 // self.pool)
        proj = torch.randn(input_dim, self.embedding_dim, generator=generator)
        self.proj = proj / math.sqrt(float(input_dim))

    def transform(self, x: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
        chunks = []
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            if self.pool > 1:
                xb = F.avg_pool2d(xb, kernel_size=self.pool, stride=self.pool)
            flat = xb.flatten(1)
            flat = flat - flat.mean(dim=1, keepdim=True)
            z = flat @ self.proj.to(flat.device, flat.dtype)
            chunks.append(F.normalize(z, dim=1))
        return torch.cat(chunks, dim=0)


def load_mnist_arrays(
    root: Path,
    *,
    download: bool,
    synthetic: bool,
    n_train: int,
    n_eval: int,
    seed: int,
) -> ImageArrays:
    if synthetic:
        return make_synthetic_mnist_like(n_train=n_train, n_eval=n_eval, seed=seed)

    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("torchvision is required for the real MNIST smoke test") from exc

    train = MNIST(root=str(root), train=True, download=download)
    eval_ds = MNIST(root=str(root), train=False, download=download)
    train_x = train.data[:n_train].float().unsqueeze(1) / 255.0
    train_y = train.targets[:n_train].long()
    eval_x = eval_ds.data[:n_eval].float().unsqueeze(1) / 255.0
    eval_y = eval_ds.targets[:n_eval].long()
    return ImageArrays(train_x=train_x, train_y=train_y, eval_x=eval_x, eval_y=eval_y, source="MNIST")


def make_synthetic_mnist_like(n_train: int, n_eval: int, seed: int) -> ImageArrays:
    generator = torch.Generator().manual_seed(seed)
    prototypes = torch.rand(10, 1, 28, 28, generator=generator)

    def sample(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.randint(0, 10, (n,), generator=generator)
        noise = 0.18 * torch.randn(n, 1, 28, 28, generator=generator)
        x = (prototypes[y] + noise).clamp(0.0, 1.0)
        return x, y

    train_x, train_y = sample(n_train)
    eval_x, eval_y = sample(n_eval)
    return ImageArrays(train_x=train_x, train_y=train_y, eval_x=eval_x, eval_y=eval_y, source="synthetic")


class GlobalFrequency:
    def __init__(self, n_classes: int, alpha: float) -> None:
        self.counts = torch.zeros(n_classes)
        self.alpha = float(alpha)

    def predict(self, n: int) -> torch.Tensor:
        probs = (self.counts + self.alpha) / (self.counts.sum() + self.alpha * len(self.counts))
        return probs.expand(n, -1)

    def observe(self, y: torch.Tensor) -> None:
        for item in y.flatten():
            self.counts[int(item.item())] += 1.0


def nll_and_accuracy(probs: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
    target = probs.gather(1, y[:, None]).squeeze(1).clamp_min(EPS)
    loss = -torch.log(target)
    correct = int((torch.argmax(probs, dim=1) == y).sum().item())
    return loss, correct


@torch.no_grad()
def evaluate_dgm(model: CategoricalDGM, h: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict[str, float]:
    losses = []
    correct = 0
    start = time.perf_counter()
    for begin in range(0, h.shape[0], batch_size):
        hb = h[begin : begin + batch_size]
        yb = y[begin : begin + batch_size]
        probs = model.predict(hb)
        assert isinstance(probs, torch.Tensor)
        loss, batch_correct = nll_and_accuracy(probs, yb)
        losses.append(loss)
        correct += batch_correct
    elapsed = time.perf_counter() - start
    return {
        "nll": float(torch.cat(losses).mean().item()),
        "accuracy": correct / float(h.shape[0]),
        "query_seconds": elapsed,
    }


@torch.no_grad()
def concept_purity(
    model: CategoricalDGM,
    h: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    n_label_values: int,
) -> float:
    if model.num_concepts == 0:
        return 0.0
    counts = torch.zeros(model.num_concepts, n_label_values)
    for begin in range(0, h.shape[0], batch_size):
        hb = h[begin : begin + batch_size]
        lb = labels[begin : begin + batch_size]
        _, aux = model.predict(hb, return_aux=True)
        selected = aux["selected_idx"].cpu()
        for concept, label in zip(selected, lb.cpu(), strict=True):
            idx = int(concept.item())
            if idx >= 0:
                counts[idx, int(label.item())] += 1.0
    total = float(counts.sum().item())
    if total <= 0.0:
        return 0.0
    return float(counts.max(dim=1).values.sum().item() / total)


@torch.no_grad()
def mean_routing_margin(model: CategoricalDGM, h: torch.Tensor, batch_size: int) -> float:
    if model.num_concepts <= 1:
        return 0.0
    margins = []
    for begin in range(0, h.shape[0], batch_size):
        hb = h[begin : begin + batch_size]
        _, aux = model.predict(hb, return_aux=True)
        weights = aux["routing_weights"]
        if weights.shape[1] == 1:
            margins.append(torch.ones(weights.shape[0]))
        else:
            top2 = torch.topk(weights, k=2, dim=1).values
            margins.append(top2[:, 0] - top2[:, 1])
    return float(torch.cat(margins).mean().item())


def run_dgm_prequential(
    h_train: torch.Tensor,
    y_train: torch.Tensor,
    h_eval: torch.Tensor,
    y_eval: torch.Tensor,
    *,
    n_classes: int,
    true_purity_labels: torch.Tensor,
    eval_purity_labels: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    model = CategoricalDGM(
        dim=h_train.shape[1],
        n_classes=n_classes,
        k=args.k,
        alpha=args.alpha,
        distance_temperature=args.distance_temperature,
        edge_temperature=args.edge_temperature,
        edge_weight=args.edge_weight,
        centroid_lr=args.centroid_lr,
        refine_on_error=True,
        max_concepts=args.max_concepts,
    )
    global_freq = GlobalFrequency(n_classes, args.alpha)

    totals = {
        "dgm_nll": 0.0,
        "dgm_correct": 0,
        "global_nll": 0.0,
        "global_correct": 0,
    }
    start = time.perf_counter()
    for h_i, y_i in zip(h_train, y_train, strict=True):
        hb = h_i.unsqueeze(0)
        yb = y_i.reshape(1)
        loss, aux = model.loss(hb, yb, return_aux=True)
        totals["dgm_nll"] += float(loss.item())
        totals["dgm_correct"] += int(torch.argmax(aux["probs"], dim=1).item() == int(y_i.item()))
        model.observe(hb, yb, aux=aux)

        global_probs = global_freq.predict(1)
        global_loss, global_correct = nll_and_accuracy(global_probs, yb)
        totals["global_nll"] += float(global_loss.item())
        totals["global_correct"] += global_correct
        global_freq.observe(yb)
    train_seconds = time.perf_counter() - start

    heldout = evaluate_dgm(model, h_eval, y_eval, args.batch_size)
    return {
        "prequential_nll": totals["dgm_nll"] / float(h_train.shape[0]),
        "prequential_accuracy": totals["dgm_correct"] / float(h_train.shape[0]),
        "heldout_nll": heldout["nll"],
        "heldout_accuracy": heldout["accuracy"],
        "global_prequential_nll": totals["global_nll"] / float(h_train.shape[0]),
        "global_prequential_accuracy": totals["global_correct"] / float(h_train.shape[0]),
        "concepts": float(model.num_concepts),
        "edges": float(model.num_edges),
        "concept_purity_train": concept_purity(model, h_train, true_purity_labels, args.batch_size, 10),
        "concept_purity_eval": concept_purity(model, h_eval, eval_purity_labels, args.batch_size, 10),
        "routing_margin_eval": mean_routing_margin(model, h_eval, args.batch_size),
        "train_seconds": train_seconds,
        "heldout_query_seconds": heldout["query_seconds"],
    }


def make_balanced_mask_task(
    z: torch.Tensor,
    digit: torch.Tensor,
    *,
    seed: int,
    eta: float,
    mask_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    mask = torch.randint(0, 2, (z.shape[0], 10), generator=generator).float()
    y = mask.gather(1, digit[:, None]).squeeze(1).long()
    if eta > 0.0:
        flips = torch.rand(z.shape[0], generator=generator) < eta
        y[flips] = 1 - y[flips]
    signed_mask = (2.0 * mask - 1.0) * float(mask_scale)
    return torch.cat([z, signed_mask], dim=1), y, mask


def run(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    if args.quick:
        args.n_train = min(args.n_train, 1200)
        args.n_eval = min(args.n_eval, 500)
        args.embedding_dim = min(args.embedding_dim, 48)
        args.max_concepts = min(args.max_concepts, 256)

    arrays = load_mnist_arrays(
        args.data_root,
        download=args.download,
        synthetic=args.synthetic,
        n_train=args.n_train,
        n_eval=args.n_eval,
        seed=args.seed,
    )
    encoder = RandomProjectionEncoder(args.embedding_dim, seed=args.seed + 17)
    embed_start = time.perf_counter()
    z_train = encoder.transform(arrays.train_x, args.batch_size)
    z_eval = encoder.transform(arrays.eval_x, args.batch_size)
    embed_seconds = time.perf_counter() - embed_start

    digit_metrics = run_dgm_prequential(
        z_train,
        arrays.train_y,
        z_eval,
        arrays.eval_y,
        n_classes=10,
        true_purity_labels=arrays.train_y,
        eval_purity_labels=arrays.eval_y,
        args=args,
    )

    hqs_train, hqs_y_train, _ = make_balanced_mask_task(
        z_train,
        arrays.train_y,
        seed=args.seed + 101,
        eta=args.eta,
        mask_scale=args.mask_scale,
    )
    hqs_eval, hqs_y_eval, _ = make_balanced_mask_task(
        z_eval,
        arrays.eval_y,
        seed=args.seed + 202,
        eta=args.eta,
        mask_scale=args.mask_scale,
    )
    hqs_metrics = run_dgm_prequential(
        hqs_train,
        hqs_y_train,
        hqs_eval,
        hqs_y_eval,
        n_classes=2,
        true_purity_labels=arrays.train_y,
        eval_purity_labels=arrays.eval_y,
        args=args,
    )

    return {
        "description": "MNIST smoke tests for the PyTorch CategoricalDGM reference implementation.",
        "source": arrays.source,
        "seed": args.seed,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "embedding_dim": args.embedding_dim,
        "max_concepts": args.max_concepts,
        "alpha": args.alpha,
        "k": args.k,
        "eta": args.eta,
        "embed_seconds": embed_seconds,
        "tasks": {
            "one_pass_digit_classification": digit_metrics,
            "image_balanced_mask_hqs": hqs_metrics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("experiments/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/mnist_dgm_smoke_results"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-concepts", type=int, default=512)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--distance-temperature", type=float, default=0.35)
    parser.add_argument("--edge-temperature", type=float, default=8.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--centroid-lr", type=float, default=0.05)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--mask-scale", type=float, default=0.35)
    args = parser.parse_args()

    results = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print(f"MNIST DGM smoke source={results['source']} n_train={results['n_train']} n_eval={results['n_eval']}")
    for task, row in results["tasks"].items():
        print(
            f"  {task}: heldout_nll={row['heldout_nll']:.3f}, "
            f"heldout_acc={row['heldout_accuracy']:.3f}, "
            f"concepts={row['concepts']:.0f}, edges={row['edges']:.0f}, "
            f"purity={row['concept_purity_eval']:.3f}"
        )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

