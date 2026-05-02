from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.dgm_reference import CategoricalDGM


NUM_FINE = 100
NUM_COARSE = 20
EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


@dataclass
class CIFAR100Images:
    train_x: torch.Tensor
    train_fine: torch.Tensor
    train_coarse: torch.Tensor
    eval_x: torch.Tensor
    eval_fine: torch.Tensor
    eval_coarse: torch.Tensor
    fine_names: list[str]
    coarse_names: list[str]
    fine_to_coarse: list[int]


class RandomProjectionImageEncoder:
    def __init__(self, in_channels: int, image_size: int, embedding_dim: int, seed: int, pool: int = 4) -> None:
        self.pool = int(pool)
        generator = torch.Generator().manual_seed(seed)
        pooled = image_size // self.pool
        input_dim = in_channels * pooled * pooled
        proj = torch.randn(input_dim, embedding_dim, generator=generator)
        self.proj = proj / math.sqrt(float(input_dim))

    def transform(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
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


class SupervisedPrototypeEncoder:
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        embedding_dim: int,
        seed: int,
        pool: int,
        n_classes: int,
    ) -> None:
        self.base = RandomProjectionImageEncoder(
            in_channels=in_channels,
            image_size=image_size,
            embedding_dim=embedding_dim,
            seed=seed,
            pool=pool,
        )
        self.n_classes = int(n_classes)
        self.prototypes: torch.Tensor | None = None

    def fit(self, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> None:
        z = self.base.transform(x, batch_size)
        proto = torch.zeros(self.n_classes, z.shape[1])
        counts = torch.zeros(self.n_classes, 1)
        for cls in range(self.n_classes):
            mask = y == cls
            if bool(mask.any()):
                proto[cls] = z[mask].mean(dim=0)
                counts[cls] = float(mask.sum().item())
        missing = counts.squeeze(1) == 0
        if bool(missing.any()):
            proto[missing] = torch.randn(int(missing.sum().item()), z.shape[1])
        self.prototypes = F.normalize(proto, dim=1)

    def transform(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.prototypes is None:
            raise RuntimeError("SupervisedPrototypeEncoder must be fit before transform")
        chunks = []
        for start in range(0, x.shape[0], batch_size):
            z = self.base.transform(x[start : start + batch_size], batch_size)
            sims = z @ self.prototypes.T
            chunks.append(F.normalize(sims, dim=1))
        return torch.cat(chunks, dim=0)


class GlobalFrequency:
    def __init__(self, alpha: float = 1.0) -> None:
        self.counts = torch.zeros(2)
        self.alpha = float(alpha)

    def predict(self, n: int) -> torch.Tensor:
        p = (self.counts + self.alpha) / (self.counts.sum() + 2.0 * self.alpha)
        return p.expand(n, -1)

    def observe(self, y: torch.Tensor) -> None:
        for item in y.flatten():
            self.counts[int(item.item())] += 1.0


class OnlineLogistic:
    def __init__(self, dim: int, lr: float = 0.4, l2: float = 1e-5) -> None:
        self.w = torch.zeros(dim)
        self.b = torch.zeros(())
        self.g_w = torch.full((dim,), 1e-8)
        self.g_b = torch.tensor(1e-8)
        self.lr = float(lr)
        self.l2 = float(l2)

    def predict_prob(self, h: torch.Tensor) -> torch.Tensor:
        logit = h @ self.w + self.b
        p1 = torch.sigmoid(logit)
        return torch.stack([1.0 - p1, p1], dim=1).clamp_min(EPS)

    def observe(self, h: torch.Tensor, y: torch.Tensor) -> None:
        p1 = self.predict_prob(h)[:, 1]
        grad = p1 - y.float()
        grad_w = (grad[:, None] * h).mean(dim=0) + self.l2 * self.w
        grad_b = grad.mean()
        self.g_w += grad_w.square()
        self.g_b += grad_b.square()
        self.w -= self.lr * grad_w / torch.sqrt(self.g_w)
        self.b -= self.lr * grad_b / torch.sqrt(self.g_b)


def cifar_tensor(raw: np.ndarray) -> torch.Tensor:
    x = raw.reshape(-1, 3, 32, 32).astype("float32") / 255.0
    return torch.from_numpy(x)


def load_cifar100(root: Path) -> CIFAR100Images:
    base = root / "cifar-100-python"
    if not base.exists():
        raise FileNotFoundError(
            f"missing {base}; run an existing CIFAR-100 downloader or place the extracted archive under experiments/data"
        )
    train_raw = _load_pickle(base / "train")
    eval_raw = _load_pickle(base / "test")
    meta = _load_pickle(base / "meta")

    train_fine = torch.tensor(train_raw["fine_labels"], dtype=torch.long)
    train_coarse = torch.tensor(train_raw["coarse_labels"], dtype=torch.long)
    eval_fine = torch.tensor(eval_raw["fine_labels"], dtype=torch.long)
    eval_coarse = torch.tensor(eval_raw["coarse_labels"], dtype=torch.long)
    fine_to_coarse = [-1 for _ in range(NUM_FINE)]
    for fine, coarse in zip(train_fine.tolist(), train_coarse.tolist(), strict=True):
        if fine_to_coarse[fine] == -1:
            fine_to_coarse[fine] = coarse
        elif fine_to_coarse[fine] != coarse:
            raise ValueError(f"inconsistent fine-to-coarse mapping for fine label {fine}")
    if any(item < 0 for item in fine_to_coarse):
        raise ValueError("incomplete CIFAR-100 fine-to-coarse mapping")

    return CIFAR100Images(
        train_x=cifar_tensor(train_raw["data"]),
        train_fine=train_fine,
        train_coarse=train_coarse,
        eval_x=cifar_tensor(eval_raw["data"]),
        eval_fine=eval_fine,
        eval_coarse=eval_coarse,
        fine_names=list(meta["fine_label_names"]),
        coarse_names=list(meta["coarse_label_names"]),
        fine_to_coarse=fine_to_coarse,
    )


def sample_indices(labels: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(labels.shape[0], generator=generator)
    return perm[: min(n, labels.shape[0])]


def make_image_hqs(
    z: torch.Tensor,
    coarse: torch.Tensor,
    *,
    seed: int,
    eta: float,
    mask_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    mask = torch.randint(0, 2, (z.shape[0], NUM_COARSE), generator=generator).float()
    y = mask.gather(1, coarse[:, None]).squeeze(1).long()
    if eta > 0.0:
        flips = torch.rand(z.shape[0], generator=generator) < eta
        y[flips] = 1 - y[flips]
    signed_mask = (2.0 * mask - 1.0) * float(mask_scale)
    h = torch.cat([z, signed_mask], dim=1)
    return h, y, mask


def nll_and_correct(probs: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
    target = probs.gather(1, y[:, None]).squeeze(1).clamp_min(EPS)
    nll = -torch.log(target)
    correct = int((torch.argmax(probs, dim=1) == y).sum().item())
    return nll, correct


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
        loss, batch_correct = nll_and_correct(probs, yb)
        losses.append(loss)
        correct += batch_correct
    elapsed = time.perf_counter() - start
    return {
        "nll": float(torch.cat(losses).mean().item()),
        "accuracy": correct / float(h.shape[0]),
        "seconds": elapsed,
    }


@torch.no_grad()
def evaluate_logistic(model: OnlineLogistic, h: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict[str, float]:
    losses = []
    correct = 0
    start = time.perf_counter()
    for begin in range(0, h.shape[0], batch_size):
        hb = h[begin : begin + batch_size]
        yb = y[begin : begin + batch_size]
        probs = model.predict_prob(hb)
        loss, batch_correct = nll_and_correct(probs, yb)
        losses.append(loss)
        correct += batch_correct
    elapsed = time.perf_counter() - start
    return {
        "nll": float(torch.cat(losses).mean().item()),
        "accuracy": correct / float(h.shape[0]),
        "seconds": elapsed,
    }


@torch.no_grad()
def concept_purity(
    model: CategoricalDGM,
    h: torch.Tensor,
    labels: torch.Tensor,
    n_values: int,
    batch_size: int,
) -> float:
    if model.num_concepts == 0:
        return 0.0
    counts = torch.zeros(model.num_concepts, n_values)
    for begin in range(0, h.shape[0], batch_size):
        hb = h[begin : begin + batch_size]
        lb = labels[begin : begin + batch_size]
        _, aux = model.predict(hb, return_aux=True)
        for concept, label in zip(aux["selected_idx"].cpu(), lb.cpu(), strict=True):
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
        _, aux = model.predict(h[begin : begin + batch_size], return_aux=True)
        weights = aux["routing_weights"]
        if weights.shape[1] == 1:
            margins.append(torch.ones(weights.shape[0]))
        else:
            top2 = torch.topk(weights, k=2, dim=1).values
            margins.append(top2[:, 0] - top2[:, 1])
    return float(torch.cat(margins).mean().item())


def edge_mask_fraction(model: CategoricalDGM, image_dim: int) -> float:
    if model.num_edges == 0:
        return 0.0
    edge_u = model.edge_u.detach()
    image_norm = torch.linalg.vector_norm(edge_u[:, :image_dim], dim=1)
    mask_norm = torch.linalg.vector_norm(edge_u[:, image_dim:], dim=1)
    return float((mask_norm / (image_norm + mask_norm + EPS)).mean().item())


def run_prequential(
    h_train: torch.Tensor,
    y_train: torch.Tensor,
    h_eval: torch.Tensor,
    y_eval: torch.Tensor,
    fine_train: torch.Tensor,
    coarse_train: torch.Tensor,
    fine_eval: torch.Tensor,
    coarse_eval: torch.Tensor,
    image_dim: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dgm = CategoricalDGM(
        dim=h_train.shape[1],
        n_classes=2,
        k=args.k,
        alpha=args.alpha,
        distance_temperature=args.distance_temperature,
        edge_temperature=args.edge_temperature,
        edge_weight=args.edge_weight,
        centroid_lr=args.centroid_lr,
        refine_on_error=True,
        max_concepts=args.max_concepts,
    )
    logistic = OnlineLogistic(h_train.shape[1], lr=args.logistic_lr)
    global_freq = GlobalFrequency(args.alpha)
    totals = {
        "dgm_nll": 0.0,
        "dgm_correct": 0,
        "logistic_nll": 0.0,
        "logistic_correct": 0,
        "global_nll": 0.0,
        "global_correct": 0,
    }

    start = time.perf_counter()
    for h_i, y_i in zip(h_train, y_train, strict=True):
        hb = h_i.unsqueeze(0)
        yb = y_i.reshape(1)

        dgm_loss, aux = dgm.loss(hb, yb, return_aux=True)
        totals["dgm_nll"] += float(dgm_loss.item())
        totals["dgm_correct"] += int(torch.argmax(aux["probs"], dim=1).item() == int(y_i.item()))
        dgm.observe(hb, yb, aux=aux)

        logistic_probs = logistic.predict_prob(hb)
        logistic_loss, logistic_correct = nll_and_correct(logistic_probs, yb)
        totals["logistic_nll"] += float(logistic_loss.item())
        totals["logistic_correct"] += logistic_correct
        logistic.observe(hb, yb)

        global_probs = global_freq.predict(1)
        global_loss, global_correct = nll_and_correct(global_probs, yb)
        totals["global_nll"] += float(global_loss.item())
        totals["global_correct"] += global_correct
        global_freq.observe(yb)
    train_seconds = time.perf_counter() - start

    dgm_eval = evaluate_dgm(dgm, h_eval, y_eval, args.batch_size)
    logistic_eval = evaluate_logistic(logistic, h_eval, y_eval, args.batch_size)
    n_train = float(h_train.shape[0])
    return {
        "dgm": {
            "prequential_nll": totals["dgm_nll"] / n_train,
            "prequential_accuracy": totals["dgm_correct"] / n_train,
            "heldout_nll": dgm_eval["nll"],
            "heldout_accuracy": dgm_eval["accuracy"],
            "concepts": dgm.num_concepts,
            "edges": dgm.num_edges,
            "fine_purity_train": concept_purity(dgm, h_train, fine_train, NUM_FINE, args.batch_size),
            "coarse_purity_train": concept_purity(dgm, h_train, coarse_train, NUM_COARSE, args.batch_size),
            "fine_purity_eval": concept_purity(dgm, h_eval, fine_eval, NUM_FINE, args.batch_size),
            "coarse_purity_eval": concept_purity(dgm, h_eval, coarse_eval, NUM_COARSE, args.batch_size),
            "routing_margin_eval": mean_routing_margin(dgm, h_eval, args.batch_size),
            "edge_mask_fraction": edge_mask_fraction(dgm, image_dim),
            "heldout_query_seconds": dgm_eval["seconds"],
        },
        "online_logistic": {
            "prequential_nll": totals["logistic_nll"] / n_train,
            "prequential_accuracy": totals["logistic_correct"] / n_train,
            "heldout_nll": logistic_eval["nll"],
            "heldout_accuracy": logistic_eval["accuracy"],
            "heldout_query_seconds": logistic_eval["seconds"],
        },
        "global_frequency": {
            "prequential_nll": totals["global_nll"] / n_train,
            "prequential_accuracy": totals["global_correct"] / n_train,
        },
        "train_seconds": train_seconds,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    if args.quick:
        args.n_train = min(args.n_train, 2000)
        args.n_eval = min(args.n_eval, 800)
        args.embedding_dim = min(args.embedding_dim, 64)
        args.max_concepts = min(args.max_concepts, 256)
        args.prototype_fit = min(args.prototype_fit, 10000)

    data = load_cifar100(args.data_root)
    train_idx = sample_indices(data.train_fine, args.n_train, args.seed + 11)
    eval_idx = sample_indices(data.eval_fine, args.n_eval, args.seed + 29)
    train_x = data.train_x[train_idx]
    train_fine = data.train_fine[train_idx]
    train_coarse = data.train_coarse[train_idx]
    eval_x = data.eval_x[eval_idx]
    eval_fine = data.eval_fine[eval_idx]
    eval_coarse = data.eval_coarse[eval_idx]

    embed_start = time.perf_counter()
    if args.encoder == "random_projection":
        encoder = RandomProjectionImageEncoder(
            in_channels=3,
            image_size=32,
            embedding_dim=args.embedding_dim,
            seed=args.seed + 101,
            pool=args.pool,
        )
    elif args.encoder == "prototype":
        encoder = SupervisedPrototypeEncoder(
            in_channels=3,
            image_size=32,
            embedding_dim=args.embedding_dim,
            seed=args.seed + 101,
            pool=args.pool,
            n_classes=NUM_FINE,
        )
        fit_idx = sample_indices(data.train_fine, args.prototype_fit, args.seed + 71)
        encoder.fit(data.train_x[fit_idx], data.train_fine[fit_idx], args.batch_size)
    else:
        raise ValueError(f"unknown encoder: {args.encoder}")
    z_train = encoder.transform(train_x, args.batch_size)
    z_eval = encoder.transform(eval_x, args.batch_size)
    embed_seconds = time.perf_counter() - embed_start

    h_train, y_train, _ = make_image_hqs(
        z_train,
        train_coarse,
        seed=args.seed + 303,
        eta=args.eta,
        mask_scale=args.mask_scale,
    )
    h_eval, y_eval, _ = make_image_hqs(
        z_eval,
        eval_coarse,
        seed=args.seed + 404,
        eta=args.eta,
        mask_scale=args.mask_scale,
    )
    metrics = run_prequential(
        h_train,
        y_train,
        h_eval,
        y_eval,
        train_fine,
        train_coarse,
        eval_fine,
        eval_coarse,
        image_dim=z_train.shape[1],
        args=args,
    )

    return {
        "description": "CIFAR-100 Image-BalancedMask-HQS with image embeddings plus query masks; no oracle fine label is exposed to the learner.",
        "seed": args.seed,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "encoder": args.encoder,
        "embedding_dim": args.embedding_dim,
        "prototype_fit": args.prototype_fit if args.encoder == "prototype" else 0,
        "eta": args.eta,
        "mask_scale": args.mask_scale,
        "max_concepts": args.max_concepts,
        "k": args.k,
        "alpha": args.alpha,
        "embed_seconds": embed_seconds,
        "fine_names": data.fine_names,
        "coarse_names": data.coarse_names,
        "fine_to_coarse": data.fine_to_coarse,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("experiments/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/cifar100_image_hqs_results"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--encoder", choices=["prototype", "random_projection"], default="prototype")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--prototype-fit", type=int, default=50000)
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-concepts", type=int, default=768)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--distance-temperature", type=float, default=0.35)
    parser.add_argument("--edge-temperature", type=float, default=8.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--centroid-lr", type=float, default=0.04)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--mask-scale", type=float, default=0.35)
    parser.add_argument("--logistic-lr", type=float, default=0.4)
    args = parser.parse_args()

    results = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print(f"CIFAR-100 Image-BalancedMask-HQS n_train={results['n_train']} n_eval={results['n_eval']}")
    dgm = results["metrics"]["dgm"]
    logistic = results["metrics"]["online_logistic"]
    print(
        f"  dgm: heldout_nll={dgm['heldout_nll']:.3f}, heldout_acc={dgm['heldout_accuracy']:.3f}, "
        f"concepts={dgm['concepts']}, edges={dgm['edges']}, coarse_purity={dgm['coarse_purity_eval']:.3f}"
    )
    print(
        f"  online_logistic: heldout_nll={logistic['heldout_nll']:.3f}, "
        f"heldout_acc={logistic['heldout_accuracy']:.3f}"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
