from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets
from torch.nn import functional as F


MNIST_ROOT = "/Users/gunale/works/silifen-works/haoran_idea/data"
CIFAR10_ROOT = "/Users/gunale/works/silifen-works/SLN/data/cifar10"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_arrays(name: str, root: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if name == "MNIST":
        train = datasets.MNIST(root=root, train=True, download=False)
        test = datasets.MNIST(root=root, train=False, download=False)
        x_train = train.data.float().unsqueeze(1) / 255.0
        x_test = test.data.float().unsqueeze(1) / 255.0
        mean = torch.tensor([0.1307]).view(1, 1, 1, 1)
        std = torch.tensor([0.3081]).view(1, 1, 1, 1)
        return (x_train - mean) / std, train.targets.long(), (x_test - mean) / std, test.targets.long()
    if name == "CIFAR10":
        train = datasets.CIFAR10(root=root, train=True, download=False)
        test = datasets.CIFAR10(root=root, train=False, download=False)
        x_train = torch.tensor(np.asarray(train.data).transpose(0, 3, 1, 2)).float() / 255.0
        x_test = torch.tensor(np.asarray(test.data).transpose(0, 3, 1, 2)).float() / 255.0
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        return (x_train - mean) / std, torch.tensor(train.targets).long(), (x_test - mean) / std, torch.tensor(test.targets).long()
    raise ValueError(name)


def stratified_indices(y: torch.Tensor, per_class: int, rng: np.random.Generator) -> torch.Tensor:
    out: list[int] = []
    for cls in torch.unique(y).tolist():
        idx = torch.nonzero(y == int(cls), as_tuple=False).flatten().numpy()
        rng.shuffle(idx)
        if len(idx) < per_class:
            raise ValueError(f"class {cls} has {len(idx)} examples, need {per_class}")
        out.extend(idx[:per_class].tolist())
    rng.shuffle(out)
    return torch.tensor(out, dtype=torch.long)


def split_pool(
    y: torch.Tensor,
    fit_per_class: int,
    support_pool_per_class: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    fit: list[int] = []
    pool: list[int] = []
    for cls in torch.unique(y).tolist():
        idx = torch.nonzero(y == int(cls), as_tuple=False).flatten().numpy()
        rng.shuffle(idx)
        need = fit_per_class + support_pool_per_class
        if len(idx) < need:
            raise ValueError(f"class {cls} has {len(idx)} examples, need {need}")
        fit.extend(idx[:fit_per_class].tolist())
        pool.extend(idx[fit_per_class:need].tolist())
    rng.shuffle(fit)
    rng.shuffle(pool)
    return torch.tensor(fit, dtype=torch.long), torch.tensor(pool, dtype=torch.long)


def augment_images(x: torch.Tensor, spec: str) -> torch.Tensor:
    if spec == "hflip":
        return torch.flip(x, dims=[3])
    if spec == "shift2r":
        return torch.roll(x, shifts=2, dims=3)
    if spec == "shift2l":
        return torch.roll(x, shifts=-2, dims=3)
    if spec == "shift2d":
        return torch.roll(x, shifts=2, dims=2)
    if spec == "shift2u":
        return torch.roll(x, shifts=-2, dims=2)
    raise ValueError(f"unknown augmentation: {spec}")


def parse_aug_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


class FixedFeatureMap:
    def __init__(self, mode: str, dim: int, seed: int) -> None:
        self.mode = mode
        self.dim = dim
        self.seed = seed
        self.mean: torch.Tensor | None = None
        self.proj: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.dct_h: torch.Tensor | None = None
        self.dct_w: torch.Tensor | None = None

    @staticmethod
    def _dct_matrix(n: int, keep: int) -> torch.Tensor:
        x = torch.arange(n, dtype=torch.float32)
        k = torch.arange(keep, dtype=torch.float32).unsqueeze(1)
        basis = torch.cos(math.pi * (x + 0.5) * k / float(n))
        basis[0] = basis[0] / math.sqrt(float(n))
        if keep > 1:
            basis[1:] = basis[1:] * math.sqrt(2.0 / float(n))
        return basis

    def _handcrafted(self, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].float()
            gray = xb.mean(dim=1, keepdim=True)
            gx = F.conv2d(gray, sobel_x, padding=1)
            gy = F.conv2d(gray, sobel_y, padding=1)
            mag = torch.sqrt(gx.square() + gy.square() + 1.0e-8)
            angle = torch.atan2(gy, gx) + math.pi
            bins = torch.clamp((angle * (8.0 / (2.0 * math.pi))).long(), max=7)
            hist_parts = []
            for b in range(8):
                hist_parts.append(F.avg_pool2d(mag * (bins == b).float(), kernel_size=4, stride=4))
            hog = torch.cat(hist_parts, dim=1).flatten(1)

            color_mean_4 = F.avg_pool2d(xb, kernel_size=4, stride=4).flatten(1)
            color_sq_4 = F.avg_pool2d(xb.square(), kernel_size=4, stride=4).flatten(1)
            color_std_4 = torch.sqrt((color_sq_4 - color_mean_4.square()).clamp_min(0.0) + 1.0e-6)
            color_mean_8 = F.avg_pool2d(xb, kernel_size=8, stride=8).flatten(1)
            gray_low = F.avg_pool2d(gray, kernel_size=2, stride=2).flatten(1)
            chunks.append(torch.cat([hog, color_mean_4, color_std_4, color_mean_8, gray_low], dim=1))
        return torch.cat(chunks, dim=0)

    def _dct_lowfreq(self, xb: torch.Tensor, keep: int = 8) -> torch.Tensor:
        if self.dct_h is None or self.dct_w is None:
            self.dct_h = self._dct_matrix(int(xb.shape[2]), keep)
            self.dct_w = self._dct_matrix(int(xb.shape[3]), keep)
        coeff = torch.einsum("kh,bchw,lw->bckl", self.dct_h, xb, self.dct_w)
        gray = xb.mean(dim=1, keepdim=True)
        gray_coeff = torch.einsum("kh,bchw,lw->bckl", self.dct_h, gray, self.dct_w)
        return torch.cat([coeff.flatten(1), gray_coeff.flatten(1)], dim=1)

    def _lbp_hist(self, xb: torch.Tensor) -> torch.Tensor:
        gray = xb.mean(dim=1, keepdim=True)
        center = gray[:, :, 1:-1, 1:-1]
        neighbors = [
            gray[:, :, :-2, :-2],
            gray[:, :, :-2, 1:-1],
            gray[:, :, :-2, 2:],
            gray[:, :, 1:-1, 2:],
            gray[:, :, 2:, 2:],
            gray[:, :, 2:, 1:-1],
            gray[:, :, 2:, :-2],
            gray[:, :, 1:-1, :-2],
        ]
        code = torch.zeros_like(center, dtype=torch.long)
        for bit, neighbor in enumerate(neighbors):
            code += ((neighbor >= center).long() << bit)
        code = code[:, :, :28, :28]
        cells = code.unfold(2, 7, 7).unfold(3, 7, 7)
        hist_parts = []
        for i in range(4):
            for j in range(4):
                cell = cells[:, :, i, j].flatten(1)
                one_hot = F.one_hot(cell, num_classes=256).float()
                hist = one_hot.mean(dim=1)
                hist_parts.append(hist)
        return torch.cat(hist_parts, dim=1)

    def _vision2(self, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        base = self._handcrafted(x, batch_size=batch_size)
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].float()
            gray = xb.mean(dim=1, keepdim=True)
            gx = F.avg_pool2d(torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]), kernel_size=4, stride=4).flatten(1)
            gy = F.avg_pool2d(torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]), kernel_size=4, stride=4).flatten(1)
            dct = self._dct_lowfreq(xb, keep=8)
            lbp = self._lbp_hist(xb)
            color_global = torch.cat(
                [
                    xb.mean(dim=(2, 3)),
                    xb.std(dim=(2, 3)),
                    xb.amin(dim=(2, 3)),
                    xb.amax(dim=(2, 3)),
                ],
                dim=1,
            )
            local = torch.cat([gx, gy, dct, lbp, color_global], dim=1)
            chunks.append(local)
        return torch.cat([base, torch.cat(chunks, dim=0)], dim=1)

    def _base(self, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        if self.mode in {"raw", "random", "pca"}:
            return x.flatten(1).float()
        if self.mode in {"handcraft", "handcraft_pca"}:
            return self._handcrafted(x, batch_size=batch_size)
        if self.mode in {"vision2", "vision2_pca"}:
            return self._vision2(x, batch_size=batch_size)
        raise ValueError(f"unknown feature mode: {self.mode}")

    def fit(self, x: torch.Tensor) -> None:
        base = self._base(x)
        self.mean = base.mean(dim=0, keepdim=True)
        centered = base - self.mean
        if self.mode in {"handcraft", "handcraft_pca", "vision2", "vision2_pca"}:
            self.scale = centered.std(dim=0, keepdim=True).clamp_min(1.0e-4)
            centered = centered / self.scale
        if self.mode in {"raw", "handcraft", "vision2"}:
            self.proj = None
        elif self.mode == "random":
            generator = torch.Generator().manual_seed(self.seed)
            proj = torch.randn(centered.shape[1], self.dim, generator=generator)
            self.proj = proj / math.sqrt(float(centered.shape[1]))
        elif self.mode in {"pca", "handcraft_pca", "vision2_pca"}:
            # Unsupervised PCA. This is not DGM learning; it is a fixed
            # non-label feature map fitted before the online/read-write phase.
            _, _, vh = torch.pca_lowrank(centered, q=min(self.dim, centered.shape[0], centered.shape[1]), center=False)
            self.proj = vh[:, : self.dim].contiguous()
        else:
            raise ValueError(f"unknown feature mode: {self.mode}")

    def transform(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.mean is None:
            raise RuntimeError("feature map has not been fit")
        chunks: list[torch.Tensor] = []
        for start in range(0, x.shape[0], batch_size):
            base = self._base(x[start : start + batch_size], batch_size=batch_size)
            z = base - self.mean
            if self.scale is not None:
                z = z / self.scale
            if self.proj is not None:
                z = z @ self.proj
            chunks.append(nn.functional.normalize(z, dim=1))
        return torch.cat(chunks, dim=0)


def knn_predict(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    query_z: torch.Tensor,
    n_classes: int,
    k: int,
    temperature: float,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    support_z = nn.functional.normalize(support_z, dim=1)
    query_z = nn.functional.normalize(query_z, dim=1)
    probs: list[torch.Tensor] = []
    for start in range(0, query_z.shape[0], batch_size):
        q = query_z[start : start + batch_size]
        sim = q @ support_z.T
        kk = min(k, support_z.shape[0])
        vals, idx = torch.topk(sim, k=kk, dim=1)
        weights = torch.softmax(vals / temperature, dim=1)
        p = torch.full((q.shape[0], n_classes), 1.0e-6)
        p.scatter_add_(1, support_y[idx], weights)
        p = p / p.sum(dim=1, keepdim=True)
        probs.append(p)
    p_all = torch.cat(probs, dim=0)
    return torch.argmax(p_all, dim=1), p_all


def ridge_predict(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    query_z: torch.Tensor,
    n_classes: int,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    support_z = nn.functional.normalize(support_z, dim=1)
    query_z = nn.functional.normalize(query_z, dim=1)
    y_onehot = nn.functional.one_hot(support_y, num_classes=n_classes).float()
    if support_z.shape[0] <= support_z.shape[1]:
        kernel = support_z @ support_z.T
        kernel = kernel + ridge * torch.eye(kernel.shape[0])
        alpha = torch.linalg.solve(kernel, y_onehot)
        logits = (query_z @ support_z.T) @ alpha
    else:
        xtx = support_z.T @ support_z
        w = torch.linalg.solve(xtx + ridge * torch.eye(xtx.shape[0]), support_z.T @ y_onehot)
        logits = query_z @ w
    probs = torch.softmax(logits, dim=1)
    return torch.argmax(probs, dim=1), probs


def prototype_predict(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    query_z: torch.Tensor,
    n_classes: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    protos = []
    for cls in range(n_classes):
        group = support_z[support_y == cls]
        protos.append(group.mean(dim=0))
    proto = nn.functional.normalize(torch.stack(protos, dim=0), dim=1)
    query_z = nn.functional.normalize(query_z, dim=1)
    logits = (query_z @ proto.T) / temperature
    probs = torch.softmax(logits, dim=1)
    return torch.argmax(probs, dim=1), probs


def full_dgm_predict(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    query_views: list[torch.Tensor],
    n_classes: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    from run_real_training_benchmark import FullDGMRefineMemory

    memory = FullDGMRefineMemory(
        n_classes=n_classes,
        embedding_dim=int(support_z.shape[1]),
        max_concepts=args.full_dgm_max_concepts,
        buffer_size=args.full_dgm_buffer_size,
        top_k=args.full_dgm_top_k,
        temperature=args.full_dgm_temperature,
        alpha=1.0,
        min_buffer=args.full_dgm_min_buffer,
        min_child=args.full_dgm_min_child,
        split_penalty=args.full_dgm_split_penalty,
        max_splits_per_batch=args.full_dgm_max_splits_per_batch,
        device=support_z.device,
    )
    for start in range(0, support_z.shape[0], args.batch_size):
        memory.observe(support_z[start : start + args.batch_size], support_y[start : start + args.batch_size])
    probs = torch.stack([torch.softmax(memory.logits(query_z), dim=1) for query_z in query_views], dim=0).mean(dim=0)
    return torch.argmax(probs, dim=1), probs, memory.concepts, memory.estimated_bytes


def train_head(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    query_z: torch.Tensor,
    n_classes: int,
    steps: int,
    lr: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    torch.manual_seed(seed)
    head = nn.Linear(support_z.shape[1], n_classes)
    opt = torch.optim.SGD(head.parameters(), lr=lr)
    start = time.perf_counter()
    for _ in range(steps):
        logits = head(support_z)
        loss = nn.functional.cross_entropy(logits, support_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = torch.softmax(head(query_z), dim=1)
    return torch.argmax(probs, dim=1), probs, time.perf_counter() - start


def metrics(pred: torch.Tensor, probs: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    acc = float((pred == y).float().mean().item())
    nll = float((-torch.log(probs.gather(1, y[:, None]).squeeze(1).clamp_min(1.0e-12))).mean().item())
    return acc, nll


@dataclass
class Row:
    dataset: str
    feature: str
    shots: int
    seed: int
    method: str
    accuracy: float
    nll: float
    seconds: float
    memory_items: int
    feature_dim: int


def run_episode(
    dataset: str,
    feature: str,
    shots: int,
    seed: int,
    fmap: FixedFeatureMap,
    pool_x: torch.Tensor,
    pool_z: torch.Tensor,
    pool_y: torch.Tensor,
    test_x: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    args: argparse.Namespace,
) -> list[Row]:
    rng = np.random.default_rng(seed)
    support_idx: list[int] = []
    for cls in range(args.n_classes):
        idx = torch.nonzero(pool_y == cls, as_tuple=False).flatten().numpy()
        rng.shuffle(idx)
        support_idx.extend(idx[:shots].tolist())
    rng.shuffle(support_idx)
    support_idx_t = torch.tensor(support_idx, dtype=torch.long)
    support_z = pool_z[support_idx_t].contiguous()
    support_y = pool_y[support_idx_t].contiguous()
    if args.support_aug:
        support_x = pool_x[support_idx_t].contiguous()
        views = [support_z]
        labels = [support_y]
        for spec in parse_aug_list(args.support_aug):
            aug_x = augment_images(support_x, spec)
            views.append(fmap.transform(aug_x, batch_size=args.batch_size))
            labels.append(support_y)
        support_z = torch.cat(views, dim=0).contiguous()
        support_y = torch.cat(labels, dim=0).contiguous()

    test_views = [test_z]
    for spec in parse_aug_list(args.query_aug):
        test_views.append(fmap.transform(augment_images(test_x, spec), batch_size=args.batch_size))

    out: list[Row] = []
    for k in args.knn_k:
        start = time.perf_counter()
        probs = torch.stack(
            [
                knn_predict(support_z, support_y, query_z, args.n_classes, k=k, temperature=args.temperature, batch_size=args.batch_size)[1]
                for query_z in test_views
            ],
            dim=0,
        ).mean(dim=0)
        pred = torch.argmax(probs, dim=1)
        seconds = time.perf_counter() - start
        acc, nll = metrics(pred, probs, test_y)
        out.append(Row(dataset, feature, shots, seed, f"PureDGM-kNN{k}", acc, nll, seconds, int(support_y.numel()), int(test_z.shape[1])))

    for ridge in args.ridge:
        start = time.perf_counter()
        probs = torch.stack(
            [ridge_predict(support_z, support_y, query_z, args.n_classes, ridge=ridge)[1] for query_z in test_views],
            dim=0,
        ).mean(dim=0)
        pred = torch.argmax(probs, dim=1)
        seconds = time.perf_counter() - start
        acc, nll = metrics(pred, probs, test_y)
        out.append(Row(dataset, feature, shots, seed, f"PureDGM-ridge{ridge:g}", acc, nll, seconds, int(support_y.numel()), int(test_z.shape[1])))

    start = time.perf_counter()
    probs = torch.stack(
        [prototype_predict(support_z, support_y, query_z, args.n_classes, temperature=args.temperature)[1] for query_z in test_views],
        dim=0,
    ).mean(dim=0)
    pred = torch.argmax(probs, dim=1)
    seconds = time.perf_counter() - start
    acc, nll = metrics(pred, probs, test_y)
    out.append(Row(dataset, feature, shots, seed, "PureDGM-prototype", acc, nll, seconds, args.n_classes, int(test_z.shape[1])))

    if args.full_dgm:
        start = time.perf_counter()
        pred, probs, concepts, memory_bytes = full_dgm_predict(support_z, support_y, test_views, args.n_classes, args)
        seconds = time.perf_counter() - start
        acc, nll = metrics(pred, probs, test_y)
        out.append(Row(dataset, feature, shots, seed, "PureFullDGM-refine", acc, nll, seconds, concepts, int(test_z.shape[1])))

    for steps in args.head_steps:
        start = time.perf_counter()
        pred, probs, head_seconds = train_head(support_z, support_y, test_views[0], args.n_classes, steps=steps, lr=args.head_lr, seed=seed + steps)
        if len(test_views) > 1:
            torch.manual_seed(seed + steps)
            head = nn.Linear(support_z.shape[1], args.n_classes)
            opt = torch.optim.SGD(head.parameters(), lr=args.head_lr)
            for _ in range(steps):
                logits = head(support_z)
                loss = nn.functional.cross_entropy(logits, support_y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                probs = torch.stack([torch.softmax(head(query_z), dim=1) for query_z in test_views], dim=0).mean(dim=0)
                pred = torch.argmax(probs, dim=1)
        seconds = time.perf_counter() - start if len(test_views) > 1 else head_seconds
        acc, nll = metrics(pred, probs, test_y)
        out.append(Row(dataset, feature, shots, seed, f"linear-head-{steps}", acc, nll, seconds, int(support_y.numel()), int(test_z.shape[1])))
    return out


def summarize(rows: list[Row]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    keys = sorted({(r.dataset, r.feature, r.shots, r.method) for r in rows})
    for dataset, feature, shots, method in keys:
        group = [r for r in rows if (r.dataset, r.feature, r.shots, r.method) == (dataset, feature, shots, method)]
        prefix = f"{dataset}/{feature}/{shots}_shot/{method}"
        out[prefix] = {}
        for field in ["accuracy", "nll", "seconds", "memory_items", "feature_dim"]:
            values = np.asarray([float(getattr(r, field)) for r in group], dtype=float)
            out[prefix][f"{field}_mean"] = float(values.mean())
            out[prefix][f"{field}_std"] = float(values.std(ddof=0))
    return out


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="MNIST,CIFAR10")
    parser.add_argument("--features", default="raw,random,pca")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--shots", default="1,5,10,50")
    parser.add_argument("--output", type=Path, default=Path("experiments/results_pure_dgm_benchmark.json"))
    parser.add_argument("--mnist-root", default=MNIST_ROOT)
    parser.add_argument("--cifar-root", default=CIFAR10_ROOT)
    parser.add_argument("--fit-per-class", type=int, default=1000)
    parser.add_argument("--support-pool-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--knn-k", default="1,3,5")
    parser.add_argument("--ridge", default="0.001,0.01,0.1")
    parser.add_argument("--head-steps", default="1,50,200")
    parser.add_argument("--head-lr", type=float, default=0.2)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--full-dgm", action="store_true")
    parser.add_argument("--full-dgm-max-concepts", type=int, default=2048)
    parser.add_argument("--full-dgm-buffer-size", type=int, default=128)
    parser.add_argument("--full-dgm-top-k", type=int, default=12)
    parser.add_argument("--full-dgm-temperature", type=float, default=0.14)
    parser.add_argument("--full-dgm-min-buffer", type=int, default=32)
    parser.add_argument("--full-dgm-min-child", type=int, default=5)
    parser.add_argument("--full-dgm-split-penalty", type=float, default=1.8)
    parser.add_argument("--full-dgm-max-splits-per-batch", type=int, default=6)
    parser.add_argument(
        "--support-aug",
        default="",
        help="Comma-separated pure memory augmentations: hflip,shift2r,shift2l,shift2d,shift2u.",
    )
    parser.add_argument(
        "--query-aug",
        default="",
        help="Comma-separated pure query augmentations averaged at prediction time.",
    )
    args = parser.parse_args()
    args.knn_k = parse_int_list(args.knn_k)
    args.ridge = parse_float_list(args.ridge)
    args.head_steps = parse_int_list(args.head_steps)

    rows: list[Row] = []
    datasets_to_run = [item.strip().upper() for item in args.datasets.split(",") if item.strip()]
    features_to_run = [item.strip() for item in args.features.split(",") if item.strip()]
    seeds = parse_int_list(args.seeds)
    shots_grid = parse_int_list(args.shots)

    for dataset in datasets_to_run:
        root = args.mnist_root if dataset == "MNIST" else args.cifar_root
        x_train, y_train, x_test, y_test = load_arrays(dataset, root)
        base_rng = np.random.default_rng(12345 + len(rows))
        fit_idx, pool_idx = split_pool(y_train, args.fit_per_class, args.support_pool_per_class, base_rng)
        test_idx = stratified_indices(y_test, args.test_per_class, base_rng)
        for feature in features_to_run:
            set_seed(1000 + len(rows))
            fmap = FixedFeatureMap(feature, dim=args.feature_dim, seed=777 + len(rows))
            fit_start = time.perf_counter()
            fmap.fit(x_train[fit_idx])
            pool_x = x_train[pool_idx]
            pool_z = fmap.transform(pool_x, batch_size=args.batch_size)
            test_x = x_test[test_idx]
            test_z = fmap.transform(test_x, batch_size=args.batch_size)
            fit_seconds = time.perf_counter() - fit_start
            pool_y = y_train[pool_idx]
            test_y = y_test[test_idx]
            print(
                f"{dataset} feature={feature} dim={pool_z.shape[1]} fit+embed={fit_seconds:.2f}s "
                f"pool={pool_z.shape[0]} test={test_z.shape[0]}",
                flush=True,
            )
            for shots in shots_grid:
                for seed in seeds:
                    rows.extend(run_episode(dataset, feature, shots, seed, fmap, pool_x, pool_z, pool_y, test_x, test_z, test_y, args))
                best = max((r for r in rows if r.dataset == dataset and r.feature == feature and r.shots == shots), key=lambda r: r.accuracy)
                print(
                    f"  {shots}_shot current-best {best.method} acc={best.accuracy:.3f} nll={best.nll:.3f}",
                    flush=True,
                )
            args.output.with_suffix(args.output.suffix + ".partial").write_text(
                json.dumps({"rows": [asdict(r) for r in rows], "summary": summarize(rows)}, indent=2, sort_keys=True)
            )

    result = {
        "protocol": (
            "Pure no-backprop DGM-style read/write benchmark. Features are raw pixels or fixed "
            "unsupervised/random projections; support examples are stored in nonparametric "
            "DGM memory, while linear heads are gradient baselines on the same fixed features."
        ),
        "hyperparameters": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in {"knn_k", "ridge", "head_steps"}
        },
        "knn_k": args.knn_k,
        "ridge": args.ridge,
        "head_steps": args.head_steps,
        "rows": [asdict(r) for r in rows],
        "summary": summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
