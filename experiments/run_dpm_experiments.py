from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torchvision import datasets


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


@dataclass
class Node:
    depth: int
    n_classes: int
    u: Optional[np.ndarray] = None
    b: float = 0.0
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    indices: list[int] = field(default_factory=list)
    buffer: list[int] = field(default_factory=list)
    counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.counts = np.zeros(self.n_classes, dtype=np.int64)

    @property
    def is_leaf(self) -> bool:
        return self.u is None


class DPMClassifier:
    def __init__(
        self,
        n_classes: int,
        dim: int,
        alpha: float = 1.0,
        lambda_penalty: float = 0.05,
        tau: float = 1e-9,
        min_leaf: int = 16,
        min_child: int = 4,
        buffer_size: int = 96,
        max_depth: int = 8,
        max_candidates: int = 64,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.n_classes = n_classes
        self.dim = dim
        self.alpha = alpha
        self.lambda_penalty = lambda_penalty
        self.tau = tau
        self.min_leaf = min_leaf
        self.min_child = min_child
        self.buffer_size = buffer_size
        self.max_depth = max_depth
        self.max_candidates = max_candidates
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.root = Node(depth=0, n_classes=n_classes)
        self.x_seen: list[np.ndarray] = []
        self.y_seen: list[int] = []
        self.split_records: list[dict[str, float]] = []

    def route(self, h: np.ndarray) -> Node:
        node = self.root
        while not node.is_leaf:
            assert node.u is not None and node.left is not None and node.right is not None
            node = node.right if float(node.u @ h) > node.b else node.left
        return node

    def leaves(self) -> list[Node]:
        out: list[Node] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                out.append(node)
            else:
                assert node.left is not None and node.right is not None
                stack.append(node.left)
                stack.append(node.right)
        return out

    def internal_count(self) -> int:
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not node.is_leaf:
                count += 1
                assert node.left is not None and node.right is not None
                stack.append(node.left)
                stack.append(node.right)
        return count

    def predict_proba(self, h: np.ndarray) -> np.ndarray:
        leaf = self.route(h)
        prior = np.full(self.n_classes, 1.0 / self.n_classes)
        total = int(leaf.counts.sum())
        return (leaf.counts + self.alpha * prior) / (total + self.alpha)

    def predict(self, h: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(h)))

    def observe(self, h: np.ndarray, y: int) -> None:
        idx = len(self.x_seen)
        self.x_seen.append(np.asarray(h, dtype=float))
        self.y_seen.append(int(y))
        leaf = self.route(h)
        self._add_index_to_leaf(leaf, idx)
        if self._split_eligible(leaf):
            candidate = self._best_candidate(leaf)
            if candidate is not None:
                gain, u, b = candidate
                if gain - self.lambda_penalty > self.tau:
                    before = self.objective()
                    self._split_leaf(leaf, u, b)
                    after = self.objective()
                    self.split_records.append(
                        {
                            "gain": float(gain),
                            "objective_before": float(before),
                            "objective_after": float(after),
                            "decrease": float(before - after),
                            "depth": float(leaf.depth),
                        }
                    )

    def objective(self) -> float:
        risk = 0.0
        for leaf in self.leaves():
            n = len(leaf.indices)
            risk += n * entropy_from_counts(leaf.counts)
        return risk + self.lambda_penalty * self.internal_count()

    def _add_index_to_leaf(self, leaf: Node, idx: int) -> None:
        y = self.y_seen[idx]
        leaf.indices.append(idx)
        leaf.counts[y] += 1
        if len(leaf.buffer) < self.buffer_size:
            leaf.buffer.append(idx)
        else:
            j = int(self.rng.integers(0, len(leaf.indices)))
            if j < self.buffer_size:
                leaf.buffer[j] = idx

    def _split_eligible(self, leaf: Node) -> bool:
        if leaf.depth >= self.max_depth:
            return False
        if len(leaf.indices) < self.min_leaf:
            return False
        return np.count_nonzero(leaf.counts) > 1

    def _best_candidate(self, leaf: Node) -> Optional[tuple[float, np.ndarray, float]]:
        pairs: list[tuple[float, int, int]] = []
        buf = leaf.buffer
        for a_pos in range(len(buf)):
            for b_pos in range(a_pos + 1, len(buf)):
                i, j = buf[a_pos], buf[b_pos]
                if self.y_seen[i] == self.y_seen[j]:
                    continue
                dist = float(np.linalg.norm(self.x_seen[i] - self.x_seen[j]))
                if dist <= 1e-12:
                    continue
                pairs.append((1.0 / (dist + 1e-12), i, j))
        if not pairs:
            return None
        pairs.sort(reverse=True, key=lambda x: x[0])
        pairs = pairs[: self.max_candidates]
        best_gain = -math.inf
        best_u: Optional[np.ndarray] = None
        best_b = 0.0
        for _, i, j in pairs:
            hi, hj = self.x_seen[i], self.x_seen[j]
            u = (hi - hj) / np.linalg.norm(hi - hj)
            b = 0.5 * float(u @ (hi + hj))
            gain = self._split_gain(leaf, u, b)
            if gain > best_gain:
                best_gain, best_u, best_b = gain, u, b
        if best_u is None:
            return None
        return best_gain, best_u, best_b

    def _split_gain(self, leaf: Node, u: np.ndarray, b: float) -> float:
        left_counts = np.zeros(self.n_classes, dtype=np.int64)
        right_counts = np.zeros(self.n_classes, dtype=np.int64)
        for idx in leaf.indices:
            target = right_counts if float(u @ self.x_seen[idx]) > b else left_counts
            target[self.y_seen[idx]] += 1
        left_n = int(left_counts.sum())
        right_n = int(right_counts.sum())
        if left_n < self.min_child or right_n < self.min_child:
            return -math.inf
        parent = len(leaf.indices) * entropy_from_counts(leaf.counts)
        children = left_n * entropy_from_counts(left_counts) + right_n * entropy_from_counts(right_counts)
        return float(parent - children)

    def _split_leaf(self, leaf: Node, u: np.ndarray, b: float) -> None:
        old_indices = list(leaf.indices)
        leaf.u = np.asarray(u, dtype=float)
        leaf.b = float(b)
        leaf.left = Node(depth=leaf.depth + 1, n_classes=self.n_classes)
        leaf.right = Node(depth=leaf.depth + 1, n_classes=self.n_classes)
        leaf.indices = []
        leaf.buffer = []
        leaf.counts = np.zeros(self.n_classes, dtype=np.int64)
        for idx in old_indices:
            child = leaf.right if float(leaf.u @ self.x_seen[idx]) > leaf.b else leaf.left
            self._add_index_to_leaf(child, idx)


class DPMExemplarClassifier(DPMClassifier):
    def __init__(self, *args, k_neighbors: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k_neighbors = k_neighbors

    def predict_proba(self, h: np.ndarray) -> np.ndarray:
        if not self.x_seen:
            return np.full(self.n_classes, 1.0 / self.n_classes)
        leaf = self.route(h)
        candidates = leaf.indices if leaf.indices else list(range(len(self.x_seen)))
        if not candidates:
            candidates = list(range(len(self.x_seen)))
        x = np.stack([self.x_seen[i] for i in candidates])
        y = np.array([self.y_seen[i] for i in candidates], dtype=np.int64)
        dist = np.linalg.norm(x - h[None, :], axis=1)
        k = min(self.k_neighbors, len(candidates))
        nn_idx = np.argpartition(dist, k - 1)[:k]
        weights = 1.0 / (dist[nn_idx] + 1e-3)
        votes = np.zeros(self.n_classes, dtype=float)
        for label, weight in zip(y[nn_idx], weights, strict=True):
            votes[int(label)] += float(weight)
        votes += 1e-6
        return votes / votes.sum()


class TokenCacheLM:
    def __init__(self, vocab: int, token_offset: int, alpha: float = 1.0) -> None:
        self.vocab = vocab
        self.token_offset = token_offset
        self.alpha = alpha
        self.counts = np.zeros((vocab, vocab), dtype=np.float64)

    def _token(self, h: np.ndarray) -> int:
        return int(np.argmax(h[self.token_offset : self.token_offset + self.vocab]))

    def predict_proba(self, h: np.ndarray) -> np.ndarray:
        row = self.counts[self._token(h)]
        return (row + self.alpha / self.vocab) / (row.sum() + self.alpha)

    def observe(self, h: np.ndarray, y: int) -> None:
        self.counts[self._token(h), int(y)] += 1.0


class BudgetedKNNMemory:
    def __init__(self, n_classes: int, dim: int, budget: int = 128, k_neighbors: int = 5) -> None:
        self.n_classes = n_classes
        self.dim = dim
        self.budget = budget
        self.k_neighbors = k_neighbors
        self.x_seen: list[np.ndarray] = []
        self.y_seen: list[int] = []

    def predict_proba(self, h: np.ndarray) -> np.ndarray:
        if not self.x_seen:
            return np.full(self.n_classes, 1.0 / self.n_classes)
        x = np.stack(self.x_seen)
        y = np.asarray(self.y_seen, dtype=np.int64)
        dist = np.linalg.norm(x - h[None, :], axis=1)
        k = min(self.k_neighbors, len(y))
        idx = np.argpartition(dist, k - 1)[:k]
        weights = 1.0 / (dist[idx] + 1e-3)
        votes = np.zeros(self.n_classes, dtype=float)
        for label, weight in zip(y[idx], weights, strict=True):
            votes[int(label)] += float(weight)
        votes += 1e-6
        return votes / votes.sum()

    def observe(self, h: np.ndarray, y: int) -> None:
        self.x_seen.append(np.asarray(h, dtype=float))
        self.y_seen.append(int(y))
        if len(self.y_seen) > self.budget:
            self.x_seen.pop(0)
            self.y_seen.pop(0)


class OnlineCentroidMemory:
    def __init__(
        self,
        n_classes: int,
        dim: int,
        max_prototypes: int = 48,
        create_radius: float = 1.1,
        alpha: float = 1.0,
    ) -> None:
        self.n_classes = n_classes
        self.dim = dim
        self.max_prototypes = max_prototypes
        self.create_radius = create_radius
        self.alpha = alpha
        self.centroids: list[np.ndarray] = []
        self.counts: list[np.ndarray] = []
        self.n: list[float] = []

    def _nearest(self, h: np.ndarray) -> tuple[int, float]:
        x = np.stack(self.centroids)
        dist = np.linalg.norm(x - h[None, :], axis=1)
        idx = int(np.argmin(dist))
        return idx, float(dist[idx])

    def predict_proba(self, h: np.ndarray) -> np.ndarray:
        if not self.centroids:
            return np.full(self.n_classes, 1.0 / self.n_classes)
        idx, _ = self._nearest(h)
        row = self.counts[idx]
        return (row + self.alpha / self.n_classes) / (row.sum() + self.alpha)

    def observe(self, h: np.ndarray, y: int) -> None:
        h = np.asarray(h, dtype=float)
        if not self.centroids:
            self.centroids.append(h.copy())
            row = np.zeros(self.n_classes, dtype=float)
            row[int(y)] = 1.0
            self.counts.append(row)
            self.n.append(1.0)
            return
        idx, dist = self._nearest(h)
        if dist > self.create_radius and len(self.centroids) < self.max_prototypes:
            self.centroids.append(h.copy())
            row = np.zeros(self.n_classes, dtype=float)
            row[int(y)] = 1.0
            self.counts.append(row)
            self.n.append(1.0)
            return
        self.n[idx] += 1.0
        eta = 1.0 / self.n[idx]
        self.centroids[idx] = (1.0 - eta) * self.centroids[idx] + eta * h
        self.counts[idx][int(y)] += 1.0


def make_xor_gaussians(n: int, rng: np.random.Generator, noise: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    centers = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    labels = np.array([0, 1, 1, 0], dtype=np.int64)
    choices = rng.integers(0, 4, size=n)
    x = centers[choices] + noise * rng.normal(size=(n, 2))
    y = labels[choices]
    return x, y


def make_contextual_xor_stream(
    n_per_context: int,
    rng: np.random.Generator,
    noise: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0 = make_xor_gaussians(n_per_context, rng, noise=noise)
    x1, y1 = make_xor_gaussians(n_per_context, rng, noise=noise)
    y1 = 1 - y1
    c0 = -np.ones((n_per_context, 1))
    c1 = np.ones((n_per_context, 1))
    x = np.vstack([np.hstack([x0, c0]), np.hstack([x1, c1])])
    y = np.concatenate([y0, y1])
    return x, y


class TorchMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 32, n_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def torch_accuracy(model: TorchMLP, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    return float(np.mean(pred == y))


def train_mlp_offline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    epochs: int = 500,
    lr: float = 3e-3,
    n_classes: Optional[int] = None,
) -> dict[str, float]:
    torch.manual_seed(seed)
    if n_classes is None:
        n_classes = int(max(y_train.max(), y_test.max()) + 1)
    model = TorchMLP(dim=x_train.shape[1], n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x_t), y_t)
        loss.backward()
        opt.step()
    return {
        "train_accuracy": torch_accuracy(model, x_train, y_train),
        "test_accuracy": torch_accuracy(model, x_test, y_test),
    }


def train_mlp_online(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    lr: float = 3e-2,
    n_classes: Optional[int] = None,
) -> dict[str, float]:
    torch.manual_seed(seed)
    if n_classes is None:
        n_classes = int(max(y_train.max(), y_test.max()) + 1)
    model = TorchMLP(dim=x_train.shape[1], n_classes=n_classes)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    correct = 0
    log_loss = 0.0
    for row, label in zip(x_train, y_train, strict=True):
        x_row = torch.tensor(row[None, :], dtype=torch.float32)
        y_row = torch.tensor([int(label)], dtype=torch.long)
        model.eval()
        with torch.no_grad():
            logits = model(x_row)
            pred = int(torch.argmax(logits, dim=1).item())
            prob = torch.softmax(logits, dim=1)[0, int(label)].item()
        correct += int(pred == int(label))
        log_loss += -math.log(max(prob, 1e-12))

        model.train()
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x_row), y_row)
        loss.backward()
        opt.step()
    return {
        "prequential_accuracy": correct / len(y_train),
        "prequential_log_loss": log_loss / len(y_train),
        "test_accuracy": torch_accuracy(model, x_test, y_test),
    }


def evaluate(model: DPMClassifier, x: np.ndarray, y: np.ndarray) -> float:
    preds = np.array([model.predict(row) for row in x], dtype=np.int64)
    return float(np.mean(preds == y))


def load_mnist_arrays(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = datasets.MNIST(root=root, train=True, download=False)
    test = datasets.MNIST(root=root, train=False, download=False)
    x_train = train.data.numpy().astype("float32").reshape(-1, 28 * 28) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    x_test = test.data.numpy().astype("float32").reshape(-1, 28 * 28) / 255.0
    y_test = test.targets.numpy().astype(np.int64)
    return x_train, y_train, x_test, y_test


def load_cifar_arrays(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = datasets.CIFAR10(root=root, train=True, download=False)
    test = datasets.CIFAR10(root=root, train=False, download=False)
    x_train = train.data.astype("float32").reshape(-1, 32 * 32 * 3) / 255.0
    y_train = np.array(train.targets, dtype=np.int64)
    x_test = test.data.astype("float32").reshape(-1, 32 * 32 * 3) / 255.0
    y_test = np.array(test.targets, dtype=np.int64)
    return x_train, y_train, x_test, y_test


def stratified_subset(y: np.ndarray, n_per_class: int, rng: np.random.Generator) -> np.ndarray:
    out: list[int] = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        out.extend(rng.choice(idx, size=n_per_class, replace=False).tolist())
    rng.shuffle(out)
    return np.array(out, dtype=np.int64)


def fit_pca_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    x0 = x_train - mean
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    components = vt[:n_components].T
    train_features = x0 @ components
    test_features = (x_test - mean) @ components
    scale = train_features.std(axis=0, keepdims=True) + 1e-6
    return train_features / scale, test_features / scale


def train_online(model: DPMClassifier, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    correct = 0
    log_loss = 0.0
    for row, label in zip(x, y, strict=True):
        proba = model.predict_proba(row)
        pred = int(np.argmax(proba))
        correct += int(pred == int(label))
        log_loss += -math.log(max(float(proba[int(label)]), 1e-12))
        model.observe(row, int(label))
    return {
        "prequential_accuracy": correct / len(y),
        "prequential_log_loss": log_loss / len(y),
    }


def online_train_memory(model: object, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    correct = 0
    log_loss = 0.0
    for row, label in zip(x, y, strict=True):
        proba = model.predict_proba(row)
        pred = int(np.argmax(proba))
        correct += int(pred == int(label))
        log_loss += -math.log(max(float(proba[int(label)]), 1e-12))
        model.observe(row, int(label))
    return {
        "prequential_accuracy": correct / len(y),
        "prequential_log_loss": log_loss / len(y),
    }


def evaluate_memory(model: object, x: np.ndarray, y: np.ndarray) -> float:
    preds = []
    for row in x:
        preds.append(int(np.argmax(model.predict_proba(row))))
    return float(np.mean(np.asarray(preds, dtype=np.int64) == y))


def run_dpm_classification(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    x_train, y_train = make_xor_gaussians(800, rng)
    x_test, y_test = make_xor_gaussians(2000, rng)
    order = rng.permutation(len(y_train))
    x_train, y_train = x_train[order], y_train[order]

    repair_only = DPMClassifier(
        n_classes=2,
        dim=2,
        lambda_penalty=1e9,
        max_depth=0,
        rng=np.random.default_rng(seed + 1),
    )
    repair_metrics = train_online(repair_only, x_train, y_train)

    dpm = DPMClassifier(
        n_classes=2,
        dim=2,
        lambda_penalty=0.05,
        tau=1e-9,
        min_leaf=24,
        min_child=6,
        buffer_size=128,
        max_depth=8,
        max_candidates=128,
        rng=np.random.default_rng(seed + 2),
    )
    dpm_metrics = train_online(dpm, x_train, y_train)
    split_decreases = [rec["decrease"] for rec in dpm.split_records]
    return {
        "dataset": "xor_gaussians",
        "repair_only": {
            **repair_metrics,
            "test_accuracy": evaluate(repair_only, x_test, y_test),
            "leaves": len(repair_only.leaves()),
            "splits": repair_only.internal_count(),
            "objective": repair_only.objective(),
        },
        "dpm": {
            **dpm_metrics,
            "train_accuracy": evaluate(dpm, x_train, y_train),
            "test_accuracy": evaluate(dpm, x_test, y_test),
            "leaves": len(dpm.leaves()),
            "splits": dpm.internal_count(),
            "objective": dpm.objective(),
            "split_decrease_min": min(split_decreases) if split_decreases else None,
            "split_decrease_max": max(split_decreases) if split_decreases else None,
            "objective_monotone_at_splits": all(x > 0 for x in split_decreases),
        },
    }


def summarize(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=float)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    return out


def run_one_backprop_comparison(seed: int, n_per_context: int = 100) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    x_train, y_train = make_contextual_xor_stream(n_per_context, rng)
    x_test, y_test = make_contextual_xor_stream(1000, rng)

    dpm = DPMClassifier(
        n_classes=2,
        dim=3,
        lambda_penalty=0.05,
        tau=1e-9,
        min_leaf=16,
        min_child=4,
        buffer_size=128,
        max_depth=8,
        max_candidates=128,
        rng=np.random.default_rng(seed + 11),
    )
    dpm_metrics = train_online(dpm, x_train, y_train)
    split_decreases = [rec["decrease"] for rec in dpm.split_records]
    offline_mlp = train_mlp_offline(x_train, y_train, x_test, y_test, seed=seed + 21)
    online_mlp = train_mlp_online(x_train, y_train, x_test, y_test, seed=seed + 31)
    return {
        "seed": seed,
        "n_per_context": n_per_context,
        "backprop_mlp_offline": offline_mlp,
        "backprop_mlp_online": online_mlp,
        "dpm_online": {
            **dpm_metrics,
            "train_accuracy": evaluate(dpm, x_train, y_train),
            "test_accuracy": evaluate(dpm, x_test, y_test),
            "leaves": len(dpm.leaves()),
            "splits": dpm.internal_count(),
            "objective": dpm.objective(),
            "split_decrease_min": min(split_decreases) if split_decreases else None,
            "objective_monotone_at_splits": all(x > 0 for x in split_decreases),
        },
    }


def run_backprop_comparison(seed: int) -> dict[str, object]:
    runs = [run_one_backprop_comparison(seed + i, n_per_context=100) for i in range(5)]
    offline_rows = [r["backprop_mlp_offline"] for r in runs]
    online_rows = [r["backprop_mlp_online"] for r in runs]
    dpm_rows = [r["dpm_online"] for r in runs]
    return {
        "dataset": "contextual_xor_switch",
        "description": "Two-context XOR stream with 100 examples per context. The second context flips labels; the context coordinate is observable.",
        "runs": runs,
        "backprop_mlp_offline": summarize(offline_rows, ["train_accuracy", "test_accuracy"]),
        "backprop_mlp_online": summarize(online_rows, ["prequential_accuracy", "prequential_log_loss", "test_accuracy"]),
        "dpm_online": summarize(
            dpm_rows,
            [
                "prequential_accuracy",
                "prequential_log_loss",
                "train_accuracy",
                "test_accuracy",
                "leaves",
                "splits",
                "objective",
            ],
        ),
        "dpm_monotone_all_runs": all(bool(r["dpm_online"]["objective_monotone_at_splits"]) for r in runs),
    }


def run_one_image_stream(
    dataset_name: str,
    x_all: np.ndarray,
    y_all: np.ndarray,
    x_test_all: np.ndarray,
    y_test_all: np.ndarray,
    seed: int,
    n_per_class: int,
    n_components: Optional[int],
    k_neighbors: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    train_idx = stratified_subset(y_all, n_per_class=n_per_class, rng=rng)
    test_idx = stratified_subset(y_test_all, n_per_class=300, rng=rng)
    x_train_raw, y_train = x_all[train_idx], y_all[train_idx]
    x_test_raw, y_test = x_test_all[test_idx], y_test_all[test_idx]
    if n_components is None:
        x_train, x_test = x_train_raw, x_test_raw
    else:
        x_train, x_test = fit_pca_features(x_train_raw, x_test_raw, n_components=n_components)
    order = rng.permutation(len(y_train))
    x_train, y_train = x_train[order], y_train[order]

    dpm = DPMExemplarClassifier(
        n_classes=10,
        dim=x_train.shape[1],
        alpha=1.0,
        lambda_penalty=0.05,
        tau=1e-9,
        min_leaf=20,
        min_child=4,
        buffer_size=128,
        max_depth=0,
        max_candidates=96,
        k_neighbors=k_neighbors,
        rng=np.random.default_rng(seed + 1),
    )
    dpm_metrics = train_online(dpm, x_train, y_train)
    online_mlp = train_mlp_online(x_train, y_train, x_test, y_test, seed=seed + 2, lr=0.03, n_classes=10)
    offline_mlp = train_mlp_offline(
        x_train,
        y_train,
        x_test,
        y_test,
        seed=seed + 3,
        epochs=400,
        lr=0.003,
        n_classes=10,
    )
    return {
        "seed": seed,
        "dataset": dataset_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_components": int(n_components) if n_components is not None else None,
        "backprop_mlp_offline": offline_mlp,
        "backprop_mlp_online": online_mlp,
        "dpm_online": {
            **dpm_metrics,
            "train_accuracy": evaluate(dpm, x_train, y_train),
            "test_accuracy": evaluate(dpm, x_test, y_test),
            "leaves": len(dpm.leaves()),
            "splits": dpm.internal_count(),
            "objective": dpm.objective(),
            "objective_monotone_at_splits": all(rec["decrease"] > 0 for rec in dpm.split_records),
        },
    }


def run_image_backprop_comparison(seed: int) -> dict[str, object]:
    mnist_root = "/Users/gunale/works/silifen-works/haoran_idea/data"
    cifar_root = "/Users/gunale/works/silifen-works/SLN/data/cifar10"
    mnist = load_mnist_arrays(mnist_root)
    cifar = load_cifar_arrays(cifar_root)
    specs = [
        ("MNIST", mnist, 50, None, 3),
        ("CIFAR10", cifar, 100, None, 5),
    ]
    results: dict[str, object] = {}
    for name, arrays, n_per_class, n_components, k_neighbors in specs:
        runs = [
            run_one_image_stream(
                name,
                arrays[0],
                arrays[1],
                arrays[2],
                arrays[3],
                seed=seed + 100 * i + (0 if name == "MNIST" else 1000),
                n_per_class=n_per_class,
                n_components=n_components,
                k_neighbors=k_neighbors,
            )
            for i in range(3)
        ]
        offline_rows = [r["backprop_mlp_offline"] for r in runs]
        online_rows = [r["backprop_mlp_online"] for r in runs]
        dpm_rows = [r["dpm_online"] for r in runs]
        results[name] = {
            "runs": runs,
            "n_per_class": n_per_class,
            "n_components": n_components,
            "k_neighbors": k_neighbors,
            "backprop_mlp_offline": summarize(offline_rows, ["train_accuracy", "test_accuracy"]),
            "backprop_mlp_online": summarize(online_rows, ["prequential_accuracy", "prequential_log_loss", "test_accuracy"]),
            "dpm_online": summarize(
                dpm_rows,
                [
                    "prequential_accuracy",
                    "prequential_log_loss",
                    "train_accuracy",
                    "test_accuracy",
                    "leaves",
                    "splits",
                ],
            ),
            "dpm_monotone_all_runs": all(bool(r["dpm_online"]["objective_monotone_at_splits"]) for r in runs),
        }
    return results


def make_synthetic_lm_stream(
    perms: np.ndarray,
    context_emb: np.ndarray,
    docs_per_context: int,
    seq_len: int,
    rng: np.random.Generator,
    noise: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    n_contexts, vocab = perms.shape
    rows: list[np.ndarray] = []
    labels: list[int] = []
    doc_contexts = np.repeat(np.arange(n_contexts), docs_per_context)
    rng.shuffle(doc_contexts)
    for context in doc_contexts:
        token = int(rng.integers(0, vocab))
        for _ in range(seq_len):
            onehot = np.zeros(vocab, dtype=float)
            onehot[token] = 1.0
            rows.append(np.concatenate([context_emb[context], onehot]))
            if rng.random() < noise:
                next_token = int(rng.integers(0, vocab))
            else:
                next_token = int(perms[context, token])
            labels.append(next_token)
            token = next_token
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=np.int64)


def run_one_synthetic_lm(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_contexts = 8
    vocab = 24
    context_dim = 8
    context_emb = rng.normal(size=(n_contexts, context_dim))
    context_emb /= np.linalg.norm(context_emb, axis=1, keepdims=True)
    perms = np.stack([rng.permutation(vocab) for _ in range(n_contexts)])
    x_train, y_train = make_synthetic_lm_stream(
        perms,
        context_emb,
        docs_per_context=6,
        seq_len=32,
        rng=np.random.default_rng(seed + 1),
    )
    x_test, y_test = make_synthetic_lm_stream(
        perms,
        context_emb,
        docs_per_context=3,
        seq_len=32,
        rng=np.random.default_rng(seed + 2),
        noise=0.0,
    )
    dgm = DPMExemplarClassifier(
        n_classes=vocab,
        dim=x_train.shape[1],
        alpha=1.0,
        lambda_penalty=0.05,
        tau=1e-9,
        min_leaf=20,
        min_child=4,
        buffer_size=256,
        max_depth=0,
        max_candidates=96,
        k_neighbors=1,
        rng=np.random.default_rng(seed + 3),
    )
    dgm_metrics = train_online(dgm, x_train, y_train)
    online_mlp = train_mlp_online(x_train, y_train, x_test, y_test, seed=seed + 4, lr=0.03, n_classes=vocab)
    offline_mlp = train_mlp_offline(
        x_train,
        y_train,
        x_test,
        y_test,
        seed=seed + 5,
        epochs=500,
        lr=0.003,
        n_classes=vocab,
    )
    return {
        "seed": seed,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "backprop_mlp_offline": offline_mlp,
        "backprop_mlp_online": online_mlp,
        "dgm_online": {
            **dgm_metrics,
            "train_accuracy": evaluate(dgm, x_train, y_train),
            "test_accuracy": evaluate(dgm, x_test, y_test),
        },
    }


def run_synthetic_lm_comparison(seed: int) -> dict[str, object]:
    runs = [run_one_synthetic_lm(seed + 50 * i) for i in range(3)]
    offline_rows = [r["backprop_mlp_offline"] for r in runs]
    online_rows = [r["backprop_mlp_online"] for r in runs]
    dgm_rows = [r["dgm_online"] for r in runs]
    return {
        "runs": runs,
        "backprop_mlp_offline": summarize(offline_rows, ["train_accuracy", "test_accuracy"]),
        "backprop_mlp_online": summarize(online_rows, ["prequential_accuracy", "prequential_log_loss", "test_accuracy"]),
        "dgm_online": summarize(
            dgm_rows,
            ["prequential_accuracy", "prequential_log_loss", "train_accuracy", "test_accuracy"],
        ),
    }


def make_hidden_regime_stream(
    seed: int,
    n_blocks: int,
    block_len: int,
    n_regimes: int = 4,
    vocab: int = 16,
    cue_dim: int = 4,
    cue_noise: float = 0.30,
    transition_noise: float = 0.02,
    centers: Optional[np.ndarray] = None,
    perms: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = rng.normal(size=(n_regimes, cue_dim))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    if perms is None:
        perms = np.stack([rng.permutation(vocab) for _ in range(n_regimes)])
    rows: list[np.ndarray] = []
    labels: list[int] = []
    prev_regime = -1
    token = int(rng.integers(0, vocab))
    for _ in range(n_blocks):
        regime = int(rng.integers(0, n_regimes - 1))
        if regime >= prev_regime and prev_regime >= 0:
            regime += 1
        prev_regime = regime
        for _ in range(block_len):
            cue = centers[regime] + cue_noise * rng.normal(size=cue_dim)
            onehot = np.zeros(vocab, dtype=float)
            onehot[token] = 1.0
            rows.append(np.concatenate([cue, onehot]))
            if rng.random() < transition_noise:
                next_token = int(rng.integers(0, vocab))
            else:
                next_token = int(perms[regime, token])
            labels.append(next_token)
            token = next_token
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=np.int64)


def run_one_hidden_regime(seed: int) -> dict[str, object]:
    vocab = 16
    cue_dim = 4
    task_rng = np.random.default_rng(seed)
    centers = task_rng.normal(size=(4, cue_dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    perms = np.stack([task_rng.permutation(vocab) for _ in range(4)])
    x_train, y_train = make_hidden_regime_stream(
        seed + 1,
        n_blocks=96,
        block_len=18,
        vocab=vocab,
        cue_dim=cue_dim,
        centers=centers,
        perms=perms,
    )
    x_test, y_test = make_hidden_regime_stream(
        seed + 2,
        n_blocks=32,
        block_len=18,
        vocab=vocab,
        cue_dim=cue_dim,
        transition_noise=0.0,
        centers=centers,
        perms=perms,
    )
    dim = x_train.shape[1]
    models: dict[str, object] = {
        "cache_lm": TokenCacheLM(vocab=vocab, token_offset=cue_dim, alpha=1.0),
        "budgeted_knn_lm": BudgetedKNNMemory(n_classes=vocab, dim=dim, budget=64, k_neighbors=5),
        "nearest_centroid_memory": OnlineCentroidMemory(
            n_classes=vocab,
            dim=dim,
            max_prototypes=16,
            create_radius=1.05,
            alpha=1.0,
        ),
        "repair_only": DPMClassifier(
            n_classes=vocab,
            dim=dim,
            lambda_penalty=1e9,
            max_depth=0,
            rng=np.random.default_rng(seed + 10),
        ),
        "refine_repair": DPMExemplarClassifier(
            n_classes=vocab,
            dim=dim,
            lambda_penalty=0.02,
            tau=1e-9,
            min_leaf=32,
            min_child=6,
            buffer_size=160,
            max_depth=12,
            max_candidates=160,
            k_neighbors=1,
            rng=np.random.default_rng(seed + 20),
        ),
    }
    out: dict[str, object] = {"seed": seed, "n_train": int(len(y_train)), "n_test": int(len(y_test))}
    for name, model in models.items():
        metrics = online_train_memory(model, x_train, y_train)
        row: dict[str, object] = {
            **metrics,
            "test_accuracy": evaluate_memory(model, x_test, y_test),
        }
        if isinstance(model, DPMClassifier):
            row["leaves"] = len(model.leaves())
            row["splits"] = model.internal_count()
            row["objective_monotone_at_splits"] = all(rec["decrease"] > 0 for rec in model.split_records)
        if isinstance(model, OnlineCentroidMemory):
            row["prototypes"] = len(model.centroids)
        out[name] = row
    online_mlp = train_mlp_online(x_train, y_train, x_test, y_test, seed=seed + 30, lr=0.03, n_classes=vocab)
    out["online_mlp"] = online_mlp
    return out


def run_hidden_regime_comparison(seed: int) -> dict[str, object]:
    runs = [run_one_hidden_regime(seed + 100 * i) for i in range(5)]
    names = [
        "cache_lm",
        "budgeted_knn_lm",
        "nearest_centroid_memory",
        "repair_only",
        "refine_repair",
        "online_mlp",
    ]
    summary: dict[str, object] = {
        "dataset": "hidden_regime_online_sequence",
        "description": "A hidden regime selects the next-token transition map. The learner observes only a noisy continuous cue and the current token, predicts the next token prequentially, then receives the next-token consequence.",
        "runs": runs,
    }
    for name in names:
        rows = [r[name] for r in runs]
        keys = ["prequential_accuracy", "prequential_log_loss", "test_accuracy"]
        summary[name] = summarize(rows, keys)
    summary["refine_repair_monotone_all_runs"] = all(
        bool(r["refine_repair"]["objective_monotone_at_splits"]) for r in runs
    )
    return summary


def run_projection_memory_check(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    max_abs_error = 0.0
    max_before = 0.0
    max_after = 0.0
    for _ in range(200):
        m, r = 5, 7
        w = rng.normal(size=(m, r))
        k = rng.normal(size=r)
        v = rng.normal(size=m)
        eta = float(10 ** rng.uniform(-2.0, 1.0))
        e = v - w @ k
        w_next = w + (eta / (1.0 + eta * float(k @ k))) * np.outer(e, k)
        e_next = v - w_next @ k
        predicted_ratio = 1.0 / (1.0 + eta * float(k @ k))
        observed_ratio = float(np.linalg.norm(e_next) / max(np.linalg.norm(e), 1e-12))
        max_abs_error = max(max_abs_error, abs(observed_ratio - predicted_ratio))
        max_before = max(max_before, float(np.linalg.norm(e)))
        max_after = max(max_after, float(np.linalg.norm(e_next)))
    return {
        "trials": 200,
        "max_residual_ratio_error": max_abs_error,
        "max_residual_before": max_before,
        "max_residual_after": max_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("experiments/results_dpm.json"))
    parser.add_argument("--suite", choices=["all", "hidden"], default="all")
    args = parser.parse_args()

    if args.suite == "hidden":
        results = {"hidden_regime_comparison": run_hidden_regime_comparison(args.seed + 4000)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True))
        hidden = results["hidden_regime_comparison"]
        print("Hidden-regime online sequence prediction")
        for name in [
            "cache_lm",
            "budgeted_knn_lm",
            "nearest_centroid_memory",
            "repair_only",
            "refine_repair",
            "online_mlp",
        ]:
            row = hidden[name]
            print(
                f"{name:24s} preq_acc={row['prequential_accuracy_mean']:.3f}+/-{row['prequential_accuracy_std']:.3f} "
                f"test_acc={row['test_accuracy_mean']:.3f}+/-{row['test_accuracy_std']:.3f}"
            )
        print(f"refine_repair_monotone_all_runs={hidden['refine_repair_monotone_all_runs']}")
        print(f"wrote {args.output}")
        return

    results = {
        "hidden_regime_comparison": run_hidden_regime_comparison(args.seed + 4000),
        "image_backprop_comparison": run_image_backprop_comparison(args.seed + 2000),
        "synthetic_lm_comparison": run_synthetic_lm_comparison(args.seed + 3000),
        "dpm_sanity": run_dpm_classification(args.seed),
        "backprop_comparison": run_backprop_comparison(args.seed + 1000),
        "projection_memory": run_projection_memory_check(args.seed + 100),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True))

    comp = results["backprop_comparison"]
    print("DGM-style memory vs backprop experiment")
    offline = comp["backprop_mlp_offline"]
    online = comp["backprop_mlp_online"]
    dpm = comp["dpm_online"]
    print(
        f"backprop_offline test_acc={offline['test_accuracy_mean']:.3f}±{offline['test_accuracy_std']:.3f} "
        f"train_acc={offline['train_accuracy_mean']:.3f}±{offline['train_accuracy_std']:.3f}"
    )
    print(
        f"backprop_online  preq_acc={online['prequential_accuracy_mean']:.3f}±{online['prequential_accuracy_std']:.3f} "
        f"test_acc={online['test_accuracy_mean']:.3f}±{online['test_accuracy_std']:.3f}"
    )
    print(
        f"dgm_style        preq_acc={dpm['prequential_accuracy_mean']:.3f}±{dpm['prequential_accuracy_std']:.3f} "
        f"test_acc={dpm['test_accuracy_mean']:.3f}±{dpm['test_accuracy_std']:.3f} "
        f"leaves={dpm['leaves_mean']:.1f} splits={dpm['splits_mean']:.1f}"
    )
    cls = results["dpm_sanity"]
    print("Distinction-tree sanity experiment")
    for name in ["repair_only", "dpm"]:
        row = cls[name]
        print(
            f"{name:12s} preq_acc={row['prequential_accuracy']:.3f} "
            f"test_acc={row['test_accuracy']:.3f} leaves={row['leaves']} "
            f"splits={row['splits']} objective={row['objective']:.3f}"
        )
    print("Image-stream DGM-style memory vs backprop")
    for name, table in results["image_backprop_comparison"].items():
        off = table["backprop_mlp_offline"]
        on = table["backprop_mlp_online"]
        dpm_img = table["dpm_online"]
        print(
            f"{name:7s} offline={off['test_accuracy_mean']:.3f}±{off['test_accuracy_std']:.3f} "
            f"online_bp={on['test_accuracy_mean']:.3f}±{on['test_accuracy_std']:.3f} "
            f"dpm={dpm_img['test_accuracy_mean']:.3f}±{dpm_img['test_accuracy_std']:.3f} "
            f"dpm_preq={dpm_img['prequential_accuracy_mean']:.3f}±{dpm_img['prequential_accuracy_std']:.3f}"
        )
    lm = results["synthetic_lm_comparison"]
    lm_off = lm["backprop_mlp_offline"]
    lm_on = lm["backprop_mlp_online"]
    lm_dgm = lm["dgm_online"]
    print("Synthetic language-modeling stream")
    print(
        f"LM offline={lm_off['test_accuracy_mean']:.3f}±{lm_off['test_accuracy_std']:.3f} "
        f"online_bp={lm_on['test_accuracy_mean']:.3f}±{lm_on['test_accuracy_std']:.3f} "
        f"dgm={lm_dgm['test_accuracy_mean']:.3f}±{lm_dgm['test_accuracy_std']:.3f} "
        f"dgm_preq={lm_dgm['prequential_accuracy_mean']:.3f}±{lm_dgm['prequential_accuracy_std']:.3f}"
    )
    pm = results["projection_memory"]
    print("Projection-memory residual contraction check")
    print(f"max_ratio_error={pm['max_residual_ratio_error']:.3e}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
