from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "projection_learning_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "projection_learning_mplconfig"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np


EPS = 1e-12
NUM_FINE = 100
NUM_COARSE = 20


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def binary_entropy(p: float) -> float:
    p = min(max(float(p), EPS), 1.0 - EPS)
    return float(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))


def bernoulli_nll(y: int, p: float) -> float:
    p = min(max(float(p), EPS), 1.0 - EPS)
    return float(-math.log(p if int(y) == 1 else 1.0 - p))


@dataclass
class CIFAR100Hierarchy:
    train_fine: np.ndarray
    train_coarse: np.ndarray
    test_fine: np.ndarray
    test_coarse: np.ndarray
    fine_names: list[str]
    coarse_names: list[str]
    fine_to_coarse: np.ndarray


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def load_cifar100_hierarchy(root: Path, download: bool) -> CIFAR100Hierarchy:
    try:
        from torchvision.datasets import CIFAR100
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("torchvision is required to load CIFAR-100") from exc

    train_ds = CIFAR100(root=str(root), train=True, download=download)
    test_ds = CIFAR100(root=str(root), train=False, download=download)
    base = Path(root) / train_ds.base_folder
    train_raw = _load_pickle(base / "train")
    test_raw = _load_pickle(base / "test")
    meta = _load_pickle(base / "meta")

    train_fine = np.asarray(train_raw["fine_labels"], dtype=np.int64)
    train_coarse = np.asarray(train_raw["coarse_labels"], dtype=np.int64)
    test_fine = np.asarray(test_raw["fine_labels"], dtype=np.int64)
    test_coarse = np.asarray(test_raw["coarse_labels"], dtype=np.int64)
    fine_names = list(meta["fine_label_names"])
    coarse_names = list(meta["coarse_label_names"])

    fine_to_coarse = np.full(NUM_FINE, -1, dtype=np.int64)
    for fine, coarse in zip(train_fine, train_coarse, strict=True):
        if fine_to_coarse[int(fine)] == -1:
            fine_to_coarse[int(fine)] = int(coarse)
        elif fine_to_coarse[int(fine)] != int(coarse):
            raise ValueError(f"inconsistent CIFAR-100 hierarchy for fine label {fine}")
    if np.any(fine_to_coarse < 0):
        raise ValueError("missing fine-to-coarse labels in CIFAR-100 metadata")

    # Touch the torchvision objects so static analysis does not view them as unused.
    assert len(train_ds) == len(train_fine)
    assert len(test_ds) == len(test_fine)
    return CIFAR100Hierarchy(
        train_fine=train_fine,
        train_coarse=train_coarse,
        test_fine=test_fine,
        test_coarse=test_coarse,
        fine_names=fine_names,
        coarse_names=coarse_names,
        fine_to_coarse=fine_to_coarse,
    )


@dataclass
class HQSStream:
    fine: np.ndarray
    query: np.ndarray
    bits: np.ndarray
    y: np.ndarray


def make_hqs_stream(
    fine_labels: np.ndarray,
    coarse_labels: np.ndarray,
    n: int,
    rng: np.random.Generator,
    eta: float,
    n_spurious: int,
    variant: str,
) -> HQSStream:
    indices = rng.integers(0, len(fine_labels), size=n)
    fine = fine_labels[indices].astype(np.int64)
    coarse = coarse_labels[indices].astype(np.int64)
    query = np.full(n, -1, dtype=np.int64)
    candidate_bits = np.zeros((n, NUM_COARSE), dtype=np.int8)

    if variant == "single_query":
        for i, c in enumerate(coarse):
            if rng.random() < 0.5:
                q = int(c)
            else:
                r = int(rng.integers(0, NUM_COARSE - 1))
                q = r if r < int(c) else r + 1
            query[i] = q
            candidate_bits[i, q] = 1
    elif variant == "balanced_mask":
        candidate_bits = rng.integers(0, 2, size=(n, NUM_COARSE), dtype=np.int8)
    else:
        raise ValueError(f"unknown HQS variant: {variant}")

    y = candidate_bits[np.arange(n), coarse].astype(np.int64)
    flips = rng.random(n) < eta
    y[flips] = 1 - y[flips]
    spurious = rng.integers(0, 2, size=(n, n_spurious), dtype=np.int8)
    bits = np.concatenate([candidate_bits, spurious], axis=1)
    return HQSStream(fine=fine, query=query, bits=bits, y=y)


class BetaMemory:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.n0 = 0.0
        self.n1 = 0.0

    def prob(self) -> float:
        return float((self.n1 + self.alpha) / (self.n0 + self.n1 + 2.0 * self.alpha))

    def update(self, y: int) -> None:
        if int(y) == 1:
            self.n1 += 1.0
        else:
            self.n0 += 1.0

    def add_counts(self, n0: float, n1: float) -> None:
        self.n0 += float(n0)
        self.n1 += float(n1)


class RepairOnlyMemory:
    def __init__(self, alpha: float = 1.0) -> None:
        self.mem = [BetaMemory(alpha) for _ in range(NUM_FINE)]

    def predict(self, z: int, q: int, spurious: np.ndarray | None = None) -> float:
        return self.mem[int(z)].prob()

    def observe(self, z: int, q: int, y: int, spurious: np.ndarray | None = None) -> None:
        self.mem[int(z)].update(int(y))

    def structure_size(self) -> dict[str, int]:
        return {"atoms": NUM_FINE, "edges": 0, "cells": NUM_FINE}


class OracleRefinedMemory:
    def __init__(self, fine_to_coarse: np.ndarray, alpha: float = 1.0) -> None:
        self.fine_to_coarse = fine_to_coarse
        self.pos = [BetaMemory(alpha) for _ in range(NUM_FINE)]
        self.neg = [BetaMemory(alpha) for _ in range(NUM_FINE)]

    def predict(self, z: int, q: int, spurious: np.ndarray | None = None) -> float:
        assert spurious is not None
        return self.pos[int(z)].prob() if int(spurious[int(self.fine_to_coarse[int(z)])]) == 1 else self.neg[int(z)].prob()

    def observe(self, z: int, q: int, y: int, spurious: np.ndarray | None = None) -> None:
        assert spurious is not None
        mem = self.pos[int(z)] if int(spurious[int(self.fine_to_coarse[int(z)])]) == 1 else self.neg[int(z)]
        mem.update(int(y))

    def structure_size(self) -> dict[str, int]:
        return {"atoms": 2 * NUM_FINE, "edges": NUM_FINE, "cells": 2 * NUM_FINE}


class CrossProductMemory:
    def __init__(self, alpha: float = 1.0) -> None:
        self.mem = [[BetaMemory(alpha) for _ in range(NUM_COARSE)] for _ in range(NUM_FINE)]

    def predict(self, z: int, q: int, spurious: np.ndarray | None = None) -> float:
        if int(q) < 0:
            return 0.5
        return self.mem[int(z)][int(q)].prob()

    def observe(self, z: int, q: int, y: int, spurious: np.ndarray | None = None) -> None:
        if int(q) < 0:
            return
        self.mem[int(z)][int(q)].update(int(y))

    def structure_size(self) -> dict[str, int]:
        return {"atoms": NUM_FINE * NUM_COARSE, "edges": 0, "cells": NUM_FINE * NUM_COARSE}


class AdditiveLogistic:
    def __init__(self, lr: float = 0.35) -> None:
        self.bias = 0.0
        self.w_z = np.zeros(NUM_FINE, dtype=float)
        self.w_q = np.zeros(NUM_COARSE, dtype=float)
        self.g_bias = 1e-8
        self.g_z = np.full(NUM_FINE, 1e-8, dtype=float)
        self.g_q = np.full(NUM_COARSE, 1e-8, dtype=float)
        self.lr = float(lr)

    def predict(self, z: int, q: int, spurious: np.ndarray | None = None) -> float:
        assert spurious is not None
        return sigmoid(self.bias + self.w_z[int(z)] + float(np.dot(self.w_q, spurious[:NUM_COARSE])))

    def observe(self, z: int, q: int, y: int, spurious: np.ndarray | None = None) -> None:
        assert spurious is not None
        p = self.predict(z, q, spurious)
        grad = p - int(y)
        self.g_bias += grad * grad
        self.g_z[int(z)] += grad * grad
        active = spurious[:NUM_COARSE].astype(float)
        self.g_q += (grad * active) ** 2
        self.bias -= self.lr * grad / math.sqrt(self.g_bias)
        self.w_z[int(z)] -= self.lr * grad / math.sqrt(self.g_z[int(z)])
        self.w_q -= self.lr * grad * active / np.sqrt(self.g_q)

    def structure_size(self) -> dict[str, int]:
        params = NUM_FINE + NUM_COARSE + 1
        return {"atoms": params, "edges": 0, "cells": params}


class CrossedLogistic:
    def __init__(self, lr: float = 1.0) -> None:
        self.bias = 0.0
        self.w = np.zeros((NUM_FINE, NUM_COARSE), dtype=float)
        self.g_bias = 1e-8
        self.g = np.full((NUM_FINE, NUM_COARSE), 1e-8, dtype=float)
        self.lr = float(lr)

    def predict(self, z: int, q: int, spurious: np.ndarray | None = None) -> float:
        assert spurious is not None
        return sigmoid(self.bias + float(np.dot(self.w[int(z)], spurious[:NUM_COARSE])))

    def observe(self, z: int, q: int, y: int, spurious: np.ndarray | None = None) -> None:
        assert spurious is not None
        p = self.predict(z, q, spurious)
        grad = p - int(y)
        active = spurious[:NUM_COARSE].astype(float)
        self.g_bias += grad * grad
        self.g[int(z)] += (grad * active) ** 2
        self.bias -= 0.02 * self.lr * grad / math.sqrt(self.g_bias)
        self.w[int(z)] -= self.lr * grad * active / np.sqrt(self.g[int(z)])

    def structure_size(self) -> dict[str, int]:
        params = NUM_FINE * NUM_COARSE + 1
        return {"atoms": params, "edges": 0, "cells": params}


class BudgetedCache:
    def __init__(self, budget: int | None, alpha: float = 1.0) -> None:
        self.budget = budget
        self.alpha = float(alpha)
        self.queue: deque[tuple[tuple[int, int], int]] = deque()
        self.counts: dict[tuple[int, int], np.ndarray] = {}
        self.global_counts = np.zeros(2, dtype=np.int64)

    def _key(self, z: int, q: int, spurious: np.ndarray | None) -> tuple[int, int]:
        if int(q) >= 0:
            return int(z), int(q)
        assert spurious is not None
        signature = 0
        for k, bit in enumerate(spurious[:NUM_COARSE]):
            signature |= int(bit) << k
        return int(z), signature

    def predict(self, z: int, q: int, spurious: np.ndarray | None = None) -> float:
        counts = self.counts.get(self._key(z, q, spurious), np.zeros(2, dtype=np.int64))
        total = int(counts.sum())
        if total == 0:
            counts = self.global_counts
        return float((counts[1] + self.alpha) / (counts.sum() + 2.0 * self.alpha))

    def observe(self, z: int, q: int, y: int, spurious: np.ndarray | None = None) -> None:
        key = self._key(z, q, spurious)
        y = int(y)
        if key not in self.counts:
            self.counts[key] = np.zeros(2, dtype=np.int64)
        self.queue.append((key, y))
        self.counts[key][y] += 1
        self.global_counts[y] += 1
        if self.budget is not None and len(self.queue) > self.budget:
            old_key, old_y = self.queue.popleft()
            self.counts[old_key][old_y] -= 1
            if int(self.counts[old_key].sum()) == 0:
                del self.counts[old_key]
            self.global_counts[old_y] -= 1

    def structure_size(self) -> dict[str, int]:
        size = len(self.queue) if self.budget is None else self.budget
        return {"atoms": size, "edges": 0, "cells": size}


def counts_from_rows(rows: list[tuple[int, int, np.ndarray]], bit_fn: Any | None = None) -> dict[int, np.ndarray] | np.ndarray:
    if bit_fn is None:
        counts = np.zeros(2, dtype=np.int64)
        for _, y, _ in rows:
            counts[int(y)] += 1
        return counts
    out = {0: np.zeros(2, dtype=np.int64), 1: np.zeros(2, dtype=np.int64)}
    for q, y, spurious in rows:
        out[int(bit_fn(q, spurious))][int(y)] += 1
    return out


def prob_from_counts(counts: np.ndarray, alpha: float) -> float:
    return float((counts[1] + alpha) / (counts.sum() + 2.0 * alpha))


def mean_loss_for_rows(rows: list[tuple[int, int, np.ndarray]], p: float) -> float:
    return float(np.mean([bernoulli_nll(y, p) for _, y, _ in rows]))


def candidate_gain(
    proposal: list[tuple[int, int, np.ndarray]],
    scoring: list[tuple[int, int, np.ndarray]],
    bit_fn: Any,
    alpha: float,
    m_min: int,
) -> float:
    proposal_child = counts_from_rows(proposal, bit_fn)
    scoring_child = counts_from_rows(scoring, bit_fn)
    assert isinstance(proposal_child, dict)
    assert isinstance(scoring_child, dict)
    if min(int(proposal_child[0].sum()), int(proposal_child[1].sum())) < m_min:
        return -math.inf
    if min(int(scoring_child[0].sum()), int(scoring_child[1].sum())) < m_min:
        return -math.inf

    parent_counts = counts_from_rows(proposal)
    assert isinstance(parent_counts, np.ndarray)
    parent_p = prob_from_counts(parent_counts, alpha)
    child_p = {bit: prob_from_counts(proposal_child[bit], alpha) for bit in (0, 1)}

    parent_loss = mean_loss_for_rows(scoring, parent_p)
    split_loss = float(np.mean([bernoulli_nll(y, child_p[int(bit_fn(q, spurious))]) for q, y, spurious in scoring]))
    return parent_loss - split_loss


class DGMHierarchyQuery:
    def __init__(
        self,
        alpha: float = 1.0,
        proposal_size: int = 64,
        score_size: int = 64,
        m_min: int = 10,
        tau: float = 0.05,
        lambda_edge: float = 0.01,
        n_spurious: int = 20,
    ) -> None:
        self.alpha = float(alpha)
        self.proposal_size = int(proposal_size)
        self.score_size = int(score_size)
        self.m_min = int(m_min)
        self.tau = float(tau)
        self.lambda_edge = float(lambda_edge)
        self.n_spurious = int(n_spurious)
        self.parent = [BetaMemory(alpha) for _ in range(NUM_FINE)]
        self.pos = [BetaMemory(alpha) for _ in range(NUM_FINE)]
        self.neg = [BetaMemory(alpha) for _ in range(NUM_FINE)]
        self.accepted_kind: list[str | None] = [None for _ in range(NUM_FINE)]
        self.accepted_value: list[int | None] = [None for _ in range(NUM_FINE)]
        self.proposal: list[list[tuple[int, int, np.ndarray]]] = [[] for _ in range(NUM_FINE)]
        self.scoring: list[list[tuple[int, int, np.ndarray]]] = [[] for _ in range(NUM_FINE)]
        self.accepted_records: list[dict[str, float | int | str]] = []
        self.rejected_records: list[dict[str, float | int | str]] = []

    def _bit(self, z: int, q: int, spurious: np.ndarray) -> int:
        kind = self.accepted_kind[int(z)]
        value = self.accepted_value[int(z)]
        if kind == "query":
            return int(spurious[int(value)])
        if kind == "spurious":
            return int(spurious[NUM_COARSE + int(value)])
        return 0

    def predict(self, z: int, q: int, spurious: np.ndarray | None = None) -> float:
        z = int(z)
        if self.accepted_kind[z] is None:
            return self.parent[z].prob()
        assert spurious is not None
        return self.pos[z].prob() if self._bit(z, int(q), spurious) == 1 else self.neg[z].prob()

    def observe(self, z: int, q: int, y: int, spurious: np.ndarray | None = None) -> None:
        z = int(z)
        y = int(y)
        assert spurious is not None
        if self.accepted_kind[z] is None:
            self.parent[z].update(y)
            self._add_to_buffers(z, int(q), y, spurious.copy())
            self._try_refine(z)
        elif self._bit(z, int(q), spurious) == 1:
            self.pos[z].update(y)
        else:
            self.neg[z].update(y)

    def _add_to_buffers(self, z: int, q: int, y: int, spurious: np.ndarray) -> None:
        row = (int(q), int(y), spurious)
        if len(self.proposal[z]) < self.proposal_size:
            self.proposal[z].append(row)
        elif len(self.scoring[z]) < self.score_size:
            self.scoring[z].append(row)

    def _try_refine(self, z: int) -> None:
        if len(self.proposal[z]) < self.proposal_size or len(self.scoring[z]) < self.score_size:
            return

        candidates: list[tuple[str, int, Any]] = []
        for k in range(NUM_COARSE):
            candidates.append(("query", k, lambda q, s, kk=k: int(s[kk])))
        for i in range(self.n_spurious):
            candidates.append(("spurious", i, lambda q, s, ii=i: int(s[NUM_COARSE + ii])))

        best_kind = ""
        best_value = -1
        best_raw = -math.inf
        for kind, value, bit_fn in candidates:
            raw = candidate_gain(self.proposal[z], self.scoring[z], bit_fn, self.alpha, self.m_min)
            if raw > best_raw:
                best_kind = kind
                best_value = value
                best_raw = raw
        penalized = best_raw - self.lambda_edge
        record: dict[str, float | int | str] = {
            "z": z,
            "kind": best_kind,
            "value": best_value,
            "raw_gain": float(best_raw),
            "penalized_gain": float(penalized),
        }
        if penalized > self.tau:
            self.accepted_kind[z] = best_kind
            self.accepted_value[z] = best_value
            bit_fn = (
                (lambda q, s, kk=best_value: int(s[kk]))
                if best_kind == "query"
                else (lambda q, s, ii=best_value: int(s[NUM_COARSE + ii]))
            )
            child_counts = counts_from_rows(self.proposal[z], bit_fn)
            assert isinstance(child_counts, dict)
            self.neg[z].add_counts(child_counts[0][0], child_counts[0][1])
            self.pos[z].add_counts(child_counts[1][0], child_counts[1][1])
            self.accepted_records.append(record)
            self.scoring[z] = []
        else:
            self.rejected_records.append(record)
            combined = (self.proposal[z] + self.scoring[z])[-self.proposal_size :]
            self.proposal[z] = combined
            self.scoring[z] = []

    def structure_size(self) -> dict[str, int]:
        splits = sum(kind is not None for kind in self.accepted_kind)
        query_edges = sum(kind == "query" for kind in self.accepted_kind)
        return {"atoms": NUM_FINE + splits, "edges": query_edges, "cells": NUM_FINE + splits}

    def edge_metrics(self, fine_to_coarse: np.ndarray) -> dict[str, float]:
        tp = 0
        fp = 0
        spurious = 0
        for z, kind in enumerate(self.accepted_kind):
            if kind == "query":
                if int(self.accepted_value[z]) == int(fine_to_coarse[z]):
                    tp += 1
                else:
                    fp += 1
            elif kind == "spurious":
                spurious += 1
                fp += 1
        fn = NUM_FINE - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return {
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": f1,
            "true_edges": float(tp),
            "false_edges": float(fp),
            "spurious_splits": float(spurious),
        }

    def adjacency(self) -> np.ndarray:
        mat = np.zeros((NUM_FINE, NUM_COARSE), dtype=np.int64)
        for z, kind in enumerate(self.accepted_kind):
            if kind == "query":
                mat[z, int(self.accepted_value[z])] = 1
        return mat


class LocalStumpBaseline(DGMHierarchyQuery):
    """Per-fine one-level decision stump using the same held-out split score."""

    def __init__(
        self,
        alpha: float = 1.0,
        proposal_size: int = 64,
        score_size: int = 64,
        m_min: int = 10,
        tau: float = 0.05,
        lambda_edge: float = 0.01,
    ) -> None:
        super().__init__(
            alpha=alpha,
            proposal_size=proposal_size,
            score_size=score_size,
            m_min=m_min,
            tau=tau,
            lambda_edge=lambda_edge,
            n_spurious=0,
        )


class FrequencyOnlyEdge:
    """Recover one candidate bit per fine class using only marginal bit frequency."""

    def __init__(self, alpha: float = 1.0, min_samples: int = 128) -> None:
        self.alpha = float(alpha)
        self.min_samples = int(min_samples)
        self.parent = [BetaMemory(alpha) for _ in range(NUM_FINE)]
        self.pos = [BetaMemory(alpha) for _ in range(NUM_FINE)]
        self.neg = [BetaMemory(alpha) for _ in range(NUM_FINE)]
        self.counts = np.zeros((NUM_FINE, NUM_COARSE), dtype=np.int64)
        self.n_seen = np.zeros(NUM_FINE, dtype=np.int64)
        self.accepted_value: list[int | None] = [None for _ in range(NUM_FINE)]

    def predict(self, z: int, q: int, spurious: np.ndarray | None = None) -> float:
        z = int(z)
        edge = self.accepted_value[z]
        if edge is None:
            return self.parent[z].prob()
        assert spurious is not None
        return self.pos[z].prob() if int(spurious[int(edge)]) == 1 else self.neg[z].prob()

    def observe(self, z: int, q: int, y: int, spurious: np.ndarray | None = None) -> None:
        z = int(z)
        y = int(y)
        assert spurious is not None
        edge = self.accepted_value[z]
        if edge is None:
            self.parent[z].update(y)
            self.counts[z] += spurious[:NUM_COARSE].astype(np.int64)
            self.n_seen[z] += 1
            if self.n_seen[z] >= self.min_samples:
                self.accepted_value[z] = int(np.argmax(self.counts[z]))
        elif int(spurious[int(edge)]) == 1:
            self.pos[z].update(y)
        else:
            self.neg[z].update(y)

    def structure_size(self) -> dict[str, int]:
        edges = sum(edge is not None for edge in self.accepted_value)
        return {"atoms": NUM_FINE + edges, "edges": edges, "cells": NUM_FINE + edges}

    def edge_metrics(self, fine_to_coarse: np.ndarray) -> dict[str, float]:
        tp = 0
        fp = 0
        for z, edge in enumerate(self.accepted_value):
            if edge is None:
                continue
            if int(edge) == int(fine_to_coarse[z]):
                tp += 1
            else:
                fp += 1
        fn = NUM_FINE - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return {
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": f1,
            "true_edges": float(tp),
            "false_edges": float(fp),
            "spurious_splits": 0.0,
        }

    def adjacency(self) -> np.ndarray:
        mat = np.zeros((NUM_FINE, NUM_COARSE), dtype=np.int64)
        for z, edge in enumerate(self.accepted_value):
            if edge is not None:
                mat[z, int(edge)] = 1
        return mat


def evaluate_model(model: Any, stream: HQSStream) -> dict[str, float]:
    losses = []
    briers = []
    correct = 0
    for z, q, bits, y in zip(stream.fine, stream.query, stream.bits, stream.y, strict=True):
        p = model.predict(int(z), int(q), bits)
        losses.append(bernoulli_nll(int(y), p))
        briers.append((p - int(y)) ** 2)
        correct += int((p >= 0.5) == bool(int(y)))
    return {
        "nll": float(np.mean(losses)),
        "accuracy": correct / len(stream.y),
        "brier": float(np.mean(briers)),
    }


def train_prequential(
    models: dict[str, Any],
    train_stream: HQSStream,
    eval_stream: HQSStream,
    checkpoint_every: int,
) -> tuple[dict[str, dict[str, float]], dict[str, list[float]], dict[str, list[dict[str, float]]]]:
    totals = {
        name: {"loss": 0.0, "brier": 0.0, "correct": 0.0, "window_loss": 0.0, "window_n": 0.0}
        for name in models
    }
    curves = {name: [] for name in models}
    heldout_curves = {name: [] for name in models}
    steps: list[dict[str, float]] = []

    n = len(train_stream.y)
    for t, (z, q, bits, y) in enumerate(
        zip(train_stream.fine, train_stream.query, train_stream.bits, train_stream.y, strict=True),
        start=1,
    ):
        for name, model in models.items():
            p = model.predict(int(z), int(q), bits)
            loss = bernoulli_nll(int(y), p)
            totals[name]["loss"] += loss
            totals[name]["brier"] += (p - int(y)) ** 2
            totals[name]["correct"] += int((p >= 0.5) == bool(int(y)))
            totals[name]["window_loss"] += loss
            totals[name]["window_n"] += 1.0
            model.observe(int(z), int(q), int(y), bits)

        if t % checkpoint_every == 0 or t == n:
            step_record: dict[str, float] = {"step": float(t)}
            for name, model in models.items():
                curves[name].append(totals[name]["window_loss"] / max(totals[name]["window_n"], 1.0))
                totals[name]["window_loss"] = 0.0
                totals[name]["window_n"] = 0.0
                heldout = evaluate_model(model, eval_stream)
                heldout_curves[name].append(heldout["nll"])
                step_record[f"{name}_heldout_nll"] = heldout["nll"]
            steps.append(step_record)

    metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        heldout = evaluate_model(model, eval_stream)
        metrics[name] = {
            "prequential_nll": totals[name]["loss"] / n,
            "prequential_accuracy": totals[name]["correct"] / n,
            "prequential_brier": totals[name]["brier"] / n,
            "heldout_nll": heldout["nll"],
            "heldout_accuracy": heldout["accuracy"],
            "heldout_brier": heldout["brier"],
            **model.structure_size(),
        }
        if hasattr(model, "edge_metrics"):
            metrics[name].update(model.edge_metrics(models["oracle_refined"].fine_to_coarse))
    curve_payload = {f"{name}_window_nll": vals for name, vals in curves.items()}
    curve_payload.update({f"{name}_heldout_nll": vals for name, vals in heldout_curves.items()})
    return metrics, curve_payload, steps


def summarize_runs(runs: list[dict[str, Any]], model_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metric_keys = [
        "prequential_nll",
        "prequential_accuracy",
        "prequential_brier",
        "heldout_nll",
        "heldout_accuracy",
        "heldout_brier",
        "atoms",
        "edges",
        "cells",
        "edge_precision",
        "edge_recall",
        "edge_f1",
        "true_edges",
        "false_edges",
        "spurious_splits",
    ]
    for name in model_names:
        rows = [run["metrics"][name] for run in runs]
        out: dict[str, float] = {}
        for key in metric_keys:
            vals = [float(row[key]) for row in rows if key in row]
            if vals:
                arr = np.asarray(vals, dtype=float)
                out[f"{key}_mean"] = float(arr.mean())
                out[f"{key}_std"] = float(arr.std(ddof=0))
        summary[name] = out
    return summary


def write_summary_csv(path: Path, summary: dict[str, Any], model_names: list[str]) -> None:
    fields = [
        "method",
        "prequential_nll_mean",
        "heldout_nll_mean",
        "prequential_accuracy_mean",
        "heldout_accuracy_mean",
        "heldout_brier_mean",
        "atoms_mean",
        "edges_mean",
        "cells_mean",
        "edge_precision_mean",
        "edge_recall_mean",
        "edge_f1_mean",
        "true_edges_mean",
        "false_edges_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name in model_names:
            row = {"method": name}
            row.update({field: summary[name].get(field) for field in fields if field != "method"})
            writer.writerow(row)


def write_markdown_summary(path: Path, results: dict[str, Any], model_names: list[str]) -> None:
    labels = {
        "repair_only": "Repair-only",
        "frequency_only": "Frequency-only edge",
        "additive_logistic": "Additive logistic",
        "cross_product_memory": "Cross-product memory",
        "crossed_logistic": "Crossed logistic",
        "cache_200": "Budgeted cache M=200",
        "cache_500": "Budgeted cache M=500",
        "cache_2000": "Budgeted cache M=2000",
        "local_stump": "Per-fine local stump",
        "dgm_hqs": "DGM-HQS",
        "oracle_refined": "Oracle refined",
    }
    lines = [
        "# HQS CIFAR-100 Hierarchy Query Stream",
        "",
        (
            f"variant={results['variant']}, eta={results['eta']}, steps={results['n_steps']}, repeats={results['repeats']}, "
            f"proposal={results['proposal_size']}, score={results['score_size']}"
        ),
        "",
        "| Method | Held-out NLL | Held-out acc. | Atoms/cells | Edges | Edge F1 | True edges | False edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in model_names:
        row = results["summary"][name]

        def val(key: str, digits: int = 3) -> str:
            if key not in row:
                return "--"
            return f"{float(row[key]):.{digits}f}"

        lines.append(
            f"| {labels.get(name, name)} | {val('heldout_nll_mean')} | {val('heldout_accuracy_mean')} | "
            f"{val('atoms_mean', 1)} | {val('edges_mean', 1)} | {val('edge_f1_mean')} | "
            f"{val('true_edges_mean', 1)} | {val('false_edges_mean', 1)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_hqs_figures(out_dir: Path, results: dict[str, Any], hierarchy: CIFAR100Hierarchy) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {
        "repair_only": "#6B7280",
        "frequency_only": "#DC2626",
        "additive_logistic": "#9CA3AF",
        "cross_product_memory": "#D55E00",
        "crossed_logistic": "#E69F00",
        "cache_200": "#A78BFA",
        "cache_500": "#8B5CF6",
        "cache_2000": "#6D28D9",
        "local_stump": "#56B4E9",
        "dgm_hqs": "#0072B2",
        "oracle_refined": "#009E73",
    }
    labels = {
        "repair_only": "Repair only",
        "frequency_only": "Frequency only",
        "additive_logistic": "Additive logistic",
        "cross_product_memory": "Cross-product",
        "crossed_logistic": "Crossed logistic",
        "cache_200": "Cache-200",
        "cache_500": "Cache-500",
        "cache_2000": "Cache-2000",
        "local_stump": "Local stump",
        "dgm_hqs": "DGM-HQS",
        "oracle_refined": "Oracle refined",
    }
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    first_run = results["runs"][0]
    steps = np.asarray(first_run["curve_steps"], dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for name in [
        "repair_only",
        "frequency_only",
        "additive_logistic",
        "cross_product_memory",
        "crossed_logistic",
        "cache_500",
        "local_stump",
        "dgm_hqs",
        "oracle_refined",
    ]:
        if name not in results["summary"]:
            continue
        curves = []
        for run in results["runs"]:
            curves.append(np.asarray(run["curves"][f"{name}_heldout_nll"], dtype=float))
        arr = np.vstack(curves)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(steps, mean, color=colors[name], linewidth=1.15, label=labels[name])
        ax.fill_between(steps, mean - std, mean + std, color=colors[name], alpha=0.10, linewidth=0)
    ax.axhline(math.log(2.0), color="#111111", linewidth=0.7, linestyle=":", label=r"$\log 2$")
    ax.axhline(binary_entropy(float(results["eta"])), color="#374151", linewidth=0.7, linestyle="--", label=r"$H(\eta)$")
    ax.set_xlabel("Online steps")
    ax.set_ylabel("Held-out NLL")
    ax.set_ylim(0.0, 0.85)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.6)
    fig.savefig(out_dir / "hqs_online_nll.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "hqs_online_nll.svg", bbox_inches="tight")
    plt.close(fig)

    adjacency = np.asarray(first_run["dgm_adjacency"], dtype=int)
    truth = np.zeros((NUM_FINE, NUM_COARSE), dtype=int)
    for z, c in enumerate(hierarchy.fine_to_coarse):
        truth[z, int(c)] = 1
    overlay = truth + 2 * adjacency
    fig, ax = plt.subplots(figsize=(3.45, 5.2), constrained_layout=True)
    cmap = plt.matplotlib.colors.ListedColormap(["#FFFFFF", "#D1D5DB", "#EF4444", "#0072B2"])
    ax.imshow(overlay, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=3)
    ax.set_xlabel("Coarse query")
    ax.set_ylabel("Fine class")
    ax.set_xticks(np.arange(NUM_COARSE))
    ax.set_xticklabels([name.replace("_", " ") for name in hierarchy.coarse_names], rotation=70, ha="right")
    ax.set_yticks(np.arange(0, NUM_FINE, 5))
    ax.set_yticklabels([hierarchy.fine_names[i].replace("_", " ") for i in range(0, NUM_FINE, 5)])
    handles = [
        plt.matplotlib.patches.Patch(color="#0072B2", label="True + accepted"),
        plt.matplotlib.patches.Patch(color="#EF4444", label="Accepted only"),
        plt.matplotlib.patches.Patch(color="#D1D5DB", label="True only"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=1)
    fig.savefig(out_dir / "hqs_hierarchy_heatmap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "hqs_hierarchy_heatmap.svg", bbox_inches="tight")
    plt.close(fig)

    size_names = [
        "repair_only",
        "frequency_only",
        "additive_logistic",
        "cache_200",
        "cache_500",
        "local_stump",
        "dgm_hqs",
        "oracle_refined",
        "cross_product_memory",
        "crossed_logistic",
    ]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for name in size_names:
        if name not in results["summary"]:
            continue
        row = results["summary"][name]
        x = float(row.get("atoms_mean", row.get("cells_mean", 0.0)))
        y = float(row["heldout_nll_mean"])
        marker = "*" if name == "dgm_hqs" else "o"
        size = 85 if name == "dgm_hqs" else 36
        ax.scatter(x, y, color=colors[name], s=size, marker=marker, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(x * 1.035, y + 0.006, labels[name], fontsize=6.8, color="#374151")
    ax.axhline(binary_entropy(float(results["eta"])), color="#374151", linewidth=0.7, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Atoms / cells / stored examples")
    ax.set_ylabel("Final held-out NLL")
    ax.set_ylim(0.0, 0.85)
    ax.grid(True, axis="both", color="#E5E7EB", linewidth=0.6)
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.18, top=0.94)
    fig.savefig(out_dir / "hqs_nll_vs_structure.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "hqs_nll_vs_structure.svg", bbox_inches="tight")
    plt.close(fig)


def run_one(
    seed: int,
    hierarchy: CIFAR100Hierarchy,
    args: argparse.Namespace,
    model_names: list[str],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_stream = make_hqs_stream(
        hierarchy.train_fine,
        hierarchy.train_coarse,
        args.n_steps,
        rng,
        args.eta,
        args.n_spurious,
        args.variant,
    )
    eval_stream = make_hqs_stream(
        hierarchy.test_fine,
        hierarchy.test_coarse,
        args.n_eval,
        rng,
        args.eta,
        args.n_spurious,
        args.variant,
    )
    all_models: dict[str, Any] = {
        "repair_only": RepairOnlyMemory(args.alpha),
        "frequency_only": FrequencyOnlyEdge(args.alpha, args.proposal_size + args.score_size),
        "additive_logistic": AdditiveLogistic(args.additive_lr),
        "cross_product_memory": CrossProductMemory(args.alpha),
        "crossed_logistic": CrossedLogistic(args.crossed_lr),
        "cache_200": BudgetedCache(200, args.alpha),
        "cache_500": BudgetedCache(500, args.alpha),
        "cache_2000": BudgetedCache(2000, args.alpha),
        "local_stump": LocalStumpBaseline(
            alpha=args.alpha,
            proposal_size=args.proposal_size,
            score_size=args.score_size,
            m_min=args.m_min,
            tau=args.tau,
            lambda_edge=args.lambda_edge,
        ),
        "dgm_hqs": DGMHierarchyQuery(
            alpha=args.alpha,
            proposal_size=args.proposal_size,
            score_size=args.score_size,
            m_min=args.m_min,
            tau=args.tau,
            lambda_edge=args.lambda_edge,
            n_spurious=args.n_spurious,
        ),
        "oracle_refined": OracleRefinedMemory(hierarchy.fine_to_coarse, args.alpha),
    }
    models = {name: all_models[name] for name in model_names}
    metrics, curves, step_records = train_prequential(models, train_stream, eval_stream, args.checkpoint_every)
    dgm = models["dgm_hqs"]
    assert isinstance(dgm, DGMHierarchyQuery)
    return {
        "seed": seed,
        "metrics": metrics,
        "curves": curves,
        "curve_steps": [int(row["step"]) for row in step_records],
        "dgm_accepted_records": dgm.accepted_records,
        "dgm_rejected_records": dgm.rejected_records,
        "dgm_adjacency": dgm.adjacency().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=60000)
    parser.add_argument("--n-eval", type=int, default=12000)
    parser.add_argument("--checkpoint-every", type=int, default=3000)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--proposal-size", type=int, default=64)
    parser.add_argument("--score-size", type=int, default=64)
    parser.add_argument("--m-min", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--lambda-edge", type=float, default=0.01)
    parser.add_argument("--n-spurious", type=int, default=20)
    parser.add_argument("--additive-lr", type=float, default=0.35)
    parser.add_argument("--crossed-lr", type=float, default=1.0)
    parser.add_argument("--variant", choices=["single_query", "balanced_mask"], default="balanced_mask")
    parser.add_argument("--data-root", type=Path, default=Path("experiments/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/hqs_results"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.repeats = min(args.repeats, 2)
        args.n_steps = min(args.n_steps, 18000)
        args.n_eval = min(args.n_eval, 4000)
        args.checkpoint_every = min(args.checkpoint_every, 1500)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hierarchy = load_cifar100_hierarchy(args.data_root, args.download)
    model_names = [
        "repair_only",
        "frequency_only",
        "additive_logistic",
        "crossed_logistic",
        "cache_200",
        "cache_500",
        "cache_2000",
        "local_stump",
        "dgm_hqs",
        "oracle_refined",
    ]
    if args.variant == "single_query":
        model_names.insert(3, "cross_product_memory")
    runs = [run_one(args.seed + 1000 * i, hierarchy, args, model_names) for i in range(args.repeats)]
    summary = summarize_runs(runs, model_names)
    results = {
        "description": "CIFAR-100 Hierarchy Query Stream oracle-partition diagnostic.",
        "variant": args.variant,
        "seed": args.seed,
        "repeats": args.repeats,
        "n_steps": args.n_steps,
        "n_eval": args.n_eval,
        "eta": args.eta,
        "alpha": args.alpha,
        "proposal_size": args.proposal_size,
        "score_size": args.score_size,
        "m_min": args.m_min,
        "tau": args.tau,
        "lambda_edge": args.lambda_edge,
        "n_spurious": args.n_spurious,
        "theory": {
            "repair_only_nll": math.log(2.0),
            "refined_nll": binary_entropy(args.eta),
            "gap": math.log(2.0) - binary_entropy(args.eta),
        },
        "fine_names": hierarchy.fine_names,
        "coarse_names": hierarchy.coarse_names,
        "fine_to_coarse": hierarchy.fine_to_coarse.tolist(),
        "summary": summary,
        "runs": runs,
    }
    json_path = out_dir / "results_hqs.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_csv(out_dir / "hqs_summary.csv", summary, model_names)
    write_markdown_summary(out_dir / "summary.md", results, model_names)
    plot_hqs_figures(out_dir, results, hierarchy)

    print(f"HQS CIFAR-100 oracle-partition diagnostic ({args.variant})")
    print(
        f"  theory: repair={results['theory']['repair_only_nll']:.3f}, "
        f"refined={results['theory']['refined_nll']:.3f}, gap={results['theory']['gap']:.3f}"
    )
    for name in model_names:
        row = summary[name]
        heldout = row.get("heldout_nll_mean", float("nan"))
        acc = row.get("heldout_accuracy_mean", float("nan"))
        atoms = row.get("atoms_mean", row.get("cells_mean", float("nan")))
        edge_f1 = row.get("edge_f1_mean")
        edge_text = "" if edge_f1 is None else f", edge_f1={edge_f1:.3f}"
        print(f"  {name:22s} heldout_nll={heldout:.3f}, acc={acc:.3f}, atoms={atoms:.1f}{edge_text}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
