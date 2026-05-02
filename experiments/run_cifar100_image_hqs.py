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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.dgm_reference import CategoricalDGM


NUM_FINE = 100
NUM_COARSE = 20
EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def cifar_tensor(raw: np.ndarray) -> torch.Tensor:
    x = raw.reshape(-1, 3, 32, 32).astype("float32") / 255.0
    return torch.from_numpy(x)


def load_cifar100(root: Path) -> CIFAR100Images:
    base = root / "cifar-100-python"
    if not base.exists():
        raise FileNotFoundError(f"missing extracted CIFAR-100 archive under {base}")
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
            raise ValueError(f"inconsistent CIFAR-100 hierarchy for fine label {fine}")
    if any(item < 0 for item in fine_to_coarse):
        raise ValueError("incomplete CIFAR-100 fine-to-coarse map")

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


class RandomProjectionImageEncoder:
    def __init__(self, in_channels: int, image_size: int, embedding_dim: int, seed: int, pool: int, device: torch.device):
        self.pool = int(pool)
        generator = torch.Generator().manual_seed(seed)
        pooled = image_size // self.pool
        input_dim = in_channels * pooled * pooled
        proj = torch.randn(input_dim, embedding_dim, generator=generator) / math.sqrt(float(input_dim))
        self.proj = proj.to(device)

    @torch.no_grad()
    def transform(self, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        chunks = []
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device, non_blocking=True)
            if self.pool > 1:
                xb = F.avg_pool2d(xb, kernel_size=self.pool, stride=self.pool)
            flat = xb.flatten(1)
            flat = flat - flat.mean(dim=1, keepdim=True)
            chunks.append(F.normalize(flat @ self.proj.to(flat.dtype), dim=1))
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
        device: torch.device,
    ) -> None:
        self.base = RandomProjectionImageEncoder(in_channels, image_size, embedding_dim, seed, pool, device)
        self.n_classes = int(n_classes)
        self.prototypes: torch.Tensor | None = None

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> None:
        z = self.base.transform(x, batch_size, device)
        y = y.to(device)
        proto = torch.zeros(self.n_classes, z.shape[1], device=device)
        for cls in range(self.n_classes):
            mask = y == cls
            if bool(mask.any()):
                proto[cls] = z[mask].mean(dim=0)
        self.prototypes = F.normalize(proto, dim=1)

    @torch.no_grad()
    def transform(self, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.prototypes is None:
            raise RuntimeError("SupervisedPrototypeEncoder must be fit before transform")
        chunks = []
        for start in range(0, x.shape[0], batch_size):
            z = self.base.transform(x[start : start + batch_size], batch_size, device)
            chunks.append(F.normalize(z @ self.prototypes.T, dim=1))
        return torch.cat(chunks, dim=0)


class SmallConvEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(192, embedding_dim)
        self.head = nn.Linear(embedding_dim, NUM_FINE)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).flatten(1)
        return F.normalize(self.proj(z), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def train_conv_encoder(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> SmallConvEncoder:
    set_seed(seed)
    model = SmallConvEncoder(embedding_dim).to(device)
    ds = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    synchronize(device)
    return model.eval()


@torch.no_grad()
def embed_conv(model: SmallConvEncoder, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    chunks = []
    for start in range(0, x.shape[0], batch_size):
        chunks.append(model.encode(x[start : start + batch_size].to(device, non_blocking=True)))
    return torch.cat(chunks, dim=0)


def build_embeddings(data: CIFAR100Images, train_idx: torch.Tensor, eval_idx: torch.Tensor, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    start = time.perf_counter()
    if args.encoder == "random_projection":
        encoder = RandomProjectionImageEncoder(3, 32, args.embedding_dim, args.seed + 101, args.pool, device)
        z_train = encoder.transform(data.train_x[train_idx], args.batch_size, device)
        z_eval = encoder.transform(data.eval_x[eval_idx], args.batch_size, device)
    elif args.encoder == "prototype":
        encoder = SupervisedPrototypeEncoder(3, 32, args.embedding_dim, args.seed + 101, args.pool, NUM_FINE, device)
        fit_idx = sample_indices(data.train_fine, args.prototype_fit, args.seed + 71)
        encoder.fit(data.train_x[fit_idx], data.train_fine[fit_idx], args.batch_size, device)
        z_train = encoder.transform(data.train_x[train_idx], args.batch_size, device)
        z_eval = encoder.transform(data.eval_x[eval_idx], args.batch_size, device)
    elif args.encoder == "convnet":
        fit_idx = sample_indices(data.train_fine, args.prototype_fit, args.seed + 71)
        model = train_conv_encoder(
            data.train_x[fit_idx],
            data.train_fine[fit_idx],
            embedding_dim=args.embedding_dim,
            epochs=args.encoder_epochs,
            batch_size=args.encoder_batch_size,
            lr=args.encoder_lr,
            seed=args.seed + 101,
            device=device,
        )
        z_train = embed_conv(model, data.train_x[train_idx], args.batch_size, device)
        z_eval = embed_conv(model, data.eval_x[eval_idx], args.batch_size, device)
    else:
        raise ValueError(f"unknown encoder: {args.encoder}")
    synchronize(device)
    return z_train, z_eval, time.perf_counter() - start


def make_image_hqs(
    z: torch.Tensor,
    coarse: torch.Tensor,
    *,
    seed: int,
    eta: float,
    mask_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    mask_cpu = torch.randint(0, 2, (z.shape[0], NUM_COARSE), generator=generator).float()
    coarse_cpu = coarse.cpu()
    y_cpu = mask_cpu.gather(1, coarse_cpu[:, None]).squeeze(1).long()
    if eta > 0.0:
        flips = torch.rand(z.shape[0], generator=generator) < eta
        y_cpu[flips] = 1 - y_cpu[flips]
    mask = mask_cpu.to(device)
    y = y_cpu.to(device)
    signed_mask = (2.0 * mask - 1.0) * float(mask_scale)
    return torch.cat([z, signed_mask], dim=1), y, mask


class GlobalFrequency:
    def __init__(self, alpha: float, device: torch.device) -> None:
        self.counts = torch.zeros(2, device=device)
        self.alpha = float(alpha)

    def predict_prob(self, h: torch.Tensor) -> torch.Tensor:
        p = (self.counts + self.alpha) / (self.counts.sum() + 2.0 * self.alpha)
        return p.expand(h.shape[0], -1).clamp_min(EPS)

    def observe(self, h: torch.Tensor, y: torch.Tensor) -> None:
        del h
        self.counts.scatter_add_(0, y.long(), torch.ones_like(y, dtype=self.counts.dtype))


class OnlineLogistic:
    def __init__(self, dim: int, lr: float, l2: float, device: torch.device) -> None:
        self.w = torch.zeros(dim, device=device)
        self.b = torch.zeros((), device=device)
        self.g_w = torch.full((dim,), 1e-8, device=device)
        self.g_b = torch.tensor(1e-8, device=device)
        self.lr = float(lr)
        self.l2 = float(l2)

    def features(self, h: torch.Tensor) -> torch.Tensor:
        return h

    def predict_prob(self, h: torch.Tensor) -> torch.Tensor:
        x = self.features(h)
        logit = x @ self.w + self.b
        p1 = torch.sigmoid(logit)
        return torch.stack([1.0 - p1, p1], dim=1).clamp_min(EPS)

    def observe(self, h: torch.Tensor, y: torch.Tensor) -> None:
        x = self.features(h)
        p1 = self.predict_prob(h)[:, 1]
        grad = p1 - y.float()
        grad_w = grad[:, None] * x + self.l2 * self.w
        grad_b = grad
        self.g_w += grad_w.squeeze(0).square()
        self.g_b += grad_b.squeeze(0).square()
        self.w -= self.lr * grad_w.squeeze(0) / torch.sqrt(self.g_w)
        self.b -= self.lr * grad_b.squeeze(0) / torch.sqrt(self.g_b)


class CrossedOnlineLogistic(OnlineLogistic):
    def __init__(self, image_dim: int, lr: float, l2: float, device: torch.device) -> None:
        self.image_dim = int(image_dim)
        super().__init__(self.image_dim + NUM_COARSE + self.image_dim * NUM_COARSE, lr, l2, device)

    def features(self, h: torch.Tensor) -> torch.Tensor:
        z = h[:, : self.image_dim]
        mask = h[:, self.image_dim :]
        crossed = (z[:, :, None] * mask[:, None, :]).reshape(h.shape[0], -1)
        return torch.cat([z, mask, crossed], dim=1)


class ImageOnlyMaskDGM:
    """Image-only routing with consequence-scored mask-bit refinements."""

    def __init__(
        self,
        image_dim: int,
        *,
        k: int,
        alpha: float,
        distance_temperature: float,
        centroid_lr: float,
        max_concepts: int,
        concept_radius: float,
        proposal_size: int,
        score_size: int,
        m_min: int,
        split_tau: float,
        lambda_edge: float,
        device: torch.device,
    ) -> None:
        self.image_dim = int(image_dim)
        self.k = int(k)
        self.alpha = float(alpha)
        self.distance_temperature = float(distance_temperature)
        self.centroid_lr = float(centroid_lr)
        self.max_concepts = int(max_concepts)
        self.concept_radius = float(concept_radius)
        self.proposal_size = int(proposal_size)
        self.score_size = int(score_size)
        self.m_min = int(m_min)
        self.split_tau = float(split_tau)
        self.lambda_edge = float(lambda_edge)
        self.device = device
        self.centroids = torch.empty(0, self.image_dim, device=device)
        self.counts = torch.empty(0, 2, device=device)
        self.totals = torch.empty(0, device=device)
        self.child_counts = torch.empty(0, NUM_COARSE, 2, 2, device=device)
        self.accepted_bits: list[int | None] = []
        self.buffers: list[list[tuple[torch.Tensor, int]]] = []

    @property
    def num_concepts(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def num_edges(self) -> int:
        return sum(bit is not None for bit in self.accepted_bits)

    def _memory_probs(self, counts: torch.Tensor) -> torch.Tensor:
        probs = (counts + self.alpha / 2.0) / (counts.sum(dim=-1, keepdim=True) + self.alpha).clamp_min(EPS)
        return probs.clamp_min(EPS)

    def predict(self, h: torch.Tensor, *, return_aux: bool = False, p0_logits: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del p0_logits
        h = h.to(self.device)
        z = h[:, : self.image_dim]
        mask = h[:, self.image_dim :] > 0
        if self.num_concepts == 0:
            probs = torch.full((h.shape[0], 2), 0.5, device=self.device)
            aux = {
                "selected_idx": torch.full((h.shape[0],), -1, device=self.device, dtype=torch.long),
                "selected_distance": torch.full((h.shape[0],), float("inf"), device=self.device),
                "routing_weights": torch.empty(h.shape[0], 0, device=self.device),
                "candidate_idx": torch.empty(h.shape[0], 0, device=self.device, dtype=torch.long),
                "probs": probs,
            }
            return (probs, aux) if return_aux else probs
        distances = torch.cdist(z, self.centroids, p=2).square()
        kk = min(self.k, self.num_concepts)
        neg_dist, cand_idx = torch.topk(-distances, k=kk, dim=1, sorted=True)
        weights = F.softmax(neg_dist / self.distance_temperature, dim=1)
        rows = []
        for row in range(h.shape[0]):
            row_probs = []
            for concept in cand_idx[row]:
                idx = int(concept.item())
                bit = self.accepted_bits[idx]
                if bit is None:
                    row_probs.append(self._memory_probs(self.counts[idx].unsqueeze(0)).squeeze(0))
                else:
                    side = int(mask[row, bit].item())
                    row_probs.append(self._memory_probs(self.child_counts[idx, bit, side].unsqueeze(0)).squeeze(0))
            rows.append(torch.stack(row_probs, dim=0))
        cand_probs = torch.stack(rows, dim=0)
        probs = (weights[:, :, None] * cand_probs).sum(dim=1).clamp_min(EPS)
        probs = probs / probs.sum(dim=1, keepdim=True)
        selected_pos = torch.argmax(weights, dim=1)
        selected_idx = cand_idx.gather(1, selected_pos[:, None]).squeeze(1)
        selected_distance = (-neg_dist).gather(1, selected_pos[:, None]).squeeze(1)
        aux = {
            "selected_idx": selected_idx,
            "selected_distance": selected_distance,
            "routing_weights": weights,
            "candidate_idx": cand_idx,
            "probs": probs,
        }
        return (probs, aux) if return_aux else probs

    def loss(self, h: torch.Tensor, y: torch.Tensor, *, return_aux: bool = False, reduction: str = "mean", p0_logits: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probs, aux = self.predict(h, return_aux=True, p0_logits=p0_logits)
        y = y.to(self.device).long().flatten()
        target_prob = probs.gather(1, y[:, None]).squeeze(1).clamp_min(EPS)
        losses = -torch.log(target_prob)
        out = losses if reduction == "none" else losses.sum() if reduction == "sum" else losses.mean()
        aux["target_prob"] = target_prob
        return (out, aux) if return_aux else out

    @torch.no_grad()
    def observe(self, h: torch.Tensor, y: torch.Tensor, *, aux: dict[str, torch.Tensor] | None = None) -> None:
        h = h.to(self.device)
        y = y.to(self.device).long().flatten()
        if aux is None:
            _, aux = self.predict(h, return_aux=True)
        z = h[:, : self.image_dim]
        mask = h[:, self.image_dim :] > 0
        for row in range(h.shape[0]):
            y_i = int(y[row].item())
            selected = int(aux["selected_idx"][row].item())
            selected_distance = float(aux["selected_distance"][row].item())
            if self.num_concepts == 0 or selected < 0:
                self._add_concept(z[row], mask[row], y_i)
            elif selected_distance > self.concept_radius and self.num_concepts < self.max_concepts:
                self._add_concept(z[row], mask[row], y_i)
            else:
                self._repair(selected, z[row], mask[row], y_i)

    def _add_concept(self, z: torch.Tensor, mask: torch.Tensor, y: int) -> None:
        self.centroids = torch.cat([self.centroids, z.detach().reshape(1, -1)], dim=0)
        self.counts = torch.cat([self.counts, torch.zeros(1, 2, device=self.device)], dim=0)
        self.totals = torch.cat([self.totals, torch.zeros(1, device=self.device)], dim=0)
        self.child_counts = torch.cat([self.child_counts, torch.zeros(1, NUM_COARSE, 2, 2, device=self.device)], dim=0)
        self.accepted_bits.append(None)
        self.buffers.append([])
        self._repair(self.num_concepts - 1, z, mask, y)

    def _repair(self, idx: int, z: torch.Tensor, mask: torch.Tensor, y: int) -> None:
        self.counts[idx, y] += 1.0
        self.totals[idx] += 1.0
        bit = self.accepted_bits[idx]
        if bit is not None:
            self.child_counts[idx, bit, int(mask[bit].item()), y] += 1.0
        else:
            self.buffers[idx].append((mask.detach().cpu(), int(y)))
            max_buffer = self.proposal_size + self.score_size
            if len(self.buffers[idx]) > max_buffer:
                self.buffers[idx] = self.buffers[idx][-max_buffer:]
            self._try_split(idx)
        if self.centroid_lr > 0.0:
            self.centroids[idx].mul_(1.0 - self.centroid_lr).add_(z, alpha=self.centroid_lr)

    def _try_split(self, idx: int) -> None:
        rows = self.buffers[idx]
        if len(rows) < self.proposal_size + self.score_size:
            return
        proposal = rows[: self.proposal_size]
        scoring = rows[self.proposal_size : self.proposal_size + self.score_size]
        parent = torch.zeros(2)
        for _, y in proposal:
            parent[y] += 1.0
        parent_probs = ((parent + self.alpha / 2.0) / (parent.sum() + self.alpha)).clamp_min(EPS)
        best_bit = None
        best_gain = -float("inf")
        for bit in range(NUM_COARSE):
            child = torch.zeros(2, 2)
            score_child = torch.zeros(2, 2)
            for mask, y in proposal:
                child[int(mask[bit].item()), y] += 1.0
            for mask, y in scoring:
                score_child[int(mask[bit].item()), y] += 1.0
            if child.sum(dim=1).min().item() < self.m_min or score_child.sum(dim=1).min().item() < self.m_min:
                continue
            child_probs = ((child + self.alpha / 2.0) / (child.sum(dim=1, keepdim=True) + self.alpha)).clamp_min(EPS)
            parent_loss = 0.0
            split_loss = 0.0
            for mask, y in scoring:
                side = int(mask[bit].item())
                parent_loss += -math.log(float(parent_probs[y]))
                split_loss += -math.log(float(child_probs[side, y]))
            gain = parent_loss / len(scoring) - split_loss / len(scoring)
            if gain > best_gain:
                best_gain = gain
                best_bit = bit
        if best_bit is not None and best_gain - self.lambda_edge > self.split_tau:
            self.accepted_bits[idx] = int(best_bit)
            all_child = torch.zeros(2, 2, device=self.device)
            for mask, y in rows:
                all_child[int(mask[best_bit].item()), y] += 1.0
            self.child_counts[idx, best_bit] = all_child


def nll_and_correct(probs: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
    target = probs.gather(1, y[:, None]).squeeze(1).clamp_min(EPS)
    loss = -torch.log(target)
    correct = int((torch.argmax(probs, dim=1) == y).sum().item())
    return loss, correct


def update_totals(totals: dict[str, float], prefix: str, probs: torch.Tensor, y: torch.Tensor) -> None:
    loss, correct = nll_and_correct(probs, y)
    totals[f"{prefix}_nll"] += float(loss.sum().item())
    totals[f"{prefix}_correct"] += float(correct)


class TrustAggregator:
    def __init__(self, lambda_grid: list[float], device: torch.device) -> None:
        self.lambda_grid = torch.tensor(lambda_grid, device=device, dtype=torch.float32)
        self.weights = torch.full((len(lambda_grid),), 1.0 / len(lambda_grid), device=device)

    def expert_probs(self, base: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        lam = self.lambda_grid[:, None, None]
        return (1.0 - lam) * base.unsqueeze(0) + lam * memory.unsqueeze(0)

    def predict_prob(self, base: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        experts = self.expert_probs(base, memory)
        return (self.weights[:, None, None] * experts).sum(dim=0).clamp_min(EPS)

    def observe(self, base: torch.Tensor, memory: torch.Tensor, y: torch.Tensor) -> None:
        experts = self.expert_probs(base, memory)
        likelihood = experts[:, torch.arange(y.shape[0], device=y.device), y].prod(dim=1).clamp_min(EPS)
        self.weights = self.weights * likelihood
        self.weights = self.weights / self.weights.sum().clamp_min(EPS)

    def as_dict(self) -> dict[str, float]:
        return {f"{float(lam):g}": float(weight) for lam, weight in zip(self.lambda_grid.cpu(), self.weights.cpu(), strict=True)}


@torch.no_grad()
def evaluate_prob_model(model: Any, h: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, float]:
    losses = []
    correct = 0
    start = time.perf_counter()
    for begin in range(0, h.shape[0], batch_size):
        hb = h[begin : begin + batch_size].to(device)
        yb = y[begin : begin + batch_size].to(device)
        probs = model.predict_prob(hb) if hasattr(model, "predict_prob") else model.predict(hb)
        assert isinstance(probs, torch.Tensor)
        loss, batch_correct = nll_and_correct(probs, yb)
        losses.append(loss.detach())
        correct += batch_correct
    synchronize(device)
    return {
        "heldout_nll": float(torch.cat(losses).mean().item()),
        "heldout_accuracy": correct / float(h.shape[0]),
        "heldout_query_seconds": time.perf_counter() - start,
    }


@torch.no_grad()
def evaluate_trust(base_model: Any, dgm: Any, trust: TrustAggregator, h: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, float]:
    losses = []
    correct = 0
    for begin in range(0, h.shape[0], batch_size):
        hb = h[begin : begin + batch_size].to(device)
        yb = y[begin : begin + batch_size].to(device)
        base = base_model.predict_prob(hb)
        memory = dgm.predict(hb)
        probs = trust.predict_prob(base, memory)
        loss, batch_correct = nll_and_correct(probs, yb)
        losses.append(loss.detach())
        correct += batch_correct
    return {
        "heldout_nll": float(torch.cat(losses).mean().item()),
        "heldout_accuracy": correct / float(h.shape[0]),
    }


@torch.no_grad()
def routing_info(model: Any, h: torch.Tensor, batch_size: int, device: torch.device) -> tuple[torch.Tensor, float, float]:
    if getattr(model, "num_concepts", 0) == 0:
        return torch.full((h.shape[0],), -1, dtype=torch.long), 0.0, 0.0
    selected = []
    entropies = []
    margins = []
    for begin in range(0, h.shape[0], batch_size):
        hb = h[begin : begin + batch_size].to(device)
        _, aux = model.predict(hb, return_aux=True)
        weights = aux["routing_weights"].detach().clamp_min(EPS)
        selected.append(aux["selected_idx"].detach().cpu())
        entropies.append((-(weights * torch.log(weights)).sum(dim=1)).detach().cpu())
        if weights.shape[1] == 1:
            margins.append(torch.ones(weights.shape[0]))
        else:
            top2 = torch.topk(weights, k=2, dim=1).values
            margins.append((top2[:, 0] - top2[:, 1]).detach().cpu())
    return torch.cat(selected), float(torch.cat(entropies).mean().item()), float(torch.cat(margins).mean().item())


def purity_from_assignments(assignments: torch.Tensor, labels: torch.Tensor, n_concepts: int, n_labels: int) -> float:
    if n_concepts <= 0:
        return 0.0
    counts = torch.zeros(n_concepts, n_labels)
    for concept, label in zip(assignments, labels.cpu(), strict=True):
        idx = int(concept.item())
        if idx >= 0:
            counts[idx, int(label.item())] += 1.0
    total = float(counts.sum().item())
    return 0.0 if total <= 0.0 else float(counts.max(dim=1).values.sum().item() / total)


def binary_mi(counts: torch.Tensor) -> float:
    total = float(counts.sum().item())
    if total <= 0.0:
        return 0.0
    joint = counts / total
    px = joint.sum(dim=1, keepdim=True)
    py = joint.sum(dim=0, keepdim=True)
    expected = px @ py
    mask = joint > 0
    return float((joint[mask] * torch.log((joint[mask] / expected[mask]).clamp_min(EPS))).sum().item())


def mask_bit_diagnostics(assignments: torch.Tensor, mask: torch.Tensor, coarse: torch.Tensor, y: torch.Tensor, n_concepts: int, accepted_bits: list[int | None] | None) -> dict[str, float | None]:
    ranks = []
    true_mi = []
    best_spurious_mi = []
    edge_tp = 0
    edge_fp = 0
    eligible = 0
    mask_cpu = mask.detach().cpu() > 0.5
    coarse_cpu = coarse.detach().cpu()
    y_cpu = y.detach().cpu()
    for concept in range(n_concepts):
        rows = assignments == concept
        n_rows = int(rows.sum().item())
        if n_rows < 8:
            continue
        eligible += 1
        concept_coarse = coarse_cpu[rows]
        majority = int(torch.mode(concept_coarse).values.item())
        scores = []
        for bit in range(NUM_COARSE):
            counts = torch.zeros(2, 2)
            for side, label in zip(mask_cpu[rows, bit], y_cpu[rows], strict=True):
                counts[int(side.item()), int(label.item())] += 1.0
            scores.append(binary_mi(counts))
        ordered = sorted(range(NUM_COARSE), key=lambda b: scores[b], reverse=True)
        rank = ordered.index(majority) + 1
        ranks.append(rank)
        true_mi.append(scores[majority])
        spurious = [scores[b] for b in range(NUM_COARSE) if b != majority]
        best_spurious_mi.append(max(spurious) if spurious else 0.0)
        if accepted_bits is not None and concept < len(accepted_bits) and accepted_bits[concept] is not None:
            if int(accepted_bits[concept]) == majority:
                edge_tp += 1
            else:
                edge_fp += 1
    if ranks:
        precision = edge_tp / (edge_tp + edge_fp) if edge_tp + edge_fp > 0 else None
        recall = edge_tp / eligible if eligible > 0 else None
        f1 = None if precision is None or recall is None or precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        return {
            "true_mask_bit_rank_mean": float(np.mean(ranks)),
            "true_mask_bit_rank_median": float(np.median(ranks)),
            "avg_local_mi_true_bit": float(np.mean(true_mi)),
            "avg_local_mi_best_spurious_bit": float(np.mean(best_spurious_mi)),
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": f1,
            "diagnostic_concepts": float(eligible),
        }
    return {
        "true_mask_bit_rank_mean": None,
        "true_mask_bit_rank_median": None,
        "avg_local_mi_true_bit": None,
        "avg_local_mi_best_spurious_bit": None,
        "edge_precision": None,
        "edge_recall": None,
        "edge_f1": None,
        "diagnostic_concepts": 0.0,
    }


def summarize_concept_sizes(model: Any) -> dict[str, float]:
    totals = getattr(model, "totals", torch.empty(0)).detach().cpu()
    if totals.numel() == 0:
        return {"avg_samples_per_concept": 0.0, "median_samples_per_concept": 0.0}
    return {
        "avg_samples_per_concept": float(totals.float().mean().item()),
        "median_samples_per_concept": float(totals.float().median().item()),
    }


def edge_mask_fraction(model: Any, image_dim: int) -> float | None:
    if getattr(model, "num_edges", 0) == 0:
        return 0.0
    if hasattr(model, "accepted_bits"):
        return 1.0
    edge_u = getattr(model, "edge_u", None)
    if edge_u is None or edge_u.numel() == 0:
        return None
    edge_u = edge_u.detach()
    image_norm = torch.linalg.vector_norm(edge_u[:, :image_dim], dim=1)
    mask_norm = torch.linalg.vector_norm(edge_u[:, image_dim:], dim=1)
    return float((mask_norm / (image_norm + mask_norm + EPS)).mean().item())


def build_dgm(args: argparse.Namespace, image_dim: int, device: torch.device) -> Any:
    if args.dgm_variant == "categorical":
        return CategoricalDGM(
            dim=image_dim + NUM_COARSE,
            n_classes=2,
            k=args.k,
            alpha=args.alpha,
            distance_temperature=args.distance_temperature,
            edge_temperature=args.edge_temperature,
            edge_weight=args.edge_weight,
            centroid_lr=args.centroid_lr,
            min_refine_total=args.min_refine_total,
            refine_loss_threshold=args.refine_loss_threshold,
            refine_on_error=True,
            max_concepts=args.max_concepts,
        ).to(device)
    if args.dgm_variant == "image_only_mask":
        return ImageOnlyMaskDGM(
            image_dim=image_dim,
            k=args.k,
            alpha=args.alpha,
            distance_temperature=args.distance_temperature,
            centroid_lr=args.centroid_lr,
            max_concepts=args.max_concepts,
            concept_radius=args.concept_radius,
            proposal_size=args.proposal_size,
            score_size=args.score_size,
            m_min=args.m_min,
            split_tau=args.split_tau,
            lambda_edge=args.lambda_edge,
            device=device,
        )
    raise ValueError(f"unknown dgm_variant: {args.dgm_variant}")


def run_prequential(
    h_train: torch.Tensor,
    y_train: torch.Tensor,
    mask_train: torch.Tensor,
    h_eval: torch.Tensor,
    y_eval: torch.Tensor,
    mask_eval: torch.Tensor,
    fine_train: torch.Tensor,
    coarse_train: torch.Tensor,
    fine_eval: torch.Tensor,
    coarse_eval: torch.Tensor,
    image_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    dgm = build_dgm(args, image_dim, device)
    online_logistic = OnlineLogistic(h_train.shape[1], args.logistic_lr, args.logistic_l2, device)
    crossed_logistic = CrossedOnlineLogistic(image_dim, args.crossed_lr, args.logistic_l2, device)
    global_frequency = GlobalFrequency(args.alpha, device)
    trust = TrustAggregator(args.lambda_grid, device)
    totals = {f"{name}_{kind}": 0.0 for name in ["dgm", "online_logistic", "crossed_logistic", "global_frequency", "logistic_dgm_trust"] for kind in ["nll", "correct"]}

    start = time.perf_counter()
    for idx in range(h_train.shape[0]):
        hb = h_train[idx : idx + 1].to(device)
        yb = y_train[idx : idx + 1].to(device)

        dgm_loss, aux = dgm.loss(hb, yb, return_aux=True)
        dgm_probs = aux["probs"]
        update_totals(totals, "dgm", dgm_probs, yb)

        logistic_probs = online_logistic.predict_prob(hb)
        crossed_probs = crossed_logistic.predict_prob(hb)
        global_probs = global_frequency.predict_prob(hb)
        trust_probs = trust.predict_prob(logistic_probs, dgm_probs)
        update_totals(totals, "online_logistic", logistic_probs, yb)
        update_totals(totals, "crossed_logistic", crossed_probs, yb)
        update_totals(totals, "global_frequency", global_probs, yb)
        update_totals(totals, "logistic_dgm_trust", trust_probs, yb)

        trust.observe(logistic_probs, dgm_probs, yb)
        dgm.observe(hb, yb, aux=aux)
        online_logistic.observe(hb, yb)
        crossed_logistic.observe(hb, yb)
        global_frequency.observe(hb, yb)
    synchronize(device)
    train_seconds = time.perf_counter() - start

    n_train = float(h_train.shape[0])
    metrics: dict[str, Any] = {}
    models = {
        "dgm": dgm,
        "online_logistic": online_logistic,
        "crossed_logistic": crossed_logistic,
        "global_frequency": global_frequency,
    }
    for name, model in models.items():
        heldout = evaluate_prob_model(model, h_eval, y_eval, args.batch_size, device)
        metrics[name] = {
            "prequential_nll": totals[f"{name}_nll"] / n_train,
            "prequential_accuracy": totals[f"{name}_correct"] / n_train,
            **heldout,
        }
    trust_heldout = evaluate_trust(online_logistic, dgm, trust, h_eval, y_eval, args.batch_size, device)
    metrics["logistic_dgm_trust"] = {
        "prequential_nll": totals["logistic_dgm_trust_nll"] / n_train,
        "prequential_accuracy": totals["logistic_dgm_trust_correct"] / n_train,
        **trust_heldout,
        "final_weights": trust.as_dict(),
        "log_grid_size": math.log(len(args.lambda_grid)),
    }

    assignments, routing_entropy, routing_margin = routing_info(dgm, h_train, args.batch_size, device)
    eval_assignments, eval_entropy, eval_margin = routing_info(dgm, h_eval, args.batch_size, device)
    accepted_bits = getattr(dgm, "accepted_bits", None)
    diagnostics = {
        "actual_concepts": float(getattr(dgm, "num_concepts", 0)),
        "accepted_edges": float(getattr(dgm, "num_edges", 0)),
        **summarize_concept_sizes(dgm),
        "concept_purity_fine": purity_from_assignments(assignments, fine_train, getattr(dgm, "num_concepts", 0), NUM_FINE),
        "concept_purity_coarse": purity_from_assignments(assignments, coarse_train, getattr(dgm, "num_concepts", 0), NUM_COARSE),
        "concept_purity_fine_heldout": purity_from_assignments(eval_assignments, fine_eval, getattr(dgm, "num_concepts", 0), NUM_FINE),
        "concept_purity_coarse_heldout": purity_from_assignments(eval_assignments, coarse_eval, getattr(dgm, "num_concepts", 0), NUM_COARSE),
        "routing_entropy": routing_entropy,
        "top1_routing_margin": routing_margin,
        "routing_entropy_heldout": eval_entropy,
        "top1_routing_margin_heldout": eval_margin,
        "edge_mask_fraction": edge_mask_fraction(dgm, image_dim),
        **mask_bit_diagnostics(assignments, mask_train, coarse_train, y_train, getattr(dgm, "num_concepts", 0), accepted_bits),
    }
    metrics["dgm"].update(
        {
            "concepts": getattr(dgm, "num_concepts", 0),
            "edges": getattr(dgm, "num_edges", 0),
            "fine_purity_train": diagnostics["concept_purity_fine"],
            "coarse_purity_train": diagnostics["concept_purity_coarse"],
            "fine_purity_eval": diagnostics["concept_purity_fine_heldout"],
            "coarse_purity_eval": diagnostics["concept_purity_coarse_heldout"],
            "routing_margin_eval": diagnostics["top1_routing_margin_heldout"],
            "edge_mask_fraction": diagnostics["edge_mask_fraction"],
        }
    )
    return {"metrics": metrics, "diagnostics": diagnostics, "train_seconds": train_seconds}


def run_one(seed: int, data: CIFAR100Images, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    set_seed(seed)
    train_idx = sample_indices(data.train_fine, args.n_train, seed + 11)
    eval_idx = sample_indices(data.eval_fine, args.n_eval, seed + 29)
    z_train, z_eval, embed_seconds = build_embeddings(data, train_idx, eval_idx, args, device)
    train_coarse = data.train_coarse[train_idx]
    eval_coarse = data.eval_coarse[eval_idx]
    h_train, y_train, mask_train = make_image_hqs(z_train, train_coarse, seed=seed + 303, eta=args.eta, mask_scale=args.mask_scale, device=device)
    h_eval, y_eval, mask_eval = make_image_hqs(z_eval, eval_coarse, seed=seed + 404, eta=args.eta, mask_scale=args.mask_scale, device=device)
    payload = run_prequential(
        h_train,
        y_train,
        mask_train,
        h_eval,
        y_eval,
        mask_eval,
        data.train_fine[train_idx],
        train_coarse,
        data.eval_fine[eval_idx],
        eval_coarse,
        image_dim=z_train.shape[1],
        args=args,
        device=device,
    )
    payload.update({"seed": seed, "embed_seconds": embed_seconds})
    return payload


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    model_names = runs[0]["metrics"].keys()
    for model in model_names:
        aggregate[model] = {}
        keys = set().union(*(run["metrics"][model].keys() for run in runs))
        for key in sorted(keys):
            vals = [run["metrics"][model][key] for run in runs if isinstance(run["metrics"][model].get(key), (int, float))]
            if vals:
                arr = np.asarray(vals, dtype=float)
                aggregate[model][f"{key}_mean"] = float(arr.mean())
                aggregate[model][f"{key}_std"] = float(arr.std(ddof=0))
    diag_keys = set().union(*(run["diagnostics"].keys() for run in runs))
    aggregate["diagnostics"] = {}
    for key in sorted(diag_keys):
        vals = [run["diagnostics"][key] for run in runs if isinstance(run["diagnostics"].get(key), (int, float))]
        if vals:
            arr = np.asarray(vals, dtype=float)
            aggregate["diagnostics"][f"{key}_mean"] = float(arr.mean())
            aggregate["diagnostics"][f"{key}_std"] = float(arr.std(ddof=0))
    return aggregate


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = select_device(args.device)
    if args.quick:
        args.n_train = min(args.n_train, 2000)
        args.n_eval = min(args.n_eval, 800)
        args.embedding_dim = min(args.embedding_dim, 64)
        args.max_concepts = min(args.max_concepts, 256)
        args.prototype_fit = min(args.prototype_fit, 10000)
        args.encoder_epochs = min(args.encoder_epochs, 3)
    data = load_cifar100(args.data_root)
    runs = [run_one(args.seed + 1000 * rep, data, args, device) for rep in range(args.repeats)]
    return {
        "description": "CIFAR-100 Image-BalancedMask-HQS with image embeddings plus query masks; no oracle fine label is exposed to the online learner.",
        "seed": args.seed,
        "repeats": args.repeats,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "encoder": args.encoder,
        "dgm_variant": args.dgm_variant,
        "embedding_dim": args.embedding_dim,
        "prototype_fit": args.prototype_fit,
        "encoder_epochs": args.encoder_epochs if args.encoder == "convnet" else 0,
        "eta": args.eta,
        "mask_scale": args.mask_scale,
        "max_concepts": args.max_concepts,
        "k": args.k,
        "alpha": args.alpha,
        "lambda_grid": args.lambda_grid,
        "fine_names": data.fine_names,
        "coarse_names": data.coarse_names,
        "fine_to_coarse": data.fine_to_coarse,
        "metrics": runs[0]["metrics"],
        "diagnostics": runs[0]["diagnostics"],
        "embed_seconds": runs[0]["embed_seconds"],
        "train_seconds": runs[0]["train_seconds"],
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }


def parse_lambda_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if 0.0 not in values:
        values.insert(0, 0.0)
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("experiments/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/cifar100_image_hqs_results"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--encoder", choices=["prototype", "random_projection", "convnet"], default="prototype")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--prototype-fit", type=int, default=50000)
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--encoder-epochs", type=int, default=8)
    parser.add_argument("--encoder-batch-size", type=int, default=256)
    parser.add_argument("--encoder-lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dgm-variant", choices=["categorical", "image_only_mask"], default="categorical")
    parser.add_argument("--max-concepts", type=int, default=256)
    parser.add_argument("--concept-radius", type=float, default=0.32)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--distance-temperature", type=float, default=0.35)
    parser.add_argument("--edge-temperature", type=float, default=8.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--centroid-lr", type=float, default=0.04)
    parser.add_argument("--min-refine-total", type=float, default=0.0)
    parser.add_argument("--refine-loss-threshold", type=float, default=0.0)
    parser.add_argument("--proposal-size", type=int, default=32)
    parser.add_argument("--score-size", type=int, default=32)
    parser.add_argument("--m-min", type=int, default=4)
    parser.add_argument("--split-tau", type=float, default=0.005)
    parser.add_argument("--lambda-edge", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--mask-scale", type=float, default=0.20)
    parser.add_argument("--logistic-lr", type=float, default=0.4)
    parser.add_argument("--crossed-lr", type=float, default=0.15)
    parser.add_argument("--logistic-l2", type=float, default=1e-5)
    parser.add_argument("--lambda-grid", type=parse_lambda_grid, default=parse_lambda_grid("0,0.02,0.05,0.1,0.2,0.4,0.7,1.0"))
    args = parser.parse_args()

    results = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print(f"CIFAR-100 Image-BalancedMask-HQS device={results['device']} n_train={results['n_train']} n_eval={results['n_eval']}")
    for name in ["dgm", "online_logistic", "crossed_logistic", "logistic_dgm_trust", "global_frequency"]:
        row = results["metrics"][name]
        print(
            f"  {name:18s} heldout_nll={row['heldout_nll']:.3f} "
            f"heldout_acc={row['heldout_accuracy']:.3f} preq_nll={row['prequential_nll']:.3f}"
        )
    print(f"  concepts={results['diagnostics']['actual_concepts']:.0f} avg_samples={results['diagnostics']['avg_samples_per_concept']:.2f}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
