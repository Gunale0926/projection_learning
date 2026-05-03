from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


MNIST_ROOT = "/Users/gunale/works/silifen-works/haoran_idea/data"
CIFAR10_ROOT = "/Users/gunale/works/silifen-works/SLN/data/cifar10"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but torch.backends.mps.is_available() is false. "
                "In Codex this can happen inside the sandbox; run the command outside "
                "the sandbox/escalated so Metal is visible."
            )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_arrays(name: str, root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if name == "MNIST":
        train = datasets.MNIST(root=root, train=True, download=False)
        test = datasets.MNIST(root=root, train=False, download=False)
        x_train = train.data.numpy().astype("float32")[:, None, :, :] / 255.0
        x_test = test.data.numpy().astype("float32")[:, None, :, :] / 255.0
        mean = np.array([0.1307], dtype="float32")[None, :, None, None]
        std = np.array([0.3081], dtype="float32")[None, :, None, None]
        return (x_train - mean) / std, train.targets.numpy().astype("int64"), (x_test - mean) / std, test.targets.numpy().astype("int64")
    if name == "CIFAR10":
        train = datasets.CIFAR10(root=root, train=True, download=False)
        test = datasets.CIFAR10(root=root, train=False, download=False)
        x_train = np.asarray(train.data).astype("float32").transpose(0, 3, 1, 2) / 255.0
        x_test = np.asarray(test.data).astype("float32").transpose(0, 3, 1, 2) / 255.0
        mean = np.array([0.4914, 0.4822, 0.4465], dtype="float32")[None, :, None, None]
        std = np.array([0.2470, 0.2435, 0.2616], dtype="float32")[None, :, None, None]
        return (x_train - mean) / std, np.asarray(train.targets, dtype="int64"), (x_test - mean) / std, np.asarray(test.targets, dtype="int64")
    raise ValueError(f"unknown dataset: {name}")


def stratified_subset(y: np.ndarray, n_per_class: int | None, rng: np.random.Generator) -> np.ndarray:
    if n_per_class is None:
        out = np.arange(len(y), dtype=np.int64)
        rng.shuffle(out)
        return out
    out: list[int] = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == int(label))
        if len(idx) < n_per_class:
            raise ValueError(f"class {label} has only {len(idx)} examples, need {n_per_class}")
        out.extend(rng.choice(idx, size=n_per_class, replace=False).tolist())
    rng.shuffle(out)
    return np.asarray(out, dtype=np.int64)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


class SmallConvNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 10, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 160, 3, padding=1),
            nn.GroupNorm(8, 160),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(160, embedding_dim),
        )
        self.head = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = nn.functional.normalize(h, dim=1)
        return z, self.head(h)


class DGMConceptMemory:
    """Detached online concept memory with class-local prototype refinement.

    The memory predicts from the current embedding before observing the label,
    then updates its concept prototypes with detached embeddings. Slow network
    parameters can still train through the read logits because prototypes are
    constants during each forward pass.
    """

    def __init__(
        self,
        n_classes: int,
        embedding_dim: int,
        slots_per_class: int,
        temperature: float,
        create_threshold: float,
        update_mode: str,
        device: torch.device,
    ) -> None:
        self.n_classes = int(n_classes)
        self.embedding_dim = int(embedding_dim)
        self.slots_per_class = int(slots_per_class)
        self.temperature = float(temperature)
        self.create_threshold = float(create_threshold)
        if update_mode not in {"exact", "fast"}:
            raise ValueError(f"unknown update mode: {update_mode}")
        self.update_mode = update_mode
        self.device = device
        self.prototypes = torch.zeros(n_classes, slots_per_class, embedding_dim, device=device)
        self.counts = torch.zeros(n_classes, slots_per_class, device=device)
        self.examples_seen = 0

    def reset(self) -> None:
        self.prototypes.zero_()
        self.counts.zero_()
        self.examples_seen = 0

    @property
    def concepts(self) -> int:
        return int((self.counts > 0).sum().item())

    @property
    def estimated_bytes(self) -> int:
        return int(self.prototypes.numel() * self.prototypes.element_size() + self.counts.numel() * self.counts.element_size())

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        if self.examples_seen == 0:
            return torch.zeros(z.shape[0], self.n_classes, device=z.device)
        z = nn.functional.normalize(z, dim=1)
        proto = nn.functional.normalize(self.prototypes, dim=2)
        sims = torch.einsum("bd,ckd->bck", z, proto) / self.temperature
        used = self.counts > 0
        scores = sims + torch.log1p(self.counts).unsqueeze(0)
        scores = scores.masked_fill(~used.unsqueeze(0), -1.0e9)
        logits = torch.logsumexp(scores, dim=2)
        missing_class = ~used.any(dim=1)
        logits = logits.masked_fill(missing_class.unsqueeze(0), 0.0)
        return logits - logits.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def observe(self, z: torch.Tensor, y: torch.Tensor) -> None:
        if self.update_mode == "fast":
            self._observe_fast(z, y)
        else:
            self._observe_exact(z, y)

    @torch.no_grad()
    def _observe_exact(self, z: torch.Tensor, y: torch.Tensor) -> None:
        z = nn.functional.normalize(z.detach(), dim=1)
        y = y.detach()
        self.examples_seen += int(y.numel())
        for cls in range(self.n_classes):
            remaining = z[y == cls]
            if remaining.numel() == 0:
                continue
            row_counts = self.counts[cls]
            used = row_counts > 0
            if not bool(used.any().item()):
                self.prototypes[cls, 0] = remaining[0]
                self.counts[cls, 0] = 1.0
                remaining = remaining[1:]
                used = self.counts[cls] > 0
                if remaining.numel() == 0:
                    continue

            while remaining.numel() > 0:
                unused = torch.nonzero(~used, as_tuple=False).flatten()
                if len(unused) == 0:
                    break
                row_proto = nn.functional.normalize(self.prototypes[cls], dim=1)
                sims = remaining @ row_proto.T
                sims = sims.masked_fill(~used.unsqueeze(0), -1.0e9)
                best_sim = torch.max(sims, dim=1).values
                novel = best_sim < self.create_threshold
                if not bool(novel.any().item()):
                    break
                novel_idx = int(torch.nonzero(novel, as_tuple=False).flatten()[0].item())
                slot = int(unused[0].item())
                self.prototypes[cls, slot] = remaining[novel_idx]
                self.counts[cls, slot] = 1.0
                keep = torch.ones(remaining.shape[0], dtype=torch.bool, device=remaining.device)
                keep[novel_idx] = False
                remaining = remaining[keep]
                used = self.counts[cls] > 0

            if remaining.numel() == 0:
                continue
            used_idx = torch.nonzero(used, as_tuple=False).flatten()
            row_proto = nn.functional.normalize(self.prototypes[cls, used_idx], dim=1)
            assign = torch.argmax(remaining @ row_proto.T, dim=1)
            for local_slot, slot_tensor in enumerate(used_idx):
                group = remaining[assign == local_slot]
                if group.numel() == 0:
                    continue
                slot = int(slot_tensor.item())
                n_old = self.counts[cls, slot]
                n_new = float(group.shape[0])
                updated = (n_old * self.prototypes[cls, slot] + group.sum(dim=0)) / (n_old + n_new)
                self.prototypes[cls, slot] = nn.functional.normalize(updated, dim=0)
                self.counts[cls, slot] = n_old + n_new

    @torch.no_grad()
    def _observe_fast(self, z: torch.Tensor, y: torch.Tensor) -> None:
        z = nn.functional.normalize(z.detach(), dim=1)
        y = y.detach()
        self.examples_seen += int(y.numel())
        for cls in range(self.n_classes):
            remaining = z[y == cls]
            if remaining.numel() == 0:
                continue
            row_counts = self.counts[cls]
            used = row_counts > 0
            if not bool(used.any().item()):
                self.prototypes[cls, 0] = remaining[0]
                self.counts[cls, 0] = 1.0
                remaining = remaining[1:]
                used = self.counts[cls] > 0
                if remaining.numel() == 0:
                    continue

            unused = torch.nonzero(~used, as_tuple=False).flatten()
            if len(unused) > 0:
                row_proto = nn.functional.normalize(self.prototypes[cls], dim=1)
                sims = remaining @ row_proto.T
                sims = sims.masked_fill(~used.unsqueeze(0), -1.0e9)
                best_sim = torch.max(sims, dim=1).values
                novel = best_sim < self.create_threshold
                novel_idx = torch.nonzero(novel, as_tuple=False).flatten()
                if len(novel_idx) > 0:
                    seed_count = min(len(unused), len(novel_idx))
                    if len(novel_idx) > seed_count:
                        selected = torch.topk(-best_sim[novel_idx], k=seed_count).indices
                        seed_idx = novel_idx[selected]
                    else:
                        seed_idx = novel_idx[:seed_count]
                    seed_slots = unused[:seed_count]
                    self.prototypes[cls, seed_slots] = remaining[seed_idx]
                    self.counts[cls, seed_slots] = 1.0
                    keep = torch.ones(remaining.shape[0], dtype=torch.bool, device=remaining.device)
                    keep[seed_idx] = False
                    remaining = remaining[keep]
                    used = self.counts[cls] > 0

            if remaining.numel() == 0:
                continue
            used_idx = torch.nonzero(used, as_tuple=False).flatten()
            row_proto = nn.functional.normalize(self.prototypes[cls, used_idx], dim=1)
            assign = torch.argmax(remaining @ row_proto.T, dim=1)
            num_used = len(used_idx)
            assign_onehot = nn.functional.one_hot(assign, num_classes=num_used).to(dtype=remaining.dtype)
            add_counts = assign_onehot.sum(dim=0)
            nonzero = add_counts > 0
            if not bool(nonzero.any().item()):
                continue
            sums = assign_onehot.T @ remaining
            slots = used_idx[nonzero]
            n_old = self.counts[cls, slots].unsqueeze(1)
            n_add = add_counts[nonzero].unsqueeze(1)
            updated = (n_old * self.prototypes[cls, slots] + sums[nonzero]) / (n_old + n_add)
            self.prototypes[cls, slots] = nn.functional.normalize(updated, dim=1)
            self.counts[cls, slots] = self.counts[cls, slots] + add_counts[nonzero]


class FullDGMRefineMemory:
    """Edge-gated graph memory with local repair and guarded split refinement.

    Concepts are not class-indexed. Each concept stores a prototype, a local
    label-count memory, a frozen anchor, and a bounded validation buffer.
    Observation first repairs the routed concept. When the buffer shows a stable
    log-loss gain from a local distinction, the concept is split and one or more
    accepted distinction edges are added. Prediction retrieves nearby concepts
    by movable centroids, then gates those candidates by the accepted edge
    predicates before mixing compatible local memories. This is the practical
    Soft-DGM graph route used by the experiments.
    """

    def __init__(
        self,
        n_classes: int,
        embedding_dim: int,
        max_concepts: int,
        buffer_size: int,
        top_k: int,
        temperature: float,
        alpha: float,
        min_buffer: int,
        min_child: int,
        split_penalty: float,
        max_splits_per_batch: int,
        device: torch.device,
        edge_degree: int = 1,
        min_edge_divergence: float = 0.0,
        max_incident_edges: int = 128,
    ) -> None:
        self.n_classes = int(n_classes)
        self.embedding_dim = int(embedding_dim)
        self.max_concepts = int(max_concepts)
        self.buffer_size = int(buffer_size)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.min_buffer = int(min_buffer)
        self.min_child = int(min_child)
        self.split_penalty = float(split_penalty)
        self.max_splits_per_batch = int(max_splits_per_batch)
        self.edge_degree = max(1, int(edge_degree))
        self.min_edge_divergence = float(min_edge_divergence)
        self.max_incident_edges = max(1, int(max_incident_edges))
        self.max_edges = self.max_concepts * self.edge_degree
        self.device = device

        self.anchors = torch.zeros(max_concepts, embedding_dim, device=device)
        self.prototypes = torch.zeros(max_concepts, embedding_dim, device=device)
        self.counts = torch.zeros(max_concepts, n_classes, device=device)
        self.totals = torch.zeros(max_concepts, device=device)
        self.active = torch.zeros(max_concepts, dtype=torch.bool, device=device)
        self.anchor_set = torch.zeros(max_concepts, dtype=torch.bool, device=device)
        self.buffer_z = torch.zeros(max_concepts, buffer_size, embedding_dim, device=device)
        self.buffer_y = torch.zeros(max_concepts, buffer_size, dtype=torch.long, device=device)
        self.buffer_n = torch.zeros(max_concepts, dtype=torch.long, device=device)
        self.buffer_pos = torch.zeros(max_concepts, dtype=torch.long, device=device)
        self.parents = torch.full((max_concepts,), -1, dtype=torch.long, device=device)
        self.edge_gain = torch.zeros(max_concepts, device=device)
        self.edge_u = torch.zeros(self.max_edges, embedding_dim, device=device)
        self.edge_b = torch.zeros(self.max_edges, device=device)
        self.edge_src = torch.full((self.max_edges,), -1, dtype=torch.long, device=device)
        self.edge_dst = torch.full((self.max_edges,), -1, dtype=torch.long, device=device)
        self.concept_edge_ids = torch.full((max_concepts, self.max_incident_edges), -1, dtype=torch.long, device=device)
        self.concept_edge_sides = torch.zeros((max_concepts, self.max_incident_edges), dtype=torch.int8, device=device)
        self.concept_edge_n = torch.zeros(max_concepts, dtype=torch.long, device=device)
        self.examples_seen = 0
        self._concept_count = 0
        self._edge_count = 0
        self._dropped_edges = 0

    def reset(self) -> None:
        self.anchors.zero_()
        self.prototypes.zero_()
        self.counts.zero_()
        self.totals.zero_()
        self.active.zero_()
        self.anchor_set.zero_()
        self.buffer_z.zero_()
        self.buffer_y.zero_()
        self.buffer_n.zero_()
        self.buffer_pos.zero_()
        self.parents.fill_(-1)
        self.edge_gain.zero_()
        self.edge_u.zero_()
        self.edge_b.zero_()
        self.edge_src.fill_(-1)
        self.edge_dst.fill_(-1)
        self.concept_edge_ids.fill_(-1)
        self.concept_edge_sides.zero_()
        self.concept_edge_n.zero_()
        self.examples_seen = 0
        self._concept_count = 0
        self._edge_count = 0
        self._dropped_edges = 0

    @property
    def concepts(self) -> int:
        return self._concept_count

    @property
    def edges(self) -> int:
        return self._edge_count

    @property
    def dropped_edges(self) -> int:
        return self._dropped_edges

    @property
    def estimated_bytes(self) -> int:
        tensors = [
            self.anchors,
            self.prototypes,
            self.counts,
            self.totals,
            self.active,
            self.anchor_set,
            self.buffer_z,
            self.buffer_y,
            self.buffer_n,
            self.buffer_pos,
            self.parents,
            self.edge_gain,
            self.edge_u,
            self.edge_b,
            self.edge_src,
            self.edge_dst,
            self.concept_edge_ids,
            self.concept_edge_sides,
            self.concept_edge_n,
        ]
        return int(sum(t.numel() * t.element_size() for t in tensors))

    @torch.no_grad()
    def compact_for_inference(self) -> None:
        """Drop refinement buffers once the graph is built.

        The buffers are only needed by observe()/try_refine(). Prediction uses
        prototypes, counts, anchors, and accepted edge predicates, so experiments
        can report the deployed memory footprint after this call.
        """
        self.buffer_z = torch.empty(self.max_concepts, 0, self.embedding_dim, device=self.device)
        self.buffer_y = torch.empty(self.max_concepts, 0, dtype=torch.long, device=self.device)
        self.buffer_n.zero_()
        self.buffer_pos.zero_()
        self.buffer_size = 0

    def _active_idx(self) -> torch.Tensor:
        return torch.arange(self._concept_count, device=self.device)

    def _new_concept(self) -> int | None:
        if self._concept_count >= self.max_concepts:
            return None
        idx = self._concept_count
        self.active[idx] = True
        self._concept_count += 1
        return idx

    @torch.no_grad()
    def _set_concept(self, idx: int, z: torch.Tensor, y: torch.Tensor, *, set_anchor: bool) -> None:
        z = nn.functional.normalize(z, dim=1)
        center = nn.functional.normalize(z.mean(dim=0), dim=0)
        self.prototypes[idx] = center
        if set_anchor or not bool(self.anchor_set[idx].item()):
            self.anchors[idx] = center
            self.anchor_set[idx] = True
        self.counts[idx].zero_()
        self.counts[idx].scatter_add_(0, y, torch.ones_like(y, dtype=self.counts.dtype))
        self.totals[idx] = float(y.numel())
        if self.buffer_size == 0:
            self.buffer_n[idx] = 0
            self.buffer_pos[idx] = 0
            return
        n = min(y.numel(), self.buffer_size)
        self.buffer_z[idx].zero_()
        self.buffer_y[idx].zero_()
        self.buffer_z[idx, :n] = z[:n]
        self.buffer_y[idx, :n] = y[:n]
        self.buffer_n[idx] = n
        self.buffer_pos[idx] = n % self.buffer_size

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        active_idx = self._active_idx()
        if len(active_idx) == 0:
            return torch.zeros(z.shape[0], self.n_classes, device=z.device)
        z = nn.functional.normalize(z, dim=1)
        proto = nn.functional.normalize(self.prototypes[active_idx], dim=1)
        sims = z @ proto.T
        k = min(self.top_k, len(active_idx))
        vals, local = torch.topk(sims, k=k, dim=1)
        concept_idx = active_idx[local]
        route_scores = vals
        if self._edge_count > 0:
            edge_ids = self.concept_edge_ids[concept_idx]
            edge_sides = self.concept_edge_sides[concept_idx]
            valid = edge_ids >= 0
            if bool(valid.any().item()):
                needed_edges = torch.unique(edge_ids[valid], sorted=True)
                edge_bits = ((z @ self.edge_u[needed_edges].T) > self.edge_b[needed_edges]).to(torch.int8)
                edge_positions = torch.searchsorted(needed_edges, edge_ids.clamp_min(0))
                batch_idx = torch.arange(z.shape[0], device=z.device)[:, None, None].expand_as(edge_ids)
                candidate_bits = torch.zeros_like(edge_sides)
                candidate_bits[valid] = edge_bits[batch_idx[valid], edge_positions[valid]]
                violated = valid & (candidate_bits != edge_sides)
                eligible = ~violated.any(dim=2)
                has_eligible = eligible.any(dim=1, keepdim=True)
                eligible = torch.where(has_eligible, eligible, torch.ones_like(eligible))
                route_scores = vals.masked_fill(~eligible, -1.0e9)
        local_counts = self.counts[concept_idx]
        local_totals = self.totals[concept_idx].unsqueeze(-1)
        probs = (local_counts + self.alpha / self.n_classes) / (local_totals + self.alpha).clamp_min(1e-6)
        weights = torch.softmax(route_scores / self.temperature, dim=1).unsqueeze(-1)
        p = (weights * probs).sum(dim=1).clamp_min(1e-8)
        logits = torch.log(p)
        return logits - logits.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def _add_edge(self, src: int, dst: int) -> bool:
        if src == dst or self._edge_count >= self.max_edges:
            return False
        if not bool(self.anchor_set[src].item()) or not bool(self.anchor_set[dst].item()):
            return False
        src_slot = int(self.concept_edge_n[src].item())
        dst_slot = int(self.concept_edge_n[dst].item())
        if src_slot >= self.max_incident_edges or dst_slot >= self.max_incident_edges:
            self._dropped_edges += 1
            return False
        existing_src = self.edge_src[: self._edge_count]
        existing_dst = self.edge_dst[: self._edge_count]
        duplicate = ((existing_src == src) & (existing_dst == dst)) | ((existing_src == dst) & (existing_dst == src))
        if bool(duplicate.any().item()):
            return False

        src_anchor = self.anchors[src]
        dst_anchor = self.anchors[dst]
        diff = dst_anchor - src_anchor
        norm = torch.linalg.vector_norm(diff)
        if float(norm.item()) <= 1.0e-6:
            return False
        u = diff / norm
        edge = self._edge_count
        self.edge_u[edge] = u
        self.edge_b[edge] = 0.5 * torch.dot(u, src_anchor + dst_anchor)
        self.edge_src[edge] = src
        self.edge_dst[edge] = dst
        self.concept_edge_ids[src, src_slot] = edge
        self.concept_edge_sides[src, src_slot] = 0
        self.concept_edge_n[src] = src_slot + 1
        self.concept_edge_ids[dst, dst_slot] = edge
        self.concept_edge_sides[dst, dst_slot] = 1
        self.concept_edge_n[dst] = dst_slot + 1
        self._edge_count += 1
        return True

    def _label_distribution(self, concept: int) -> torch.Tensor:
        return (self.counts[concept] + self.alpha / self.n_classes) / (self.totals[concept] + self.alpha).clamp_min(1e-6)

    @torch.no_grad()
    def _add_confused_neighbor_edges(self, child: int, parent: int) -> None:
        extra_edges = self.edge_degree - 1
        if extra_edges <= 0 or self._edge_count >= self.max_edges or self.concepts <= 2:
            return
        active_idx = self._active_idx()
        keep = (active_idx != child) & (active_idx != parent)
        candidates = active_idx[keep]
        if candidates.numel() == 0:
            return
        child_anchor = nn.functional.normalize(self.anchors[child], dim=0)
        candidate_anchors = nn.functional.normalize(self.anchors[candidates], dim=1)
        sims = candidate_anchors @ child_anchor
        order = torch.argsort(sims, descending=True)
        child_prob = self._label_distribution(child)
        added = 0
        for pos in order.tolist():
            candidate = int(candidates[pos].item())
            cand_prob = self._label_distribution(candidate)
            divergence = 0.5 * torch.sum(torch.abs(child_prob - cand_prob))
            if float(divergence.item()) < self.min_edge_divergence:
                continue
            if self._add_edge(candidate, child):
                added += 1
                if added >= extra_edges:
                    break

    @torch.no_grad()
    def observe(self, z: torch.Tensor, y: torch.Tensor) -> None:
        z = nn.functional.normalize(z.detach(), dim=1)
        y = y.detach()
        self.examples_seen += int(y.numel())
        if self.concepts == 0:
            idx = self._new_concept()
            if idx is not None:
                self._set_concept(idx, z, y, set_anchor=True)
            return

        active_idx = self._active_idx()
        sims = z @ nn.functional.normalize(self.prototypes[active_idx], dim=1).T
        routed = active_idx[torch.argmax(sims, dim=1)]
        touched: list[int] = []
        for concept_tensor in torch.unique(routed):
            concept = int(concept_tensor.item())
            mask = routed == concept
            self._repair_concept(concept, z[mask], y[mask])
            touched.append(concept)

        splits = 0
        for concept in touched:
            if splits >= self.max_splits_per_batch:
                break
            if self._try_refine(concept):
                splits += 1

    @torch.no_grad()
    def _repair_concept(self, concept: int, z: torch.Tensor, y: torch.Tensor) -> None:
        if y.numel() == 0:
            return
        old_total = self.totals[concept]
        add_total = float(y.numel())
        updated = (old_total * self.prototypes[concept] + z.sum(dim=0)) / (old_total + add_total).clamp_min(1.0)
        self.prototypes[concept] = nn.functional.normalize(updated, dim=0)
        self.counts[concept].scatter_add_(0, y, torch.ones_like(y, dtype=self.counts.dtype))
        self.totals[concept] = old_total + add_total
        self._append_buffer(concept, z, y)

    @torch.no_grad()
    def _append_buffer(self, concept: int, z: torch.Tensor, y: torch.Tensor) -> None:
        m = int(y.numel())
        if m == 0 or self.buffer_size == 0:
            return
        n_write = min(m, self.buffer_size)
        z_write = z[-n_write:]
        y_write = y[-n_write:]
        pos = int(self.buffer_pos[concept].item())
        slots = (torch.arange(n_write, device=self.device) + pos) % self.buffer_size
        self.buffer_z[concept, slots] = z_write
        self.buffer_y[concept, slots] = y_write
        self.buffer_pos[concept] = (pos + n_write) % self.buffer_size
        self.buffer_n[concept] = min(int(self.buffer_n[concept].item()) + n_write, self.buffer_size)

    @staticmethod
    def _entropy(counts: torch.Tensor) -> torch.Tensor:
        total = counts.sum()
        if float(total.item()) <= 0.0:
            return torch.zeros((), device=counts.device)
        p = counts[counts > 0] / total
        return -(p * torch.log(p)).sum()

    @torch.no_grad()
    def _try_refine(self, concept: int) -> bool:
        n = int(self.buffer_n[concept].item())
        if n < self.min_buffer or self.concepts >= self.max_concepts:
            return False
        bz = self.buffer_z[concept, :n]
        by = self.buffer_y[concept, :n]
        parent_counts = torch.bincount(by, minlength=self.n_classes).to(dtype=self.counts.dtype, device=self.device)
        if int((parent_counts > 0).sum().item()) <= 1:
            return False

        parent_prob = (parent_counts + self.alpha / self.n_classes) / (parent_counts.sum() + self.alpha)
        surprise = -torch.log(parent_prob[by].clamp_min(1e-8))
        k = max(self.min_child, min(n // 4, 32))
        hard_idx = torch.topk(surprise, k=k).indices
        candidate = nn.functional.normalize(bz[hard_idx].mean(dim=0), dim=0)
        parent = nn.functional.normalize(self.prototypes[concept], dim=0)
        right = (bz @ candidate) > (bz @ parent)
        right_n = int(right.sum().item())
        left_n = n - right_n
        if left_n < self.min_child or right_n < self.min_child:
            return False

        left_counts = torch.bincount(by[~right], minlength=self.n_classes).to(dtype=self.counts.dtype, device=self.device)
        right_counts = torch.bincount(by[right], minlength=self.n_classes).to(dtype=self.counts.dtype, device=self.device)
        parent_risk = n * self._entropy(parent_counts)
        child_risk = left_n * self._entropy(left_counts) + right_n * self._entropy(right_counts)
        gain = parent_risk - child_risk
        if float(gain.item()) <= self.split_penalty:
            return False

        child = self._new_concept()
        if child is None:
            return False
        self.parents[child] = concept
        self.edge_gain[child] = gain

        self._set_concept(child, bz[right], by[right], set_anchor=True)
        self._set_concept(concept, bz[~right], by[~right], set_anchor=False)
        self._add_edge(concept, child)
        self._add_confused_neighbor_edges(child, concept)
        return True


@dataclass
class Metrics:
    accuracy: float
    nll: float


@dataclass
class RunResult:
    dataset: str
    method: str
    seed: int
    epochs: int
    train_examples: int
    test_examples: int
    batch_size: int
    train_prequential_accuracy: float
    train_prequential_nll: float
    test_accuracy: float
    test_nll: float
    train_seconds: float
    eval_seconds: float
    concepts: int
    edges: int
    dropped_edges: int
    memory_examples_seen: int
    memory_bytes: int
    device: str


class MetricTotals:
    def __init__(self, device: torch.device) -> None:
        self.nll = torch.zeros((), device=device)
        self.correct = torch.zeros((), device=device)
        self.count = 0

    @torch.no_grad()
    def observe(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        self.nll += nn.functional.cross_entropy(logits, y, reduction="sum").detach()
        pred = torch.argmax(logits.detach(), dim=1)
        self.correct += (pred == y).sum()
        self.count += int(y.numel())

    def accuracy(self) -> float:
        if self.count == 0:
            return 0.0
        return float((self.correct / self.count).item())

    def nll_value(self) -> float:
        if self.count == 0:
            return 0.0
        return float((self.nll / self.count).item())


@torch.no_grad()
def build_memory(
    model: SmallConvNet,
    loader: DataLoader,
    memory: DGMConceptMemory,
    device: torch.device,
) -> None:
    memory.reset()
    model.eval()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        z, _ = model(xb)
        memory.observe(z, yb)


@torch.no_grad()
def evaluate(
    model: SmallConvNet,
    loader: DataLoader,
    device: torch.device,
    memory: DGMConceptMemory | None,
    memory_weight: float,
) -> Metrics:
    model.eval()
    totals = MetricTotals(device)
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        z, base_logits = model(xb)
        logits = base_logits if memory is None else base_logits + memory_weight * memory.logits(z)
        totals.observe(logits, yb)
    return Metrics(
        accuracy=totals.accuracy(),
        nll=totals.nll_value(),
    )


def train_one(
    dataset: str,
    method: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    embedding_dim: int,
    slots_per_class: int,
    memory_weight: float,
    memory_temperature: float,
    create_threshold: float,
    memory_update_mode: str,
    full_dgm_max_concepts: int,
    full_dgm_buffer_size: int,
    full_dgm_top_k: int,
    full_dgm_min_buffer: int,
    full_dgm_min_child: int,
    full_dgm_split_penalty: float,
    full_dgm_max_splits_per_batch: int,
    full_dgm_edge_degree: int,
    full_dgm_min_edge_divergence: float,
    full_dgm_max_incident_edges: int,
    device: torch.device,
) -> RunResult:
    set_seed(seed)
    model = SmallConvNet(in_channels=int(x_train.shape[1]), embedding_dim=embedding_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed + 17)
    train_eval_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=False, seed=seed + 19)
    test_loader = make_loader(x_test, y_test, batch_size=batch_size, shuffle=False, seed=seed + 23)
    memory = None
    if method == "dgm_memory":
        memory = DGMConceptMemory(
            n_classes=10,
            embedding_dim=embedding_dim,
            slots_per_class=slots_per_class,
            temperature=memory_temperature,
            create_threshold=create_threshold,
            update_mode=memory_update_mode,
            device=device,
        )
    elif method == "full_dgm":
        memory = FullDGMRefineMemory(
            n_classes=10,
            embedding_dim=embedding_dim,
            max_concepts=full_dgm_max_concepts,
            buffer_size=full_dgm_buffer_size,
            top_k=full_dgm_top_k,
            temperature=memory_temperature,
            alpha=1.0,
            min_buffer=full_dgm_min_buffer,
            min_child=full_dgm_min_child,
            split_penalty=full_dgm_split_penalty,
            max_splits_per_batch=full_dgm_max_splits_per_batch,
            device=device,
            edge_degree=full_dgm_edge_degree,
            min_edge_divergence=full_dgm_min_edge_divergence,
            max_incident_edges=full_dgm_max_incident_edges,
        )
    train_totals = MetricTotals(device)
    start_train = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        if memory is not None:
            memory.reset()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            z, base_logits = model(xb)
            logits = base_logits if memory is None else base_logits + memory_weight * memory.logits(z)
            train_totals.observe(logits, yb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if memory is not None:
                memory.observe(z, yb)
        print(
            f"{dataset} seed={seed} method={method} epoch={epoch + 1}/{epochs} "
            f"preq_acc={train_totals.accuracy():.3f}",
            flush=True,
        )
    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start_train

    eval_start = time.perf_counter()
    eval_memory = None
    if memory is not None:
        if method == "dgm_memory":
            eval_memory = DGMConceptMemory(
                n_classes=10,
                embedding_dim=embedding_dim,
                slots_per_class=slots_per_class,
                temperature=memory_temperature,
                create_threshold=create_threshold,
                update_mode=memory_update_mode,
                device=device,
            )
        elif method == "full_dgm":
            eval_memory = FullDGMRefineMemory(
                n_classes=10,
                embedding_dim=embedding_dim,
                max_concepts=full_dgm_max_concepts,
                buffer_size=full_dgm_buffer_size,
                top_k=full_dgm_top_k,
                temperature=memory_temperature,
                alpha=1.0,
                min_buffer=full_dgm_min_buffer,
                min_child=full_dgm_min_child,
                split_penalty=full_dgm_split_penalty,
                max_splits_per_batch=full_dgm_max_splits_per_batch,
                device=device,
                edge_degree=full_dgm_edge_degree,
                min_edge_divergence=full_dgm_min_edge_divergence,
                max_incident_edges=full_dgm_max_incident_edges,
            )
        build_memory(model, train_eval_loader, eval_memory, device)
    test = evaluate(model, test_loader, device, eval_memory, memory_weight)
    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    return RunResult(
        dataset=dataset,
        method=method,
        seed=seed,
        epochs=epochs,
        train_examples=int(len(y_train)),
        test_examples=int(len(y_test)),
        batch_size=batch_size,
        train_prequential_accuracy=train_totals.accuracy(),
        train_prequential_nll=train_totals.nll_value(),
        test_accuracy=test.accuracy,
        test_nll=test.nll,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        concepts=0 if eval_memory is None else eval_memory.concepts,
        edges=0 if eval_memory is None else int(getattr(eval_memory, "edges", 0)),
        dropped_edges=0 if eval_memory is None else int(getattr(eval_memory, "dropped_edges", 0)),
        memory_examples_seen=0 if eval_memory is None else eval_memory.examples_seen,
        memory_bytes=0 if eval_memory is None else eval_memory.estimated_bytes,
        device=str(device),
    )


def summarize(rows: Iterable[RunResult]) -> dict[str, float]:
    row_list = list(rows)
    out: dict[str, float] = {}
    for key in [
        "train_prequential_accuracy",
        "train_prequential_nll",
        "test_accuracy",
        "test_nll",
        "train_seconds",
        "eval_seconds",
        "concepts",
        "edges",
        "dropped_edges",
        "memory_examples_seen",
        "memory_bytes",
    ]:
        values = np.asarray([float(getattr(row, key)) for row in row_list], dtype=float)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    return out


def dataset_epochs(name: str, quick: bool, override: int | None) -> int:
    if override is not None:
        return int(override)
    if quick:
        return 1
    return 2 if name == "MNIST" else 4


def dataset_train_per_class(name: str, quick: bool, override: int | None) -> int | None:
    if override is not None:
        return int(override)
    if quick:
        return 600 if name == "MNIST" else 800
    return None


def dataset_test_per_class(name: str, quick: bool, override: int | None) -> int | None:
    if override is not None:
        return int(override)
    if quick:
        return 200
    return None


def parse_dataset_list(raw: str) -> list[str]:
    values = [item.strip().upper() for item in raw.split(",") if item.strip()]
    aliases = {"CIFAR": "CIFAR10"}
    return [aliases.get(item, item) for item in values]


def parse_methods(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"backprop", "dgm_memory", "full_dgm"}
    bad = [item for item in values if item not in allowed]
    if bad:
        raise ValueError(f"unknown methods: {bad}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="MNIST,CIFAR10")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--methods", type=str, default="backprop,dgm_memory")
    parser.add_argument("--output", type=Path, default=Path("experiments/results_real_training.json"))
    parser.add_argument("--mnist-root", type=str, default=MNIST_ROOT)
    parser.add_argument("--cifar-root", type=str, default=CIFAR10_ROOT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--slots-per-class", type=int, default=16)
    parser.add_argument("--memory-weight", type=float, default=0.6)
    parser.add_argument("--memory-temperature", type=float, default=0.18)
    parser.add_argument("--create-threshold", type=float, default=0.72)
    parser.add_argument("--memory-update", choices=["exact", "fast"], default="exact")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    parser.add_argument("--full-dgm-max-concepts", type=int, default=256)
    parser.add_argument("--full-dgm-buffer-size", type=int, default=96)
    parser.add_argument("--full-dgm-top-k", type=int, default=8)
    parser.add_argument("--full-dgm-min-buffer", type=int, default=32)
    parser.add_argument("--full-dgm-min-child", type=int, default=6)
    parser.add_argument("--full-dgm-split-penalty", type=float, default=3.0)
    parser.add_argument("--full-dgm-max-splits-per-batch", type=int, default=4)
    parser.add_argument("--full-dgm-edge-degree", type=int, default=2)
    parser.add_argument("--full-dgm-min-edge-divergence", type=float, default=0.05)
    parser.add_argument("--full-dgm-max-incident-edges", type=int, default=128)
    args = parser.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = select_device(args.device)
    datasets_to_run = parse_dataset_list(args.datasets)
    methods = parse_methods(args.methods)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    all_runs: list[RunResult] = []
    by_dataset: dict[str, object] = {}
    roots = {"MNIST": args.mnist_root, "CIFAR10": args.cifar_root}
    for dataset_name in datasets_to_run:
        rng = np.random.default_rng(10_000 + len(all_runs))
        x_train_all, y_train_all, x_test_all, y_test_all = load_arrays(dataset_name, roots[dataset_name])
        train_idx = stratified_subset(
            y_train_all,
            dataset_train_per_class(dataset_name, args.quick, args.train_per_class),
            rng,
        )
        test_idx = stratified_subset(
            y_test_all,
            dataset_test_per_class(dataset_name, args.quick, args.test_per_class),
            rng,
        )
        x_train = x_train_all[train_idx]
        y_train = y_train_all[train_idx]
        x_test = x_test_all[test_idx]
        y_test = y_test_all[test_idx]
        epochs = dataset_epochs(dataset_name, args.quick, args.epochs)
        print(
            f"Running {dataset_name}: train={len(y_train)} test={len(y_test)} epochs={epochs} "
            f"device={device}",
            flush=True,
        )
        dataset_runs: list[RunResult] = []
        for seed in seeds:
            for method in methods:
                run = train_one(
                    dataset=dataset_name,
                    method=method,
                    seed=seed,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    embedding_dim=args.embedding_dim,
                    slots_per_class=args.slots_per_class,
                    memory_weight=args.memory_weight,
                    memory_temperature=args.memory_temperature,
                    create_threshold=args.create_threshold,
                    memory_update_mode=args.memory_update,
                    full_dgm_max_concepts=args.full_dgm_max_concepts,
                    full_dgm_buffer_size=args.full_dgm_buffer_size,
                    full_dgm_top_k=args.full_dgm_top_k,
                    full_dgm_min_buffer=args.full_dgm_min_buffer,
                    full_dgm_min_child=args.full_dgm_min_child,
                    full_dgm_split_penalty=args.full_dgm_split_penalty,
                    full_dgm_max_splits_per_batch=args.full_dgm_max_splits_per_batch,
                    full_dgm_edge_degree=args.full_dgm_edge_degree,
                    full_dgm_min_edge_divergence=args.full_dgm_min_edge_divergence,
                    full_dgm_max_incident_edges=args.full_dgm_max_incident_edges,
                    device=device,
                )
                dataset_runs.append(run)
                all_runs.append(run)
                print(
                    f"RESULT {dataset_name} seed={seed} method={method} "
                    f"test_acc={run.test_accuracy:.4f} test_nll={run.test_nll:.4f} "
                    f"train_seconds={run.train_seconds:.1f}",
                    flush=True,
                )
                args.output.with_suffix(args.output.suffix + ".partial").write_text(
                    json.dumps(
                        {
                            "device": str(device),
                            "mps_built": bool(torch.backends.mps.is_built()),
                            "mps_available": bool(torch.backends.mps.is_available()),
                            "torch_version": torch.__version__,
                            "quick": bool(args.quick),
                            "seeds": seeds,
                            "methods": methods,
                            "memory_update": args.memory_update,
                            "runs": [asdict(row) for row in all_runs],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
        by_dataset[dataset_name] = {
            "epochs": epochs,
            "train_examples": int(len(y_train)),
            "test_examples": int(len(y_test)),
            **{method: summarize(row for row in dataset_runs if row.method == method) for method in methods},
        }

    output = {
        "protocol": (
            "Real supervised training benchmark. Backprop uses a CNN encoder plus softmax head. "
            "DGM-memory uses the same slow CNN and head, plus a detached online concept-memory read "
            "before each label is observed; concept memory is rebuilt from the training set for test evaluation."
        ),
        "device": str(device),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "torch_version": torch.__version__,
        "quick": bool(args.quick),
        "seeds": seeds,
        "methods": methods,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "embedding_dim": args.embedding_dim,
            "slots_per_class": args.slots_per_class,
            "memory_weight": args.memory_weight,
            "memory_temperature": args.memory_temperature,
            "create_threshold": args.create_threshold,
            "memory_update": args.memory_update,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "device_requested": args.device,
            "epochs_override": args.epochs,
            "train_per_class_override": args.train_per_class,
            "test_per_class_override": args.test_per_class,
            "full_dgm_max_concepts": args.full_dgm_max_concepts,
            "full_dgm_buffer_size": args.full_dgm_buffer_size,
            "full_dgm_top_k": args.full_dgm_top_k,
            "full_dgm_min_buffer": args.full_dgm_min_buffer,
            "full_dgm_min_child": args.full_dgm_min_child,
            "full_dgm_split_penalty": args.full_dgm_split_penalty,
            "full_dgm_max_splits_per_batch": args.full_dgm_max_splits_per_batch,
            "full_dgm_edge_degree": args.full_dgm_edge_degree,
            "full_dgm_min_edge_divergence": args.full_dgm_min_edge_divergence,
            "full_dgm_max_incident_edges": args.full_dgm_max_incident_edges,
        },
        "datasets": by_dataset,
        "runs": [asdict(row) for row in all_runs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    for dataset_name, table in by_dataset.items():
        parts = []
        for method in methods:
            row = table[method]
            parts.append(
                f"{method} acc={row['test_accuracy_mean']:.4f}±{row['test_accuracy_std']:.4f} "
                f"preq_nll={row['train_prequential_nll_mean']:.4f}"
            )
        print(f"{dataset_name}: " + "; ".join(parts), flush=True)


if __name__ == "__main__":
    main()
