from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


Reduction = Literal["none", "mean", "sum"]
UpdateMode = Literal["hard", "soft"]


@dataclass(frozen=True)
class DGMStateSummary:
    concepts: int
    edges: int
    dim: int
    classes: int


class CategoricalDGM(nn.Module):
    """PyTorch reference implementation of a stateful categorical DGM.

    The class intentionally keeps the dynamic mutation path in Python and the
    differentiable read path in PyTorch. `predict` and `loss` never mutate state;
    `observe` performs the detached prequential write.
    """

    def __init__(
        self,
        dim: int,
        n_classes: int,
        *,
        k: int = 8,
        alpha: float = 1.0,
        distance_temperature: float = 1.0,
        edge_temperature: float = 8.0,
        edge_weight: float = 1.0,
        centroid_lr: float = 0.2,
        refine_on_error: bool = True,
        min_refine_total: float = 0.0,
        refine_loss_threshold: float = 0.0,
        max_concepts: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if n_classes <= 1:
            raise ValueError("n_classes must be greater than one")
        if k <= 0:
            raise ValueError("k must be positive")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if distance_temperature <= 0.0:
            raise ValueError("distance_temperature must be positive")
        if edge_temperature <= 0.0:
            raise ValueError("edge_temperature must be positive")
        if not (0.0 <= centroid_lr <= 1.0):
            raise ValueError("centroid_lr must lie in [0, 1]")
        if min_refine_total < 0.0:
            raise ValueError("min_refine_total must be nonnegative")
        if refine_loss_threshold < 0.0:
            raise ValueError("refine_loss_threshold must be nonnegative")

        self.dim = int(dim)
        self.n_classes = int(n_classes)
        self.k = int(k)
        self.alpha = float(alpha)
        self.distance_temperature = float(distance_temperature)
        self.edge_temperature = float(edge_temperature)
        self.edge_weight = float(edge_weight)
        self.centroid_lr = float(centroid_lr)
        self.refine_on_error = bool(refine_on_error)
        self.min_refine_total = float(min_refine_total)
        self.refine_loss_threshold = float(refine_loss_threshold)
        self.max_concepts = None if max_concepts is None else int(max_concepts)
        self.eps = float(eps)

        self.register_buffer("anchors", torch.empty(0, self.dim))
        self.register_buffer("centroids", torch.empty(0, self.dim))
        self.register_buffer("counts", torch.empty(0, self.n_classes))
        self.register_buffer("totals", torch.empty(0))
        self.register_buffer("edge_u", torch.empty(0, self.dim))
        self.register_buffer("edge_b", torch.empty(0))
        self.register_buffer("edge_src", torch.empty(0, dtype=torch.long))
        self.register_buffer("edge_dst", torch.empty(0, dtype=torch.long))

    @property
    def num_concepts(self) -> int:
        return int(self.anchors.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_src.shape[0])

    def summary(self) -> DGMStateSummary:
        return DGMStateSummary(
            concepts=self.num_concepts,
            edges=self.num_edges,
            dim=self.dim,
            classes=self.n_classes,
        )

    def add_concept(self, h: torch.Tensor, y: int | torch.Tensor | None = None) -> int:
        h_vec = self._single_h(h).detach().to(device=self.anchors.device, dtype=self.anchors.dtype)
        if self.max_concepts is not None and self.num_concepts >= self.max_concepts:
            raise RuntimeError("max_concepts has been reached")

        self.anchors = torch.cat([self.anchors, h_vec.unsqueeze(0)], dim=0)
        self.centroids = torch.cat([self.centroids, h_vec.unsqueeze(0)], dim=0)
        new_counts = torch.zeros(1, self.n_classes, device=self.counts.device, dtype=self.counts.dtype)
        new_total = torch.zeros(1, device=self.totals.device, dtype=self.totals.dtype)
        self.counts = torch.cat([self.counts, new_counts], dim=0)
        self.totals = torch.cat([self.totals, new_total], dim=0)

        index = self.num_concepts - 1
        if y is not None:
            self._add_counts(index, int(torch.as_tensor(y).item()), 1.0)
        return index

    def add_edge(
        self,
        src: int,
        dst: int,
        *,
        u: torch.Tensor | None = None,
        b: float | torch.Tensor | None = None,
    ) -> int:
        src = int(src)
        dst = int(dst)
        if src == dst:
            raise ValueError("edge endpoints must be distinct")
        if not (0 <= src < self.num_concepts and 0 <= dst < self.num_concepts):
            raise IndexError("edge endpoint out of range")
        existing = (self.edge_src == src) & (self.edge_dst == dst)
        reverse = (self.edge_src == dst) & (self.edge_dst == src)
        found = torch.nonzero(existing | reverse, as_tuple=False).flatten()
        if found.numel() > 0:
            return int(found[0].item())

        if u is None:
            raw_u = self.anchors[dst] - self.anchors[src]
            norm = torch.linalg.vector_norm(raw_u).clamp_min(self.eps)
            edge_u = raw_u / norm
        else:
            edge_u = self._single_h(u).detach().to(device=self.edge_u.device, dtype=self.edge_u.dtype)
        if b is None:
            midpoint = 0.5 * (self.anchors[src] + self.anchors[dst])
            edge_b = -(edge_u * midpoint).sum()
        else:
            edge_b = torch.as_tensor(b, device=self.edge_b.device, dtype=self.edge_b.dtype)

        self.edge_u = torch.cat([self.edge_u, edge_u.unsqueeze(0)], dim=0)
        self.edge_b = torch.cat([self.edge_b, edge_b.reshape(1)], dim=0)
        self.edge_src = torch.cat(
            [self.edge_src, torch.tensor([src], device=self.edge_src.device, dtype=torch.long)],
            dim=0,
        )
        self.edge_dst = torch.cat(
            [self.edge_dst, torch.tensor([dst], device=self.edge_dst.device, dtype=torch.long)],
            dim=0,
        )
        return self.num_edges - 1

    def predict(
        self,
        h: torch.Tensor,
        *,
        p0_logits: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h_batch = self._batch_h(h)
        p0 = self._prior_probs(h_batch, p0_logits)
        batch = h_batch.shape[0]

        if self.num_concepts == 0:
            empty_idx = torch.empty(batch, 0, device=h_batch.device, dtype=torch.long)
            empty_float = torch.empty(batch, 0, device=h_batch.device, dtype=h_batch.dtype)
            aux = {
                "candidate_idx": empty_idx,
                "routing_logits": empty_float,
                "routing_weights": empty_float,
                "candidate_probs": torch.empty(batch, 0, self.n_classes, device=h_batch.device, dtype=h_batch.dtype),
                "selected_idx": torch.full((batch,), -1, device=h_batch.device, dtype=torch.long),
                "probs": p0,
            }
            return (p0, aux) if return_aux else p0

        cand_idx = self._candidate_indices(h_batch)
        cand_centroids = self.centroids.to(h_batch.device)[cand_idx]
        sq_dist = ((h_batch[:, None, :] - cand_centroids) ** 2).sum(dim=-1)
        routing_logits = -sq_dist / self.distance_temperature
        routing_logits = routing_logits + self._edge_log_compatibility(h_batch, cand_idx)
        routing_weights = F.softmax(routing_logits, dim=1)

        cand_counts = self.counts.to(h_batch.device)[cand_idx]
        cand_totals = self.totals.to(h_batch.device)[cand_idx]
        candidate_probs = (cand_counts + self.alpha * p0[:, None, :]) / (
            cand_totals[:, :, None] + self.alpha
        ).clamp_min(self.eps)
        probs = (routing_weights[:, :, None] * candidate_probs).sum(dim=1)
        probs = probs.clamp_min(self.eps)
        probs = probs / probs.sum(dim=1, keepdim=True)

        selected_pos = torch.argmax(routing_weights, dim=1)
        selected_idx = cand_idx.gather(1, selected_pos[:, None]).squeeze(1)
        aux = {
            "candidate_idx": cand_idx,
            "routing_logits": routing_logits,
            "routing_weights": routing_weights,
            "candidate_probs": candidate_probs,
            "selected_pos": selected_pos,
            "selected_idx": selected_idx,
            "probs": probs,
        }
        return (probs, aux) if return_aux else probs

    def loss(
        self,
        h: torch.Tensor,
        y: torch.Tensor,
        *,
        p0_logits: torch.Tensor | None = None,
        reduction: Reduction = "mean",
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y_batch = torch.as_tensor(y, device=self.anchors.device, dtype=torch.long).flatten()
        probs, aux = self.predict(h, p0_logits=p0_logits, return_aux=True)
        if probs.shape[0] != y_batch.shape[0]:
            raise ValueError("h and y batch sizes do not match")
        target_prob = probs.gather(1, y_batch[:, None]).squeeze(1).clamp_min(self.eps)
        per_sample = -torch.log(target_prob)
        if reduction == "none":
            out = per_sample
        elif reduction == "mean":
            out = per_sample.mean()
        elif reduction == "sum":
            out = per_sample.sum()
        else:
            raise ValueError(f"unknown reduction: {reduction}")
        aux["target_prob"] = target_prob
        aux["loss_per_sample"] = per_sample
        return (out, aux) if return_aux else out

    @torch.no_grad()
    def observe(
        self,
        h: torch.Tensor,
        y: torch.Tensor,
        *,
        aux: dict[str, torch.Tensor] | None = None,
        p0_logits: torch.Tensor | None = None,
        mode: Literal["prequential"] = "prequential",
        update: UpdateMode = "hard",
    ) -> None:
        if mode != "prequential":
            raise ValueError("only prequential observe is supported")
        if update not in {"hard", "soft"}:
            raise ValueError("update must be 'hard' or 'soft'")

        h_batch = self._batch_h(h).detach().to(device=self.anchors.device, dtype=self.anchors.dtype)
        y_batch = torch.as_tensor(y, device=self.anchors.device, dtype=torch.long).flatten()
        if h_batch.shape[0] != y_batch.shape[0]:
            raise ValueError("h and y batch sizes do not match")

        if aux is None:
            _, aux = self.predict(h_batch, p0_logits=p0_logits, return_aux=True)
        probs = aux.get("probs")
        target_prob = aux.get("target_prob")
        selected_idx = aux.get("selected_idx")
        candidate_idx = aux.get("candidate_idx")
        routing_weights = aux.get("routing_weights")

        for row in range(h_batch.shape[0]):
            y_i = int(y_batch[row].item())
            h_i = h_batch[row]
            if self.num_concepts == 0 or selected_idx is None or int(selected_idx[row].item()) < 0:
                self.add_concept(h_i, y_i)
                continue

            pred_i = int(torch.argmax(probs[row]).item()) if probs is not None else y_i
            selected = int(selected_idx[row].item())
            selected_total = float(self.totals[selected].item())
            if target_prob is None:
                sample_loss = float("inf")
            else:
                sample_loss = float(-torch.log(target_prob[row].clamp_min(self.eps)).item())
            can_refine = (
                self.refine_on_error
                and pred_i != y_i
                and selected_total >= self.min_refine_total
                and sample_loss >= self.refine_loss_threshold
                and self._has_concept_capacity()
            )
            if can_refine:
                new_idx = self.add_concept(h_i, y_i)
                if selected != new_idx:
                    self.add_edge(selected, new_idx)
                continue

            if update == "hard" or candidate_idx is None or routing_weights is None:
                self._repair_concept(selected, h_i, y_i, weight=1.0)
            else:
                for cand, weight in zip(candidate_idx[row], routing_weights[row], strict=True):
                    self._repair_concept(int(cand.item()), h_i, y_i, weight=float(weight.item()))

    def _has_concept_capacity(self) -> bool:
        return self.max_concepts is None or self.num_concepts < self.max_concepts

    def _repair_concept(self, index: int, h: torch.Tensor, y: int, *, weight: float) -> None:
        self._add_counts(index, y, weight)
        lr = self.centroid_lr * float(weight)
        if lr > 0.0:
            self.centroids[index].mul_(1.0 - lr).add_(h, alpha=lr)

    def _add_counts(self, index: int, y: int, weight: float) -> None:
        if not (0 <= int(y) < self.n_classes):
            raise ValueError("class index out of range")
        self.counts[int(index), int(y)] += float(weight)
        self.totals[int(index)] += float(weight)

    def _candidate_indices(self, h: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(h, self.centroids.to(h.device), p=2) ** 2
        kk = min(self.k, self.num_concepts)
        return torch.topk(-distances, k=kk, dim=1, sorted=True).indices

    def _edge_log_compatibility(self, h: torch.Tensor, cand_idx: torch.Tensor) -> torch.Tensor:
        if self.num_edges == 0:
            return torch.zeros(cand_idx.shape, device=h.device, dtype=h.dtype)

        penalty = torch.zeros(cand_idx.shape, device=h.device, dtype=h.dtype)
        edge_u = self.edge_u.to(h.device)
        edge_b = self.edge_b.to(h.device)
        edge_src = self.edge_src.to(h.device)
        edge_dst = self.edge_dst.to(h.device)
        edge_value = h @ edge_u.T + edge_b
        for edge in range(self.num_edges):
            src_match = cand_idx == edge_src[edge]
            dst_match = cand_idx == edge_dst[edge]
            sign = dst_match.to(h.dtype) - src_match.to(h.dtype)
            mask = sign != 0
            signed_value = sign * edge_value[:, edge : edge + 1]
            edge_log_gate = F.logsigmoid(self.edge_temperature * signed_value)
            penalty = penalty + torch.where(mask, self.edge_weight * edge_log_gate, torch.zeros_like(penalty))
        return penalty

    def _prior_probs(self, h: torch.Tensor, p0_logits: torch.Tensor | None) -> torch.Tensor:
        if p0_logits is None:
            return torch.full(
                (h.shape[0], self.n_classes),
                1.0 / self.n_classes,
                device=h.device,
                dtype=h.dtype,
            )
        logits = torch.as_tensor(p0_logits, device=h.device, dtype=h.dtype)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.shape != (h.shape[0], self.n_classes):
            raise ValueError("p0_logits must have shape [B, n_classes]")
        return F.softmax(logits, dim=1)

    def _single_h(self, h: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(h, dtype=self.anchors.dtype, device=self.anchors.device)
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.shape != (self.dim,):
            raise ValueError(f"expected a single hidden vector with shape [{self.dim}]")
        return tensor

    def _batch_h(self, h: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(h, dtype=self.anchors.dtype, device=self.anchors.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.dim:
            raise ValueError(f"expected hidden states with shape [B, {self.dim}]")
        return tensor


DGMClassifier = CategoricalDGM
