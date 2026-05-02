from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

try:  # pragma: no cover - depends on optional runtime package
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised in the default ai env
    triton = None
    tl = None


TritonMode = Literal["auto", "always", "never"]


def triton_available() -> bool:
    return triton is not None


def categorical_dgm_nll(
    h: torch.Tensor,
    cand_idx: torch.Tensor,
    centroids: torch.Tensor,
    counts: torch.Tensor,
    totals: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float = 1.0,
    distance_temperature: float = 1.0,
    mode: TritonMode = "auto",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused categorical DGM target NLL over a fixed candidate set.

    This hot path intentionally covers the first optimization target only:
    distance-based soft routing over `cand_idx` and dense categorical memory
    NLL for the observed label. Dynamic graph mutation and edge maintenance
    stay in Python/PyTorch.
    """

    if mode not in {"auto", "always", "never"}:
        raise ValueError("mode must be 'auto', 'always', or 'never'")
    if mode == "always" and not _can_use_triton(h, cand_idx, centroids, counts, totals, y):
        raise RuntimeError("Triton path requested but Triton/CUDA requirements are not satisfied")
    if mode != "never" and _can_use_triton(h, cand_idx, centroids, counts, totals, y):
        return _categorical_dgm_nll_triton(
            h,
            cand_idx,
            centroids,
            counts,
            totals,
            y,
            alpha=alpha,
            distance_temperature=distance_temperature,
            eps=eps,
        )
    return categorical_dgm_nll_torch(
        h,
        cand_idx,
        centroids,
        counts,
        totals,
        y,
        alpha=alpha,
        distance_temperature=distance_temperature,
        eps=eps,
    )


def categorical_dgm_nll_torch(
    h: torch.Tensor,
    cand_idx: torch.Tensor,
    centroids: torch.Tensor,
    counts: torch.Tensor,
    totals: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float = 1.0,
    distance_temperature: float = 1.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if h.ndim != 2:
        raise ValueError("h must have shape [B, D]")
    if cand_idx.ndim != 2 or cand_idx.shape[0] != h.shape[0]:
        raise ValueError("cand_idx must have shape [B, K]")
    if centroids.ndim != 2 or centroids.shape[1] != h.shape[1]:
        raise ValueError("centroids must have shape [N, D]")
    if counts.ndim != 2 or counts.shape[0] != centroids.shape[0]:
        raise ValueError("counts must have shape [N, C]")
    if totals.shape != (centroids.shape[0],):
        raise ValueError("totals must have shape [N]")
    y = y.to(device=h.device, dtype=torch.long).flatten()
    if y.shape[0] != h.shape[0]:
        raise ValueError("y must have shape [B]")
    if distance_temperature <= 0.0:
        raise ValueError("distance_temperature must be positive")

    cand_idx = cand_idx.to(device=h.device, dtype=torch.long)
    cand_centroids = centroids.to(h.device)[cand_idx]
    sq_dist = ((h[:, None, :] - cand_centroids) ** 2).sum(dim=-1)
    weights = F.softmax(-sq_dist / float(distance_temperature), dim=1)
    target_counts = counts.to(h.device)[cand_idx, y[:, None]]
    target_totals = totals.to(h.device)[cand_idx]
    n_classes = counts.shape[1]
    target_probs = (target_counts + float(alpha) / float(n_classes)) / (target_totals + float(alpha)).clamp_min(eps)
    mixture_target_prob = (weights * target_probs).sum(dim=1).clamp_min(eps)
    return -torch.log(mixture_target_prob), mixture_target_prob


def _can_use_triton(
    h: torch.Tensor,
    cand_idx: torch.Tensor,
    centroids: torch.Tensor,
    counts: torch.Tensor,
    totals: torch.Tensor,
    y: torch.Tensor,
) -> bool:
    return (
        triton is not None
        and h.is_cuda
        and cand_idx.is_cuda
        and centroids.is_cuda
        and counts.is_cuda
        and totals.is_cuda
        and y.is_cuda
        and h.dtype in {torch.float32, torch.float16}
        and centroids.dtype == h.dtype
        and counts.dtype == h.dtype
        and totals.dtype == h.dtype
    )


def _categorical_dgm_nll_triton(
    h: torch.Tensor,
    cand_idx: torch.Tensor,
    centroids: torch.Tensor,
    counts: torch.Tensor,
    totals: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float,
    distance_temperature: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or tl is None:  # pragma: no cover
        raise RuntimeError("Triton is not installed")
    h = h.contiguous()
    cand_idx = cand_idx.contiguous()
    centroids = centroids.contiguous()
    counts = counts.contiguous()
    totals = totals.contiguous()
    y = y.contiguous()
    batch, dim = h.shape
    k = cand_idx.shape[1]
    n_classes = counts.shape[1]
    loss = torch.empty(batch, device=h.device, dtype=torch.float32)
    target_prob = torch.empty(batch, device=h.device, dtype=torch.float32)
    _categorical_dgm_nll_kernel[(batch,)](
        h,
        cand_idx,
        centroids,
        counts,
        totals,
        y,
        loss,
        target_prob,
        batch,
        dim,
        k,
        n_classes,
        h.stride(0),
        h.stride(1),
        cand_idx.stride(0),
        cand_idx.stride(1),
        centroids.stride(0),
        centroids.stride(1),
        counts.stride(0),
        counts.stride(1),
        alpha,
        distance_temperature,
        eps,
        BLOCK_D=triton.next_power_of_2(dim),
        BLOCK_K=triton.next_power_of_2(k),
    )
    return loss.to(h.dtype), target_prob.to(h.dtype)


if triton is not None and tl is not None:  # pragma: no cover - requires optional Triton

    @triton.jit
    def _categorical_dgm_nll_kernel(
        h,
        cand_idx,
        centroids,
        counts,
        totals,
        y,
        loss,
        target_prob,
        batch: tl.constexpr,
        dim: tl.constexpr,
        k: tl.constexpr,
        n_classes: tl.constexpr,
        h_stride_b: tl.constexpr,
        h_stride_d: tl.constexpr,
        cand_stride_b: tl.constexpr,
        cand_stride_k: tl.constexpr,
        cent_stride_n: tl.constexpr,
        cent_stride_d: tl.constexpr,
        count_stride_n: tl.constexpr,
        count_stride_c: tl.constexpr,
        alpha: tl.constexpr,
        distance_temperature: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs_k = tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)
        valid_k = offs_k < k
        valid_d = offs_d < dim
        h_vals = tl.load(h + row * h_stride_b + offs_d * h_stride_d, mask=valid_d, other=0.0)
        cand = tl.load(cand_idx + row * cand_stride_b + offs_k * cand_stride_k, mask=valid_k, other=0)

        dist = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for d_block in range(0, BLOCK_D):
            active_d = d_block == offs_d
            h_d = tl.sum(tl.where(active_d & valid_d, h_vals, 0.0), axis=0)
            c_d = tl.load(
                centroids + cand * cent_stride_n + d_block * cent_stride_d,
                mask=valid_k & (d_block < dim),
                other=0.0,
            )
            diff = h_d - c_d
            dist += tl.where(valid_k & (d_block < dim), diff * diff, 0.0)

        logits = tl.where(valid_k, -dist / distance_temperature, -float("inf"))
        logits = logits - tl.max(logits, axis=0)
        exp_logits = tl.exp(logits)
        weights = exp_logits / tl.sum(exp_logits, axis=0)

        target = tl.load(y + row)
        target_counts = tl.load(
            counts + cand * count_stride_n + target * count_stride_c,
            mask=valid_k,
            other=0.0,
        )
        target_totals = tl.load(totals + cand, mask=valid_k, other=0.0)
        probs = (target_counts + alpha / n_classes) / tl.maximum(target_totals + alpha, eps)
        p_y = tl.sum(weights * probs, axis=0)
        p_y = tl.maximum(p_y, eps)
        tl.store(target_prob + row, p_y)
        tl.store(loss + row, -tl.log(p_y))

