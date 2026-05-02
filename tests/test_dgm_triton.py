from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.dgm_reference import CategoricalDGM
from experiments.dgm_triton import categorical_dgm_nll, categorical_dgm_nll_torch, triton_available


def test_torch_fused_nll_matches_reference_loss_without_edges() -> None:
    torch.manual_seed(7)
    batch = 6
    dim = 5
    n_concepts = 9
    n_classes = 4
    k = 3
    alpha = 0.7
    distance_temperature = 0.9

    h = torch.randn(batch, dim)
    centroids = torch.randn(n_concepts, dim)
    counts = torch.rand(n_concepts, n_classes) * 5.0
    totals = counts.sum(dim=1)
    y = torch.randint(0, n_classes, (batch,))

    dgm = CategoricalDGM(
        dim=dim,
        n_classes=n_classes,
        k=k,
        alpha=alpha,
        distance_temperature=distance_temperature,
        refine_on_error=False,
    )
    dgm.anchors = centroids.clone()
    dgm.centroids = centroids.clone()
    dgm.counts = counts.clone()
    dgm.totals = totals.clone()
    loss_ref, aux = dgm.loss(h, y, reduction="none", return_aux=True)
    cand_idx = aux["candidate_idx"]

    loss_fused, p_y = categorical_dgm_nll_torch(
        h,
        cand_idx,
        centroids,
        counts,
        totals,
        y,
        alpha=alpha,
        distance_temperature=distance_temperature,
    )

    torch.testing.assert_close(loss_fused, loss_ref)
    torch.testing.assert_close(p_y, aux["target_prob"])


def test_auto_mode_uses_fallback_when_triton_is_unavailable() -> None:
    h = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    cand_idx = torch.tensor([[0, 1], [1, 0]])
    centroids = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    counts = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    totals = counts.sum(dim=1)
    y = torch.tensor([0, 1])

    loss_auto, p_auto = categorical_dgm_nll(
        h,
        cand_idx,
        centroids,
        counts,
        totals,
        y,
        alpha=1.0,
        distance_temperature=1.0,
        mode="auto",
    )
    loss_fallback, p_fallback = categorical_dgm_nll(
        h,
        cand_idx,
        centroids,
        counts,
        totals,
        y,
        alpha=1.0,
        distance_temperature=1.0,
        mode="never",
    )

    torch.testing.assert_close(loss_auto, loss_fallback)
    torch.testing.assert_close(p_auto, p_fallback)

    if not triton_available():
        try:
            categorical_dgm_nll(
                h,
                cand_idx,
                centroids,
                counts,
                totals,
                y,
                mode="always",
            )
        except RuntimeError:
            pass
        else:  # pragma: no cover
            raise AssertionError("mode='always' should fail when Triton is unavailable")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("dgm_triton tests passed")

