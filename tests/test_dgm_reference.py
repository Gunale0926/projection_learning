from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.dgm_reference import CategoricalDGM


def test_two_state_repair_only_lower_bound() -> None:
    dgm = CategoricalDGM(
        dim=1,
        n_classes=2,
        k=1,
        alpha=1.0,
        centroid_lr=0.0,
        refine_on_error=False,
    )
    dgm.add_concept(torch.tensor([0.0]))

    losses: list[float] = []
    for _ in range(200):
        for h, y in [(torch.tensor([[-1.0]]), torch.tensor([0])), (torch.tensor([[1.0]]), torch.tensor([1]))]:
            loss, aux = dgm.loss(h, y, return_aux=True)
            losses.append(float(loss.item()))
            dgm.observe(h, y, aux=aux)

    tail = sum(losses[-100:]) / 100.0
    assert abs(tail - math.log(2.0)) < 0.02
    assert dgm.num_concepts == 1
    assert dgm.num_edges == 0


def test_edge_separates_endpoint_routes() -> None:
    dgm = CategoricalDGM(dim=2, n_classes=2, k=2, alpha=0.1, edge_temperature=20.0)
    left = torch.tensor([-1.0, 0.0])
    right = torch.tensor([1.0, 0.0])
    dgm.add_concept(left, 0)
    dgm.add_concept(right, 1)
    dgm.add_edge(0, 1)

    _, aux = dgm.predict(torch.stack([left, right]), return_aux=True)

    assert aux["selected_idx"].tolist() == [0, 1]
    assert torch.all(aux["routing_weights"].max(dim=1).values > 0.99)


def test_routing_probability_normalization() -> None:
    dgm = CategoricalDGM(dim=2, n_classes=3, k=3, alpha=1.0)
    dgm.add_concept(torch.tensor([-1.0, 0.0]), 0)
    dgm.add_concept(torch.tensor([0.0, 1.0]), 1)
    dgm.add_concept(torch.tensor([1.0, 0.0]), 2)

    probs, aux = dgm.predict(torch.tensor([[-0.5, 0.2], [0.7, 0.1]]), return_aux=True)

    torch.testing.assert_close(aux["routing_weights"].sum(dim=1), torch.ones(2))
    torch.testing.assert_close(probs.sum(dim=1), torch.ones(2))
    assert torch.all(probs > 0.0)


def test_categorical_memory_nll() -> None:
    dgm = CategoricalDGM(dim=2, n_classes=2, k=1, alpha=2.0, refine_on_error=False)
    dgm.add_concept(torch.tensor([0.0, 0.0]))
    dgm.counts[0] = torch.tensor([3.0, 1.0])
    dgm.totals[0] = 4.0

    loss, aux = dgm.loss(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([1]),
        reduction="none",
        return_aux=True,
    )

    torch.testing.assert_close(aux["target_prob"], torch.tensor([1.0 / 3.0]))
    torch.testing.assert_close(loss, torch.tensor([math.log(3.0)]))


def test_prequential_loss_does_not_leak_label() -> None:
    dgm = CategoricalDGM(dim=1, n_classes=2, k=1, alpha=1.0, refine_on_error=False)
    dgm.add_concept(torch.tensor([0.0]))
    dgm.counts[0] = torch.tensor([10.0, 0.0])
    dgm.totals[0] = 10.0
    before_counts = dgm.counts.clone()
    before_totals = dgm.totals.clone()

    h = torch.tensor([[0.0]])
    y = torch.tensor([1])
    loss_before, aux = dgm.loss(h, y, return_aux=True)

    torch.testing.assert_close(dgm.counts, before_counts)
    torch.testing.assert_close(dgm.totals, before_totals)

    dgm.observe(h, y, aux=aux)
    loss_after = dgm.loss(h, y)

    assert float(loss_after.item()) < float(loss_before.item())
    assert dgm.counts[0, 1].item() == 1.0


def test_repair_only_does_not_change_anchors_or_edges() -> None:
    dgm = CategoricalDGM(dim=2, n_classes=2, k=2, alpha=0.1, refine_on_error=False)
    dgm.add_concept(torch.tensor([-1.0, 0.0]), 0)
    dgm.add_concept(torch.tensor([1.0, 0.0]), 1)
    dgm.add_edge(0, 1)
    anchors = dgm.anchors.clone()
    edge_u = dgm.edge_u.clone()
    edge_b = dgm.edge_b.clone()
    edge_src = dgm.edge_src.clone()
    edge_dst = dgm.edge_dst.clone()

    h = torch.tensor([[-0.9, 0.1]])
    y = torch.tensor([1])
    _, aux = dgm.loss(h, y, return_aux=True)
    dgm.observe(h, y, aux=aux)

    torch.testing.assert_close(dgm.anchors, anchors)
    torch.testing.assert_close(dgm.edge_u, edge_u)
    torch.testing.assert_close(dgm.edge_b, edge_b)
    torch.testing.assert_close(dgm.edge_src, edge_src)
    torch.testing.assert_close(dgm.edge_dst, edge_dst)
    assert dgm.num_concepts == 2
    assert dgm.num_edges == 1


def test_refine_creates_expected_edge() -> None:
    dgm = CategoricalDGM(dim=2, n_classes=2, k=1, alpha=0.1, centroid_lr=0.0, refine_on_error=True)
    left = torch.tensor([-1.0, 0.0])
    right = torch.tensor([1.0, 0.0])
    dgm.add_concept(left, 0)

    h = right.unsqueeze(0)
    y = torch.tensor([1])
    _, aux = dgm.loss(h, y, return_aux=True)
    dgm.observe(h, y, aux=aux)

    assert dgm.num_concepts == 2
    assert dgm.num_edges == 1
    torch.testing.assert_close(dgm.anchors[1], right)
    assert int(dgm.edge_src[0].item()) == 0
    assert int(dgm.edge_dst[0].item()) == 1
    assert dgm.counts[1, 1].item() == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("dgm_reference tests passed")
