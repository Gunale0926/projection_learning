from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from run_cifar100_image_hqs import (
    NUM_COARSE,
    NUM_FINE,
    RandomProjectionImageEncoder,
    SupervisedPrototypeEncoder,
    load_cifar100,
    make_image_hqs,
    run_prequential,
    sample_indices,
    set_seed,
)


def candidate_configs() -> list[dict[str, Any]]:
    base = {
        "encoder": "prototype",
        "dgm_variant": "categorical",
        "embedding_dim": 128,
        "pool": 4,
        "prototype_fit": 50000,
        "batch_size": 1024,
        "k": 8,
        "alpha": 1.0,
        "distance_temperature": 0.35,
        "edge_temperature": 8.0,
        "edge_weight": 1.0,
        "centroid_lr": 0.04,
        "min_refine_total": 0.0,
        "refine_loss_threshold": 0.0,
        "eta": 0.02,
        "mask_scale": 0.20,
        "logistic_lr": 0.4,
        "max_concepts": 768,
        "concept_radius": 0.32,
        "proposal_size": 32,
        "score_size": 32,
        "m_min": 4,
        "split_tau": 0.005,
        "lambda_edge": 0.0,
    }
    overrides = [
        {},
        {"dgm_variant": "image_hqs"},
        {"mask_scale": 0.20},
        {"mask_scale": 0.12},
        {"mask_scale": 0.20, "concept_radius": 0.20},
        {"mask_scale": 0.20, "concept_radius": 0.26},
        {"mask_scale": 0.20, "concept_radius": 0.40},
        {"mask_scale": 0.20, "max_concepts": 512, "concept_radius": 0.26},
        {"mask_scale": 0.20, "max_concepts": 512, "concept_radius": 0.32},
        {"mask_scale": 0.20, "max_concepts": 1024, "concept_radius": 0.26},
        {"mask_scale": 0.20, "max_concepts": 1024, "concept_radius": 0.32},
        {"mask_scale": 0.20, "alpha": 4.0, "max_concepts": 512, "concept_radius": 0.32},
        {"mask_scale": 0.20, "alpha": 8.0, "max_concepts": 512, "concept_radius": 0.32},
        {"mask_scale": 0.20, "distance_temperature": 0.20, "max_concepts": 512, "concept_radius": 0.32},
        {"mask_scale": 0.20, "distance_temperature": 0.70, "max_concepts": 512, "concept_radius": 0.32},
        {"mask_scale": 0.20, "proposal_size": 16, "score_size": 16, "m_min": 3, "max_concepts": 512},
        {"mask_scale": 0.20, "proposal_size": 64, "score_size": 64, "m_min": 6, "max_concepts": 512},
        {"mask_scale": 0.08},
        {"mask_scale": 0.05},
        {"mask_scale": 0.03},
        {"mask_scale": 0.08, "max_concepts": 512},
        {"mask_scale": 0.08, "max_concepts": 1024},
        {"mask_scale": 0.05, "max_concepts": 512},
        {"mask_scale": 0.05, "max_concepts": 1024},
        {"mask_scale": 0.12, "max_concepts": 512},
        {"mask_scale": 0.20, "max_concepts": 512},
        {"mask_scale": 0.20, "max_concepts": 1024, "alpha": 2.0, "min_refine_total": 2.0},
        {"mask_scale": 0.20, "max_concepts": 1024, "alpha": 4.0, "min_refine_total": 2.0},
        {"mask_scale": 0.20, "max_concepts": 1024, "alpha": 4.0, "min_refine_total": 5.0},
        {"mask_scale": 0.20, "max_concepts": 1024, "alpha": 8.0, "min_refine_total": 5.0},
        {"mask_scale": 0.20, "max_concepts": 1024, "alpha": 4.0, "min_refine_total": 10.0},
        {"mask_scale": 0.20, "max_concepts": 1024, "alpha": 4.0, "min_refine_total": 5.0, "refine_loss_threshold": 0.75},
        {"mask_scale": 0.20, "max_concepts": 1024, "alpha": 4.0, "min_refine_total": 5.0, "centroid_lr": 0.0},
        {"mask_scale": 0.08, "distance_temperature": 0.10, "max_concepts": 512},
        {"mask_scale": 0.08, "distance_temperature": 0.20, "max_concepts": 512},
        {"mask_scale": 0.08, "distance_temperature": 0.70, "max_concepts": 512},
        {"mask_scale": 0.08, "alpha": 0.10, "max_concepts": 512},
        {"mask_scale": 0.08, "alpha": 0.30, "max_concepts": 512},
        {"mask_scale": 0.08, "alpha": 2.00, "max_concepts": 512},
        {"mask_scale": 0.08, "k": 4, "max_concepts": 512},
        {"mask_scale": 0.08, "k": 16, "max_concepts": 512},
        {"mask_scale": 0.08, "centroid_lr": 0.0, "max_concepts": 512},
        {"mask_scale": 0.08, "centroid_lr": 0.10, "max_concepts": 512},
        {"mask_scale": 0.08, "edge_weight": 0.50, "max_concepts": 512},
        {"mask_scale": 0.08, "edge_weight": 2.00, "max_concepts": 512},
        {"mask_scale": 0.08, "edge_temperature": 16.0, "max_concepts": 512},
        {"mask_scale": 0.08, "pool": 2, "max_concepts": 512},
        {"mask_scale": 0.08, "pool": 1, "max_concepts": 512},
        {"mask_scale": 0.08, "embedding_dim": 256, "pool": 2, "max_concepts": 512},
        {"mask_scale": 0.05, "embedding_dim": 256, "pool": 2, "max_concepts": 1024},
    ]
    configs = []
    for idx, override in enumerate(overrides):
        cfg = dict(base)
        cfg.update(override)
        cfg["name"] = f"cfg_{idx:02d}"
        configs.append(cfg)
    return configs


def make_encoder(cfg: dict[str, Any], data: Any, seed: int, batch_size: int) -> Any:
    if cfg["encoder"] == "random_projection":
        return RandomProjectionImageEncoder(
            in_channels=3,
            image_size=32,
            embedding_dim=int(cfg["embedding_dim"]),
            seed=seed + 101,
            pool=int(cfg["pool"]),
        )
    if cfg["encoder"] == "prototype":
        encoder = SupervisedPrototypeEncoder(
            in_channels=3,
            image_size=32,
            embedding_dim=int(cfg["embedding_dim"]),
            seed=seed + 101,
            pool=int(cfg["pool"]),
            n_classes=NUM_FINE,
        )
        fit_idx = sample_indices(data.train_fine, int(cfg["prototype_fit"]), seed + 71)
        encoder.fit(data.train_x[fit_idx], data.train_fine[fit_idx], batch_size)
        return encoder
    raise ValueError(f"unknown encoder: {cfg['encoder']}")


def run_config(
    cfg: dict[str, Any],
    data: Any,
    train_idx: torch.Tensor,
    eval_idx: torch.Tensor,
    seed: int,
    n_train: int,
    n_eval: int,
) -> dict[str, Any]:
    set_seed(seed)
    train_x = data.train_x[train_idx]
    train_fine = data.train_fine[train_idx]
    train_coarse = data.train_coarse[train_idx]
    eval_x = data.eval_x[eval_idx]
    eval_fine = data.eval_fine[eval_idx]
    eval_coarse = data.eval_coarse[eval_idx]

    encoder = make_encoder(cfg, data, seed, int(cfg["batch_size"]))
    z_train = encoder.transform(train_x, int(cfg["batch_size"]))
    z_eval = encoder.transform(eval_x, int(cfg["batch_size"]))
    h_train, y_train, _ = make_image_hqs(
        z_train,
        train_coarse,
        seed=seed + 303,
        eta=float(cfg["eta"]),
        mask_scale=float(cfg["mask_scale"]),
    )
    h_eval, y_eval, _ = make_image_hqs(
        z_eval,
        eval_coarse,
        seed=seed + 404,
        eta=float(cfg["eta"]),
        mask_scale=float(cfg["mask_scale"]),
    )
    args = SimpleNamespace(**cfg)
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
        "name": cfg["name"],
        "n_train": n_train,
        "n_eval": n_eval,
        "config": cfg,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("experiments/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/cifar100_image_hqs_results"))
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=800)
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    data = load_cifar100(args.data_root)
    train_idx = sample_indices(data.train_fine, args.n_train, args.seed + 11)
    eval_idx = sample_indices(data.eval_fine, args.n_eval, args.seed + 29)
    configs = candidate_configs()
    if args.max_runs > 0:
        configs = configs[: args.max_runs]

    runs = []
    best: dict[str, Any] | None = None
    for cfg in configs:
        result = run_config(cfg, data, train_idx, eval_idx, args.seed, args.n_train, args.n_eval)
        runs.append(result)
        dgm = result["metrics"]["dgm"]
        logistic = result["metrics"]["online_logistic"]
        print(
            f"{result['name']}: dgm_acc={dgm['heldout_accuracy']:.3f} "
            f"dgm_nll={dgm['heldout_nll']:.3f} logistic_acc={logistic['heldout_accuracy']:.3f} "
            f"concepts={dgm['concepts']} mask_scale={cfg['mask_scale']} max={cfg['max_concepts']}"
        )
        if best is None or dgm["heldout_accuracy"] > best["metrics"]["dgm"]["heldout_accuracy"]:
            best = result
    assert best is not None
    payload = {
        "description": "Hyperparameter sweep for CIFAR-100 Image-BalancedMask-HQS. Task, labels, metrics, and baselines are unchanged.",
        "seed": args.seed,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "selection_metric": "dgm.heldout_accuracy",
        "best_name": best["name"],
        "best_config": best["config"],
        "best_metrics": best["metrics"],
        "runs": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "tuning_summary.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    best_acc = best["metrics"]["dgm"]["heldout_accuracy"]
    best_nll = best["metrics"]["dgm"]["heldout_nll"]
    print(f"best={best['name']} dgm_acc={best_acc:.3f} dgm_nll={best_nll:.3f}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
