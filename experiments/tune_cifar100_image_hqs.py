from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from run_cifar100_image_hqs import load_cifar100, parse_lambda_grid, run_one, select_device


def base_config() -> dict[str, Any]:
    return {
        "device": "auto",
        "encoder": "prototype",
        "embedding_dim": 128,
        "prototype_fit": 50000,
        "pool": 4,
        "encoder_epochs": 8,
        "encoder_batch_size": 256,
        "encoder_lr": 2e-3,
        "batch_size": 1024,
        "dgm_variant": "categorical",
        "max_concepts": 256,
        "concept_radius": 0.32,
        "k": 8,
        "alpha": 1.0,
        "distance_temperature": 0.35,
        "edge_temperature": 8.0,
        "edge_weight": 1.0,
        "centroid_lr": 0.04,
        "min_refine_total": 0.0,
        "refine_loss_threshold": 0.0,
        "proposal_size": 32,
        "score_size": 32,
        "m_min": 4,
        "split_tau": 0.005,
        "lambda_edge": 0.0,
        "posterior_lr": 0.5,
        "posterior_l2": 1e-4,
        "posterior_eta": None,
        "posterior_route_on": "image",
        "eta": 0.02,
        "mask_scale": 0.20,
        "logistic_lr": 0.4,
        "crossed_lr": 0.15,
        "logistic_l2": 1e-5,
        "probe_k": 20,
        "lambda_grid": parse_lambda_grid("0,0.02,0.05,0.1,0.2,0.4,0.7,1.0"),
    }


def candidate_configs(suite: str) -> list[dict[str, Any]]:
    if suite == "posterior_ablations":
        overrides = [
            {"label": "categorical_dgm"},
            {"label": "single_bit_image_only", "dgm_variant": "image_only_mask", "mask_scale": 0.20, "max_concepts": 256},
            {"label": "global_posterior", "dgm_variant": "global_mask_posterior", "mask_scale": 0.20, "max_concepts": 1},
            {"label": "posterior_image_route", "dgm_variant": "image_only_mask_posterior", "posterior_route_on": "image", "mask_scale": 0.20, "max_concepts": 256},
            {"label": "posterior_image_route_eta0", "dgm_variant": "image_only_mask_posterior", "posterior_route_on": "image", "posterior_eta": 0.0, "mask_scale": 0.20, "max_concepts": 256},
            {"label": "posterior_image_mask_route", "dgm_variant": "image_only_mask_posterior", "posterior_route_on": "image_mask", "mask_scale": 0.20, "max_concepts": 256},
        ]
    elif suite == "all":
        overrides = [
        {},
        {"mask_scale": 0.0},
        {"mask_scale": 0.05},
        {"mask_scale": 0.10},
        {"mask_scale": 0.50},
        {"max_concepts": 100},
        {"max_concepts": 200},
        {"max_concepts": 400},
        {"max_concepts": 1500},
        {"k": 1},
        {"k": 4},
        {"k": 16},
        {"k": 32},
        {"alpha": 0.05},
        {"alpha": 0.1},
        {"alpha": 0.3},
        {"alpha": 3.0},
        {"dgm_variant": "image_only_mask", "mask_scale": 0.20, "max_concepts": 400},
        {"dgm_variant": "image_only_mask", "mask_scale": 0.20, "max_concepts": 768, "concept_radius": 0.25},
        {"dgm_variant": "image_only_mask", "mask_scale": 0.20, "max_concepts": 768, "concept_radius": 0.40},
        {"dgm_variant": "image_only_mask_posterior", "mask_scale": 0.20, "max_concepts": 256, "concept_radius": 0.32, "posterior_lr": 0.2},
        {"dgm_variant": "image_only_mask_posterior", "mask_scale": 0.20, "max_concepts": 400, "concept_radius": 0.32, "posterior_lr": 0.5},
        {"dgm_variant": "image_only_mask_posterior", "mask_scale": 0.20, "max_concepts": 768, "concept_radius": 0.25, "posterior_lr": 0.5},
        {"dgm_variant": "image_only_mask_posterior", "mask_scale": 0.20, "max_concepts": 768, "concept_radius": 0.40, "posterior_lr": 1.0},
        {"dgm_variant": "image_only_mask_posterior", "posterior_route_on": "image_mask", "mask_scale": 0.20, "max_concepts": 256},
        {"dgm_variant": "global_mask_posterior", "mask_scale": 0.20, "max_concepts": 1},
    ]
    else:
        raise ValueError(f"unknown suite: {suite}")
    out = []
    for idx, override in enumerate(overrides):
        cfg = base_config()
        label = override.pop("label", None)
        cfg.update(override)
        cfg["name"] = label or f"cfg_{idx:02d}"
        out.append(cfg)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("experiments/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/cifar100_image_hqs_results"))
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=None, help="legacy alias for both dev and test")
    parser.add_argument("--n-dev", type=int, default=800)
    parser.add_argument("--n-test", type=int, default=800)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--suite", choices=["all", "posterior_ablations"], default="all")
    parser.add_argument("--output-name", default="tuning_summary.json")
    args = parser.parse_args()
    if args.n_eval is not None:
        args.n_dev = args.n_eval
        args.n_test = args.n_eval

    device = select_device(args.device)
    data = load_cifar100(args.data_root)
    configs = candidate_configs(args.suite)
    if args.max_runs > 0:
        configs = configs[: args.max_runs]
    runs = []
    best: dict[str, Any] | None = None
    for cfg in configs:
        cfg["seed"] = args.seed
        cfg["n_train"] = args.n_train
        cfg["n_dev"] = args.n_dev
        cfg["n_test"] = args.n_test
        cfg["n_eval"] = args.n_test
        cfg["quick"] = False
        cfg["data_root"] = args.data_root
        ns = SimpleNamespace(**cfg)
        result = run_one(args.seed, data, ns, device)
        row = {"name": cfg["name"], "config": cfg, **result}
        runs.append(row)
        dgm = result["metrics"]["dgm"]
        trust = result["metrics"]["logistic_dgm_trust"]
        logistic = result["metrics"]["online_logistic"]
        print(
            f"{cfg['name']}: dgm_dev={dgm['dev_heldout_nll']:.3f} dgm_test={dgm['test_heldout_nll']:.3f} "
            f"trust_dev={trust['dev_heldout_nll']:.3f} logistic_dev={logistic['dev_heldout_nll']:.3f}"
        )
        if best is None or trust["dev_heldout_nll"] < best["metrics"]["logistic_dgm_trust"]["dev_heldout_nll"]:
            best = row
    assert best is not None
    payload = {
        "description": "CUDA hyperparameter sweep for CIFAR-100 Image-BalancedMask-HQS. The task and labels are unchanged.",
        "suite": args.suite,
        "seed": args.seed,
        "n_train": args.n_train,
        "n_dev": args.n_dev,
        "n_test": args.n_test,
        "selection_metric": "logistic_dgm_trust.dev_heldout_nll",
        "best_name": best["name"],
        "best_config": best["config"],
        "best_metrics": best["metrics"],
        "best_diagnostics": best["diagnostics"],
        "runs": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / args.output_name
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"best={best['name']} trust_dev_nll={best['metrics']['logistic_dgm_trust']['dev_heldout_nll']:.3f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
