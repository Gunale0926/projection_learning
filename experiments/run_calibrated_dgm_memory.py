from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EPS = 1e-12


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.maximum(p, EPS)
    return p / p.sum()


def one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=float)
    out[int(index)] = 1.0
    return out


@dataclass(frozen=True)
class HiddenRegimeConfig:
    vocab_size: int = 64
    n_contexts: int = 16
    topic_size: int = 6
    t_steps: int = 8000
    drift_fraction: float = 0.55
    background_mass: float = 0.12
    stay_probability: float = 0.94


class HiddenRegimeStream:
    """A controlled next-token stream with repeated contexts and phase drift.

    The base predictor observes only the phase-level unigram law. The memory
    routes by the current context id and can learn context-specific token laws,
    but cumulative counts become stale after the phase drift. This makes
    untrusted memory useful but unsafe, while the lambda=0 Hedge expert gives
    a precise cumulative log-loss fallback to the base predictor.
    """

    def __init__(self, seed: int, config: HiddenRegimeConfig) -> None:
        self.rng = np.random.default_rng(seed)
        self.config = config
        self.phase_cut = int(config.t_steps * config.drift_fraction)
        self.context = int(self.rng.integers(config.n_contexts))
        self.topic_a = self._make_topic_table()
        self.topic_b = self._make_topic_table()
        self.dist_a = self._make_context_distributions(self.topic_a)
        self.dist_b = self._make_context_distributions(self.topic_b)
        self.base_a = normalize(self.dist_a.mean(axis=0))
        self.base_b = normalize(self.dist_b.mean(axis=0))

    def _make_topic_table(self) -> np.ndarray:
        rows = []
        for _ in range(self.config.n_contexts):
            rows.append(
                self.rng.choice(
                    self.config.vocab_size,
                    size=self.config.topic_size,
                    replace=False,
                )
            )
        return np.asarray(rows, dtype=np.int64)

    def _make_context_distributions(self, topics: np.ndarray) -> np.ndarray:
        rows = []
        background = np.full(self.config.vocab_size, 1.0 / self.config.vocab_size)
        for topic in topics:
            topic_weights = self.rng.dirichlet(np.full(len(topic), 0.7))
            p = self.config.background_mass * background
            p = p.copy()
            p[topic] += (1.0 - self.config.background_mass) * topic_weights
            rows.append(normalize(p))
        return np.stack(rows, axis=0)

    def _phase(self, t: int) -> int:
        return 0 if t < self.phase_cut else 1

    def base_distribution(self, t: int) -> np.ndarray:
        return self.base_a if self._phase(t) == 0 else self.base_b

    def true_distribution(self, context: int, t: int) -> np.ndarray:
        return self.dist_a[context] if self._phase(t) == 0 else self.dist_b[context]

    def next(self, t: int) -> tuple[int, int, np.ndarray, np.ndarray]:
        if self.rng.random() > self.config.stay_probability:
            self.context = int(self.rng.integers(self.config.n_contexts))
        p_true = self.true_distribution(self.context, t)
        y = int(self.rng.choice(self.config.vocab_size, p=p_true))
        return self.context, y, self.base_distribution(t), p_true


class ContextMemory:
    def __init__(self, n_contexts: int, vocab_size: int, smoothing_alpha: float) -> None:
        self.counts = np.zeros((n_contexts, vocab_size), dtype=float)
        self.totals = np.zeros(n_contexts, dtype=float)
        self.smoothing_alpha = float(smoothing_alpha)

    def predict(self, context: int, p0: np.ndarray) -> np.ndarray:
        total = self.totals[context]
        return normalize((self.counts[context] + self.smoothing_alpha * p0) / (total + self.smoothing_alpha))

    def observe(self, context: int, y: int, weight: float = 1.0) -> None:
        self.counts[context, int(y)] += float(weight)
        self.totals[context] += float(weight)


class EmpiricalMemory(ContextMemory):
    def predict(self, context: int, p0: np.ndarray) -> np.ndarray:
        total = self.totals[context]
        if total <= 0.0:
            return p0.copy()
        return normalize(self.counts[context] + EPS)


class OnlineCalibration:
    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = int(n_bins)
        self.conf_sum = np.zeros(self.n_bins, dtype=float)
        self.acc_sum = np.zeros(self.n_bins, dtype=float)
        self.counts = np.zeros(self.n_bins, dtype=float)

    def observe(self, p: np.ndarray, y: int) -> float:
        confidence = float(np.max(p))
        prediction = int(np.argmax(p))
        correct = float(prediction == int(y))
        bin_id = min(self.n_bins - 1, int(confidence * self.n_bins))
        self.conf_sum[bin_id] += confidence
        self.acc_sum[bin_id] += correct
        self.counts[bin_id] += 1.0
        total = float(self.counts.sum())
        ece = 0.0
        for i in range(self.n_bins):
            if self.counts[i] == 0:
                continue
            avg_conf = self.conf_sum[i] / self.counts[i]
            avg_acc = self.acc_sum[i] / self.counts[i]
            ece += (self.counts[i] / total) * abs(avg_acc - avg_conf)
        return float(ece)


@dataclass
class ModelTotals:
    cumulative_nll: float = 0.0
    cumulative_brier: float = 0.0
    correct: int = 0

    def observe(self, p: np.ndarray, y: int) -> tuple[float, float, int]:
        prob = float(max(p[int(y)], EPS))
        nll = -math.log(prob)
        brier = float(np.sum((p - one_hot(y, len(p))) ** 2))
        correct = int(np.argmax(p) == int(y))
        self.cumulative_nll += nll
        self.cumulative_brier += brier
        self.correct += correct
        return nll, brier, correct


def parse_lambda_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("lambda grid must contain at least one value")
    if min(values) < 0.0 or max(values) > 1.0:
        raise ValueError("lambda values must lie in [0, 1]")
    if 0.0 not in values:
        values = [0.0, *values]
    return sorted(set(values))


def run_trial(
    seed: int,
    config: HiddenRegimeConfig,
    lambda_grid: list[float],
    smoothing_alpha: float,
    csv_path: Path,
) -> dict[str, object]:
    stream = HiddenRegimeStream(seed=seed, config=config)
    hedge_memory = ContextMemory(config.n_contexts, config.vocab_size, smoothing_alpha)
    fixed_memories = {
        "overconfident_memory": EmpiricalMemory(config.n_contexts, config.vocab_size, 0.0),
        "memory_only": ContextMemory(config.n_contexts, config.vocab_size, smoothing_alpha),
        "fixed_half": ContextMemory(config.n_contexts, config.vocab_size, smoothing_alpha),
    }
    base_totals = ModelTotals()
    hedge_totals = ModelTotals()
    fixed_totals = {name: ModelTotals() for name in fixed_memories}
    calibration = OnlineCalibration()
    weights = np.full(len(lambda_grid), 1.0 / len(lambda_grid), dtype=float)
    max_no_harm_violation = -math.inf
    rows: list[dict[str, object]] = []

    for t in range(config.t_steps):
        context, y, p0, _ = stream.next(t)
        p_mem = hedge_memory.predict(context, p0)
        expert_predictions = np.stack(
            [(1.0 - lam) * p0 + lam * p_mem for lam in lambda_grid],
            axis=0,
        )
        p_hedge = normalize(weights @ expert_predictions)

        nll0, brier0, correct0 = base_totals.observe(p0, y)
        nll, brier, correct = hedge_totals.observe(p_hedge, y)
        ece = calibration.observe(p_hedge, y)

        fixed_predictions = {
            "overconfident_memory": fixed_memories["overconfident_memory"].predict(context, p0),
            "memory_only": fixed_memories["memory_only"].predict(context, p0),
            "fixed_half": normalize(0.5 * p0 + 0.5 * fixed_memories["fixed_half"].predict(context, p0)),
        }
        fixed_step_nll = {}
        for name, p_fixed in fixed_predictions.items():
            fixed_step_nll[name], _, _ = fixed_totals[name].observe(p_fixed, y)
            fixed_memories[name].observe(context, y)

        y_probs = np.maximum(expert_predictions[:, y], EPS)
        normalizer = float(np.dot(weights, y_probs))
        weights = weights * y_probs / max(normalizer, EPS)
        weights = weights / weights.sum()
        hedge_memory.observe(context, y)

        no_harm_slack = hedge_totals.cumulative_nll - base_totals.cumulative_nll - math.log(len(lambda_grid))
        max_no_harm_violation = max(max_no_harm_violation, no_harm_slack)
        row = {
            "t": t + 1,
            "context": context,
            "y": y,
            "nll": nll,
            "nll0": nll0,
            "overconfident_memory_nll": fixed_step_nll["overconfident_memory"],
            "memory_only_nll": fixed_step_nll["memory_only"],
            "fixed_half_nll": fixed_step_nll["fixed_half"],
            "cumulative_nll": hedge_totals.cumulative_nll,
            "cumulative_nll0": base_totals.cumulative_nll,
            "no_harm_slack": no_harm_slack,
            "accuracy": hedge_totals.correct / (t + 1),
            "base_accuracy": base_totals.correct / (t + 1),
            "ece": ece,
            "brier": brier,
            "base_brier": brier0,
        }
        for idx, lam in enumerate(lambda_grid):
            row[f"w_lambda_{lam:g}"] = float(weights[idx])
        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    t_steps = float(config.t_steps)
    return {
        "seed": seed,
        "t_steps": config.t_steps,
        "vocab_size": config.vocab_size,
        "n_contexts": config.n_contexts,
        "lambda_grid": lambda_grid,
        "log_grid_size": math.log(len(lambda_grid)),
        "hedge_cumulative_nll": hedge_totals.cumulative_nll,
        "base_cumulative_nll": base_totals.cumulative_nll,
        "overconfident_memory_cumulative_nll": fixed_totals["overconfident_memory"].cumulative_nll,
        "memory_only_cumulative_nll": fixed_totals["memory_only"].cumulative_nll,
        "fixed_half_cumulative_nll": fixed_totals["fixed_half"].cumulative_nll,
        "hedge_excess_over_base": hedge_totals.cumulative_nll - base_totals.cumulative_nll,
        "overconfident_memory_excess_over_base": fixed_totals["overconfident_memory"].cumulative_nll - base_totals.cumulative_nll,
        "memory_only_excess_over_base": fixed_totals["memory_only"].cumulative_nll - base_totals.cumulative_nll,
        "fixed_half_excess_over_base": fixed_totals["fixed_half"].cumulative_nll - base_totals.cumulative_nll,
        "max_no_harm_violation": max_no_harm_violation,
        "hedge_average_nll": hedge_totals.cumulative_nll / t_steps,
        "base_average_nll": base_totals.cumulative_nll / t_steps,
        "overconfident_memory_average_nll": fixed_totals["overconfident_memory"].cumulative_nll / t_steps,
        "memory_only_average_nll": fixed_totals["memory_only"].cumulative_nll / t_steps,
        "fixed_half_average_nll": fixed_totals["fixed_half"].cumulative_nll / t_steps,
        "hedge_accuracy": hedge_totals.correct / t_steps,
        "base_accuracy": base_totals.correct / t_steps,
        "overconfident_memory_accuracy": fixed_totals["overconfident_memory"].correct / t_steps,
        "memory_only_accuracy": fixed_totals["memory_only"].correct / t_steps,
        "fixed_half_accuracy": fixed_totals["fixed_half"].correct / t_steps,
        "hedge_average_brier": hedge_totals.cumulative_brier / t_steps,
        "base_average_brier": base_totals.cumulative_brier / t_steps,
        "final_weights": {f"{lam:g}": float(weight) for lam, weight in zip(lambda_grid, weights, strict=True)},
        "calibration_csv": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-steps", type=int, default=8000)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--n-contexts", type=int, default=16)
    parser.add_argument("--topic-size", type=int, default=6)
    parser.add_argument("--smoothing-alpha", type=float, default=3.0)
    parser.add_argument("--lambda-grid", type=str, default="0,0.05,0.1,0.2,0.4,0.7,1.0")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/calibrated_dgm_results"))
    args = parser.parse_args()

    config = HiddenRegimeConfig(
        vocab_size=args.vocab_size,
        n_contexts=args.n_contexts,
        topic_size=args.topic_size,
        t_steps=args.t_steps,
    )
    lambda_grid = parse_lambda_grid(args.lambda_grid)
    csv_path = args.output_dir / "calibration.csv"
    summary = run_trial(
        seed=args.seed,
        config=config,
        lambda_grid=lambda_grid,
        smoothing_alpha=args.smoothing_alpha,
        csv_path=csv_path,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(
        "NLL: "
        f"hedge={summary['hedge_average_nll']:.4f}, "
        f"base={summary['base_average_nll']:.4f}, "
        f"overconfident={summary['overconfident_memory_average_nll']:.4f}, "
        f"memory_only={summary['memory_only_average_nll']:.4f}, "
        f"fixed_half={summary['fixed_half_average_nll']:.4f}"
    )
    print(
        "cumulative no-harm check: "
        f"hedge-base={summary['hedge_excess_over_base']:.4f}, "
        f"logK={summary['log_grid_size']:.4f}, "
        f"max_violation={summary['max_no_harm_violation']:.6g}"
    )


if __name__ == "__main__":
    main()
