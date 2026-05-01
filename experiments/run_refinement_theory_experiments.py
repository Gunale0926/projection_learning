from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


EPS = 1e-12


def binary_entropy(p: float) -> float:
    p = min(max(float(p), EPS), 1.0 - EPS)
    return float(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))


def bernoulli_log_loss(p: float, y: int) -> float:
    p = min(max(float(p), EPS), 1.0 - EPS)
    return float(-math.log(p if int(y) == 1 else 1.0 - p))


def mean_and_std(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def summarize_rows(rows: list[dict[str, object]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        arr = np.asarray([float(row[key]) for row in rows], dtype=float)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_std"] = float(arr.std(ddof=0))
    return out


class BernoulliCounts:
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.counts: dict[object, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.int64))

    def predict(self, key: object) -> float:
        counts = self.counts[key]
        return float((counts[1] + self.alpha) / (counts.sum() + 2.0 * self.alpha))

    def observe(self, key: object, y: int) -> None:
        self.counts[key][int(y)] += 1


@dataclass
class SplitCell:
    counts: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.int64))
    samples: list[tuple[int, int, int]] = field(default_factory=list)
    split_bit: str | None = None
    child_counts: dict[int, np.ndarray] = field(
        default_factory=lambda: {0: np.zeros(2, dtype=np.int64), 1: np.zeros(2, dtype=np.int64)}
    )
    tested: bool = False


class OneBitDGMCounts:
    """Counts learner that may refine each Z atom by one candidate bit.

    The split test uses the first half of the cell buffer as a proposal/fitting
    buffer and the second half as a scoring buffer. Each atom is tested once.
    """

    def __init__(
        self,
        candidate_bit: str,
        alpha: float = 0.5,
        min_samples: int = 4096,
        gain_threshold: float = 0.012,
    ) -> None:
        self.candidate_bit = candidate_bit
        self.alpha = alpha
        self.min_samples = min_samples
        self.gain_threshold = gain_threshold
        self.cells: dict[int, SplitCell] = defaultdict(SplitCell)
        self.accepted: list[dict[str, float]] = []
        self.rejected: list[dict[str, float]] = []

    def predict(self, z: int, u: int, v: int) -> float:
        cell = self.cells[int(z)]
        if cell.split_bit is None:
            counts = cell.counts
        else:
            bit = int(u if cell.split_bit == "u" else v)
            counts = cell.child_counts[bit]
        return float((counts[1] + self.alpha) / (counts.sum() + 2.0 * self.alpha))

    def observe(self, z: int, u: int, v: int, y: int) -> None:
        cell = self.cells[int(z)]
        bit = int(u if self.candidate_bit == "u" else v)
        if cell.split_bit is None:
            cell.counts[int(y)] += 1
        else:
            routed = int(u if cell.split_bit == "u" else v)
            cell.child_counts[routed][int(y)] += 1
        cell.samples.append((int(u), int(v), int(y)))
        if cell.split_bit is None and not cell.tested and len(cell.samples) >= self.min_samples:
            self._maybe_split(int(z), cell)

    def _maybe_split(self, z: int, cell: SplitCell) -> None:
        cell.tested = True
        raw_gain = validation_split_gain(
            samples=[{self.candidate_bit: sample[0] if self.candidate_bit == "u" else sample[1], "y": sample[2]} for sample in cell.samples],
            candidate=self.candidate_bit,
            alpha=self.alpha,
        )
        record = {"z": float(z), "gain": float(raw_gain), "threshold": float(self.gain_threshold)}
        if raw_gain > self.gain_threshold:
            cell.split_bit = self.candidate_bit
            cell.child_counts = {0: np.zeros(2, dtype=np.int64), 1: np.zeros(2, dtype=np.int64)}
            for u, v, y in cell.samples:
                bit = int(u if self.candidate_bit == "u" else v)
                cell.child_counts[bit][int(y)] += 1
            self.accepted.append(record)
        else:
            self.rejected.append(record)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)


def validation_split_gain(samples: list[dict[str, int]], candidate: str, alpha: float = 0.5) -> float:
    if len(samples) < 4:
        return -math.inf
    mid = len(samples) // 2
    proposal = samples[:mid]
    scoring = samples[mid:]
    parent_counts = np.zeros(2, dtype=np.int64)
    child_counts = {0: np.zeros(2, dtype=np.int64), 1: np.zeros(2, dtype=np.int64)}
    for sample in proposal:
        y = int(sample["y"])
        bit = int(sample[candidate])
        parent_counts[y] += 1
        child_counts[bit][y] += 1
    parent_p = float((parent_counts[1] + alpha) / (parent_counts.sum() + 2.0 * alpha))
    child_p = {
        bit: float((child_counts[bit][1] + alpha) / (child_counts[bit].sum() + 2.0 * alpha))
        for bit in (0, 1)
    }
    parent_loss = 0.0
    child_loss = 0.0
    for sample in scoring:
        y = int(sample["y"])
        bit = int(sample[candidate])
        parent_loss += bernoulli_log_loss(parent_p, y)
        child_loss += bernoulli_log_loss(child_p[bit], y)
    return float((parent_loss - child_loss) / len(scoring))


def estimate_cumulative_slope(y: np.ndarray, tail_fraction: float = 0.5) -> float:
    start = int(len(y) * (1.0 - tail_fraction))
    x = np.arange(start + 1, len(y) + 1, dtype=float)
    slope, _ = np.polyfit(x, y[start:], deg=1)
    return float(slope)


def run_one_bayes_gap(seed: int, delta: float, t_steps: int, k_atoms: int, alpha: float) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    repair = BernoulliCounts(alpha=alpha)
    refined = BernoulliCounts(alpha=alpha)
    dgm_true = OneBitDGMCounts("u", alpha=alpha)
    dgm_noise = OneBitDGMCounts("v", alpha=alpha)

    cum_regret_counts = np.zeros(t_steps, dtype=float)
    cum_regret_bayes = np.zeros(t_steps, dtype=float)
    cum_repair_loss = 0.0
    cum_refined_loss = 0.0
    cum_bayes_repair_loss = 0.0
    cum_bayes_refined_loss = 0.0
    dgm_true_loss = 0.0
    dgm_noise_loss = 0.0

    for t in range(t_steps):
        z = int(rng.integers(0, k_atoms))
        u = int(rng.integers(0, 2))
        v = int(rng.integers(0, 2))
        p_true = 0.5 + delta if u == 1 else 0.5 - delta
        y = int(rng.random() < p_true)

        p_repair = repair.predict(z)
        p_refined = refined.predict((z, u))
        p_dgm_true = dgm_true.predict(z, u, v)
        p_dgm_noise = dgm_noise.predict(z, u, v)

        cum_repair_loss += bernoulli_log_loss(p_repair, y)
        cum_refined_loss += bernoulli_log_loss(p_refined, y)
        dgm_true_loss += bernoulli_log_loss(p_dgm_true, y)
        dgm_noise_loss += bernoulli_log_loss(p_dgm_noise, y)
        cum_bayes_repair_loss += bernoulli_log_loss(0.5, y)
        cum_bayes_refined_loss += bernoulli_log_loss(p_true, y)

        cum_regret_counts[t] = cum_repair_loss - cum_refined_loss
        cum_regret_bayes[t] = cum_bayes_repair_loss - cum_bayes_refined_loss

        repair.observe(z, y)
        refined.observe((z, u), y)
        dgm_true.observe(z, u, v, y)
        dgm_noise.observe(z, u, v, y)

    return {
        "seed": seed,
        "delta": delta,
        "theory_gap": math.log(2.0) - binary_entropy(0.5 + delta),
        "repair_vs_refined_slope": estimate_cumulative_slope(cum_regret_counts),
        "bayes_oracle_slope": estimate_cumulative_slope(cum_regret_bayes),
        "final_repair_loss": cum_repair_loss / t_steps,
        "final_refined_loss": cum_refined_loss / t_steps,
        "final_repair_minus_refined": (cum_repair_loss - cum_refined_loss) / t_steps,
        "dgm_true_loss": dgm_true_loss / t_steps,
        "dgm_noise_loss": dgm_noise_loss / t_steps,
        "dgm_true_accepted": dgm_true.accepted_count,
        "dgm_noise_accepted": dgm_noise.accepted_count,
        "regret_counts_checkpoints": cum_regret_counts[999::1000].tolist(),
        "regret_bayes_checkpoints": cum_regret_bayes[999::1000].tolist(),
    }


def run_bayes_gap_experiment(seed: int, out_dir: Path, quick: bool = False) -> dict[str, object]:
    deltas = [0.0, 0.1, 0.2, 0.3, 0.4]
    repeats = 6 if quick else 16
    t_steps = 30000 if quick else 80000
    k_atoms = 4
    alpha = 0.5
    runs: list[dict[str, object]] = []
    for delta in deltas:
        for rep in range(repeats):
            runs.append(run_one_bayes_gap(seed + 1000 * rep + int(delta * 100), delta, t_steps, k_atoms, alpha))

    by_delta: dict[str, object] = {}
    for delta in deltas:
        rows = [row for row in runs if float(row["delta"]) == delta]
        summary = {
            "theory_gap": float(rows[0]["theory_gap"]),
            **summarize_rows(
                rows,
                [
                    "repair_vs_refined_slope",
                    "bayes_oracle_slope",
                    "final_repair_loss",
                    "final_refined_loss",
                    "final_repair_minus_refined",
                    "dgm_true_loss",
                    "dgm_noise_loss",
                    "dgm_true_accepted",
                    "dgm_noise_accepted",
                ],
            ),
        }
        by_delta[f"{delta:.1f}"] = summary

    write_bayes_gap_csv(out_dir / "bayes_gap_slope_summary.csv", by_delta)
    plot_bayes_gap(out_dir, runs, t_steps)
    return {
        "description": "Known Bayes-gap Bernoulli stream; old atoms observe Z and refined atoms observe (Z,U).",
        "t_steps": t_steps,
        "repeats": repeats,
        "k_atoms": k_atoms,
        "alpha": alpha,
        "by_delta": by_delta,
        "runs": runs,
    }


def write_bayes_gap_csv(path: Path, by_delta: dict[str, object]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "delta",
                "theory_gap",
                "repair_vs_refined_slope_mean",
                "repair_vs_refined_slope_std",
                "bayes_oracle_slope_mean",
                "bayes_oracle_slope_std",
                "dgm_true_accepted_mean",
                "dgm_noise_accepted_mean",
            ]
        )
        for delta, row_obj in by_delta.items():
            row = dict(row_obj)  # type: ignore[arg-type]
            writer.writerow(
                [
                    delta,
                    row["theory_gap"],
                    row["repair_vs_refined_slope_mean"],
                    row["repair_vs_refined_slope_std"],
                    row["bayes_oracle_slope_mean"],
                    row["bayes_oracle_slope_std"],
                    row["dgm_true_accepted_mean"],
                    row["dgm_noise_accepted_mean"],
                ]
            )


def sample_guarded_cell(
    rng: np.random.Generator,
    cell_type: str,
    n: int,
    n_noise: int,
    n_geom: int,
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for _ in range(n):
        sample: dict[str, int] = {}
        u_true = int(rng.integers(0, 2))
        sample["u_true"] = u_true
        if cell_type == "repair":
            y_prob = 0.8
        elif cell_type == "refine":
            y_prob = 0.9 if u_true == 1 else 0.1
        else:
            raise ValueError(cell_type)
        sample["y"] = int(rng.random() < y_prob)
        geometry_latent = int(rng.integers(0, 2))
        for j in range(n_noise):
            sample[f"v{j}"] = int(rng.integers(0, 2))
        for j in range(n_geom):
            if j == 0:
                sample[f"g{j}"] = geometry_latent
            else:
                flip = int(rng.random() < 0.05)
                sample[f"g{j}"] = geometry_latent ^ flip
        rows.append(sample)
    return rows


def proposal_fit_parent(samples: list[dict[str, int]], alpha: float) -> float:
    counts = np.zeros(2, dtype=np.int64)
    for sample in samples:
        counts[int(sample["y"])] += 1
    return float((counts[1] + alpha) / (counts.sum() + 2.0 * alpha))


def eval_fixed_prob_loss(samples: list[dict[str, int]], p: float) -> float:
    return float(np.mean([bernoulli_log_loss(p, int(sample["y"])) for sample in samples]))


def guarded_decision_trial(
    seed: int,
    cell_type: str,
    m_score: int,
    n_noise: int = 20,
    n_geom: int = 6,
    alpha: float = 0.5,
    tau: float = 0.01,
    penalty_weight: float = 0.35,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    proposal = sample_guarded_cell(rng, cell_type, m_score, n_noise, n_geom)
    scoring = sample_guarded_cell(rng, cell_type, m_score, n_noise, n_geom)
    candidates = ["u_true"] + [f"v{j}" for j in range(n_noise)] + [f"g{j}" for j in range(n_geom)]

    current_q = 0.2 if cell_type == "repair" else 0.5
    repaired_q = proposal_fit_parent(proposal, alpha)
    repair_gain = eval_fixed_prob_loss(scoring, current_q) - eval_fixed_prob_loss(scoring, repaired_q)

    penalty = penalty_weight * math.sqrt(math.log(len(candidates) + 1.0) / max(m_score, 1))
    candidate_rows = []
    for candidate in candidates:
        raw_gain = validation_split_gain(proposal + scoring, candidate, alpha=alpha)
        candidate_rows.append(
            {
                "candidate": candidate,
                "raw_gain": raw_gain,
                "penalized_gain": raw_gain - penalty,
                "kind": "true" if candidate == "u_true" else ("noise" if candidate.startswith("v") else "geometry"),
            }
        )
    best = max(candidate_rows, key=lambda row: float(row["penalized_gain"]))
    if float(best["penalized_gain"]) > max(repair_gain, 0.0) + tau:
        decision = "refine"
        accepted = str(best["candidate"])
    elif repair_gain > tau:
        decision = "repair"
        accepted = None
    else:
        decision = "noop"
        accepted = None
    return {
        "seed": seed,
        "cell_type": cell_type,
        "m_score": m_score,
        "repair_gain": repair_gain,
        "penalty": penalty,
        "decision": decision,
        "accepted": accepted,
        "best_candidate": best["candidate"],
        "best_kind": best["kind"],
        "best_raw_gain": best["raw_gain"],
        "best_penalized_gain": best["penalized_gain"],
        "candidate_rows": candidate_rows,
    }


def run_guarded_refinement_experiment(seed: int, out_dir: Path, quick: bool = False) -> dict[str, object]:
    m_values = [32, 64, 128, 256, 512]
    trials = 200 if quick else 800
    all_runs: list[dict[str, object]] = []
    for m_score in m_values:
        for cell_type in ["repair", "refine"]:
            for trial in range(trials):
                all_runs.append(
                    guarded_decision_trial(
                        seed + 100000 * m_score + 1000 * trial + (0 if cell_type == "repair" else 1),
                        cell_type,
                        m_score,
                    )
                )

    by_m: dict[str, object] = {}
    for m_score in m_values:
        rows = [row for row in all_runs if int(row["m_score"]) == m_score]
        repair_rows = [row for row in rows if row["cell_type"] == "repair"]
        refine_rows = [row for row in rows if row["cell_type"] == "refine"]
        true_accepts = [row for row in refine_rows if row["decision"] == "refine" and row["accepted"] == "u_true"]
        false_refines = [row for row in repair_rows if row["decision"] == "refine"]
        spurious_accepts = [
            row
            for row in rows
            if row["decision"] == "refine" and isinstance(row["accepted"], str) and row["accepted"] != "u_true"
        ]
        by_m[str(m_score)] = {
            "true_refinement_recall": len(true_accepts) / len(refine_rows),
            "false_refinement_rate": len(false_refines) / len(repair_rows),
            "spurious_distinction_acceptance": len(spurious_accepts) / len(rows),
            "repair_decision_accuracy": sum(row["decision"] == "repair" for row in repair_rows) / len(repair_rows),
            "refine_cell_repair_error": sum(row["decision"] == "repair" for row in refine_rows) / len(refine_rows),
            "best_raw_gain": mean_and_std(float(row["best_raw_gain"]) for row in rows),
            "best_penalized_gain": mean_and_std(float(row["best_penalized_gain"]) for row in rows),
        }

    write_guarded_csv(out_dir / "guarded_refinement_summary.csv", by_m)
    plot_guarded_refinement(out_dir, all_runs)
    compact_runs = [
        {k: v for k, v in row.items() if k != "candidate_rows"}
        for row in all_runs
    ]
    gain_histogram = collect_gain_histogram(all_runs)
    return {
        "description": "Guarded refine-or-repair decision with true, noise, and geometry-only candidate distinctions.",
        "trials_per_cell_and_m": trials,
        "m_values": m_values,
        "by_m": by_m,
        "gain_histogram": gain_histogram,
        "runs": compact_runs,
    }


def collect_gain_histogram(runs: list[dict[str, object]]) -> dict[str, list[float]]:
    accepted: list[float] = []
    rejected_true: list[float] = []
    rejected_spurious: list[float] = []
    for row in runs:
        if int(row["m_score"]) != 256:
            continue
        accepted_name = row["accepted"] if row["decision"] == "refine" else None
        for candidate in row["candidate_rows"]:  # type: ignore[index]
            item = dict(candidate)
            gain = float(item["penalized_gain"])
            name = str(item["candidate"])
            if accepted_name == name:
                accepted.append(gain)
            elif name == "u_true":
                rejected_true.append(gain)
            else:
                rejected_spurious.append(gain)
    return {
        "accepted_penalized_gain": accepted,
        "rejected_true_penalized_gain": rejected_true,
        "rejected_spurious_penalized_gain_sample": rejected_spurious[:5000],
    }


def write_guarded_csv(path: Path, by_m: dict[str, object]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "m_score",
                "true_refinement_recall",
                "false_refinement_rate",
                "spurious_distinction_acceptance",
                "repair_decision_accuracy",
                "refine_cell_repair_error",
            ]
        )
        for m_score, row_obj in by_m.items():
            row = dict(row_obj)  # type: ignore[arg-type]
            writer.writerow(
                [
                    m_score,
                    row["true_refinement_recall"],
                    row["false_refinement_rate"],
                    row["spurious_distinction_acceptance"],
                    row["repair_decision_accuracy"],
                    row["refine_cell_repair_error"],
                ]
            )


def contextual_xor_stream(
    n: int,
    rng: np.random.Generator,
    signal_noise: float = 0.25,
    context_noise: float = 0.35,
    nuisance_dim: int = 6,
    nuisance_scale: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b1 = rng.integers(0, 2, size=n)
    b2 = rng.integers(0, 2, size=n)
    c = rng.integers(0, 2, size=n)
    x1 = (2 * b1 - 1).astype(float) + signal_noise * rng.normal(size=n)
    x2 = (2 * b2 - 1).astype(float) + signal_noise * rng.normal(size=n)
    x3 = (2 * c - 1).astype(float) + context_noise * rng.normal(size=n)
    nuisance = nuisance_scale * rng.normal(size=(n, nuisance_dim))
    x = np.column_stack([x1, x2, x3, nuisance])
    y = (b1 ^ b2 ^ c).astype(np.int64)
    target = np.column_stack([b1, b2, c]).astype(np.int64)
    order = rng.permutation(n)
    return x[order], y[order], target[order]


def base_key(row: np.ndarray) -> tuple[int, int]:
    return int(row[0] > 0.0), int(row[1] > 0.0)


def candidate_bits(row: np.ndarray) -> dict[str, int]:
    bits = {"c_hat": int(row[2] > 0.0), "geom_abs_x1": int(abs(row[0]) > 1.0), "geom_abs_x2": int(abs(row[1]) > 1.0)}
    for j in range(3, len(row)):
        bits[f"nuisance{j - 3}"] = int(row[j] > 0.0)
    return bits


@dataclass
class ContextCell:
    counts: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.int64))
    samples: list[dict[str, int]] = field(default_factory=list)
    split_name: str | None = None
    child_counts: dict[int, np.ndarray] = field(
        default_factory=lambda: {0: np.zeros(2, dtype=np.int64), 1: np.zeros(2, dtype=np.int64)}
    )
    accepted_gain: float | None = None


class ContextDGM:
    def __init__(
        self,
        repair: bool,
        refine: bool,
        inherit_counts_on_split: bool | None = None,
        alpha: float = 0.5,
        min_samples: int = 96,
        tau: float = 0.02,
        penalty: float = 0.01,
    ) -> None:
        self.repair = repair
        self.refine = refine
        self.inherit_counts_on_split = repair if inherit_counts_on_split is None else inherit_counts_on_split
        self.alpha = alpha
        self.min_samples = min_samples
        self.tau = tau
        self.penalty = penalty
        self.cells: dict[tuple[int, int], ContextCell] = defaultdict(ContextCell)
        self.accepted: list[dict[str, object]] = []
        self.rejected: list[dict[str, object]] = []

    def predict_proba(self, row: np.ndarray) -> np.ndarray:
        cell = self.cells[base_key(row)]
        bits = candidate_bits(row)
        if cell.split_name is not None:
            counts = cell.child_counts[int(bits[cell.split_name])]
        else:
            counts = cell.counts
        total = counts.sum()
        p1 = float((counts[1] + self.alpha) / (total + 2.0 * self.alpha))
        return np.asarray([1.0 - p1, p1], dtype=float)

    def observe(self, row: np.ndarray, y: int) -> None:
        key = base_key(row)
        cell = self.cells[key]
        bits = candidate_bits(row)
        sample = {**bits, "y": int(y)}
        cell.samples.append(sample)
        if self.repair:
            if cell.split_name is None:
                cell.counts[int(y)] += 1
            else:
                cell.child_counts[int(bits[cell.split_name])][int(y)] += 1
        if self.refine and cell.split_name is None and len(cell.samples) >= self.min_samples:
            self._maybe_split(key, cell)

    def _maybe_split(self, key: tuple[int, int], cell: ContextCell) -> None:
        names = [name for name in cell.samples[0] if name != "y"]
        gains = []
        for name in names:
            raw_gain = validation_split_gain(cell.samples, name, alpha=self.alpha)
            gains.append((raw_gain - self.penalty, raw_gain, name))
        gains.sort(reverse=True)
        best_penalized, best_raw, best_name = gains[0]
        record = {"base_key": key, "candidate": best_name, "raw_gain": best_raw, "penalized_gain": best_penalized}
        if best_penalized > self.tau:
            cell.split_name = best_name
            cell.accepted_gain = best_penalized
            cell.child_counts = {0: np.zeros(2, dtype=np.int64), 1: np.zeros(2, dtype=np.int64)}
            if self.inherit_counts_on_split:
                for sample in cell.samples:
                    bit = int(sample[best_name])
                    cell.child_counts[bit][int(sample["y"])] += 1
            self.accepted.append(record)
        else:
            self.rejected.append(record)

    def concept_id(self, row: np.ndarray) -> str:
        key = base_key(row)
        cell = self.cells[key]
        if cell.split_name is None:
            return f"{key}:unsplit"
        bit = candidate_bits(row)[cell.split_name]
        return f"{key}:{cell.split_name}={bit}"

    @property
    def concepts(self) -> int:
        total = 0
        for cell in self.cells.values():
            total += 2 if cell.split_name is not None else 1
        return total

    @property
    def edges(self) -> int:
        return len(self.accepted)


class BudgetedKNN:
    def __init__(self, budget: int, k_neighbors: int = 5, alpha: float = 0.5) -> None:
        self.budget = budget
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.x_seen: list[np.ndarray] = []
        self.y_seen: list[int] = []

    def predict_proba(self, row: np.ndarray) -> np.ndarray:
        if not self.y_seen:
            return np.asarray([0.5, 0.5], dtype=float)
        x = np.stack(self.x_seen)
        y = np.asarray(self.y_seen, dtype=np.int64)
        dist = np.linalg.norm(x - row[None, :], axis=1)
        k = min(self.k_neighbors, len(y))
        idx = np.argpartition(dist, k - 1)[:k]
        counts = np.zeros(2, dtype=float)
        for label in y[idx]:
            counts[int(label)] += 1.0
        counts += self.alpha
        return counts / counts.sum()

    def observe(self, row: np.ndarray, y: int) -> None:
        self.x_seen.append(np.asarray(row, dtype=float))
        self.y_seen.append(int(y))
        if len(self.y_seen) > self.budget:
            self.x_seen.pop(0)
            self.y_seen.pop(0)

    @property
    def concepts(self) -> int:
        return len(self.y_seen)


class OnlineCentroids:
    def __init__(self, max_prototypes: int, create_radius: float = 1.4, alpha: float = 0.5) -> None:
        self.max_prototypes = max_prototypes
        self.create_radius = create_radius
        self.alpha = alpha
        self.centroids: list[np.ndarray] = []
        self.counts: list[np.ndarray] = []
        self.n: list[int] = []

    def _nearest(self, row: np.ndarray) -> tuple[int, float]:
        x = np.stack(self.centroids)
        dist = np.linalg.norm(x - row[None, :], axis=1)
        idx = int(np.argmin(dist))
        return idx, float(dist[idx])

    def predict_proba(self, row: np.ndarray) -> np.ndarray:
        if not self.centroids:
            return np.asarray([0.5, 0.5], dtype=float)
        idx, _ = self._nearest(row)
        counts = self.counts[idx].astype(float) + self.alpha
        return counts / counts.sum()

    def observe(self, row: np.ndarray, y: int) -> None:
        row = np.asarray(row, dtype=float)
        if not self.centroids:
            self.centroids.append(row.copy())
            counts = np.zeros(2, dtype=np.int64)
            counts[int(y)] = 1
            self.counts.append(counts)
            self.n.append(1)
            return
        idx, dist = self._nearest(row)
        if dist > self.create_radius and len(self.centroids) < self.max_prototypes:
            self.centroids.append(row.copy())
            counts = np.zeros(2, dtype=np.int64)
            counts[int(y)] = 1
            self.counts.append(counts)
            self.n.append(1)
            return
        self.n[idx] += 1
        eta = 1.0 / self.n[idx]
        self.centroids[idx] = (1.0 - eta) * self.centroids[idx] + eta * row
        self.counts[idx][int(y)] += 1

    def concept_id(self, row: np.ndarray) -> str:
        if not self.centroids:
            return "empty"
        idx, _ = self._nearest(row)
        return f"prototype:{idx}"

    @property
    def concepts(self) -> int:
        return len(self.centroids)


def online_binary_metrics(model: object, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    correct = 0
    log_loss = 0.0
    clipped_log_loss = 0.0
    for row, label in zip(x, y, strict=True):
        proba = model.predict_proba(row)
        p = float(proba[int(label)])
        correct += int(np.argmax(proba) == int(label))
        log_loss += -math.log(max(p, EPS))
        clipped_log_loss += -math.log(min(max(p, 1e-4), 1.0 - 1e-4))
        model.observe(row, int(label))
    return {
        "prequential_accuracy": correct / len(y),
        "prequential_nll": log_loss / len(y),
        "prequential_clipped_nll": clipped_log_loss / len(y),
    }


def evaluate_binary(model: object, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    correct = 0
    log_loss = 0.0
    for row, label in zip(x, y, strict=True):
        proba = model.predict_proba(row)
        correct += int(np.argmax(proba) == int(label))
        log_loss += -math.log(max(float(proba[int(label)]), EPS))
    return {"test_accuracy": correct / len(y), "test_nll": log_loss / len(y)}


def target_labels(target: np.ndarray) -> list[str]:
    return [f"{int(a)}{int(b)}{int(c)}" for a, b, c in target]


def weighted_purity(true_labels: list[str], pred_labels: list[str]) -> float:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for true, pred in zip(true_labels, pred_labels, strict=True):
        groups[pred][true] += 1
    total = len(true_labels)
    if total == 0:
        return 0.0
    return float(sum(max(counter.values()) for counter in groups.values()) / total)


def adjusted_rand_index(true_labels: list[str], pred_labels: list[str]) -> float:
    contingency: dict[tuple[str, str], int] = Counter(zip(true_labels, pred_labels, strict=True))
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    n = len(true_labels)
    if n < 2:
        return 1.0

    def comb2(x: int) -> float:
        return x * (x - 1) / 2.0

    sum_comb = sum(comb2(v) for v in contingency.values())
    true_comb = sum(comb2(v) for v in true_counts.values())
    pred_comb = sum(comb2(v) for v in pred_counts.values())
    total_comb = comb2(n)
    expected = true_comb * pred_comb / total_comb if total_comb > 0 else 0.0
    maximum = 0.5 * (true_comb + pred_comb)
    denom = maximum - expected
    if abs(denom) < EPS:
        return 1.0
    return float((sum_comb - expected) / denom)


def concept_metrics(model: object, x: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    if not hasattr(model, "concept_id"):
        return {"concept_purity": None, "ari": None}
    true = target_labels(target)
    pred = [model.concept_id(row) for row in x]  # type: ignore[attr-defined]
    return {"concept_purity": weighted_purity(true, pred), "ari": adjusted_rand_index(true, pred)}


def run_one_contextual_xor_ablation(seed: int, quick: bool = False) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_train = 1800 if quick else 5000
    n_test = 1800 if quick else 5000
    x_train, y_train, target_train = contextual_xor_stream(n_train, rng)
    x_test, y_test, target_test = contextual_xor_stream(n_test, rng)

    models: dict[str, object] = {
        "full_dgm_refine_repair_edges": ContextDGM(repair=True, refine=True),
        "dgm_repair_only": ContextDGM(repair=True, refine=False),
        "dgm_refine_only_no_repair": ContextDGM(repair=False, refine=True, inherit_counts_on_split=False),
        "no_edge_centroid_growth": OnlineCentroids(max_prototypes=8, create_radius=1.2),
        "knn_cache_same_budget": BudgetedKNN(budget=8, k_neighbors=5),
        "knn_cache_large_budget": BudgetedKNN(budget=64, k_neighbors=5),
    }

    out: dict[str, object] = {"seed": seed, "n_train": n_train, "n_test": n_test}
    for name, model in models.items():
        train_metrics = online_binary_metrics(model, x_train, y_train)
        test_metrics = evaluate_binary(model, x_test, y_test)
        structure = concept_metrics(model, x_train, target_train)
        row: dict[str, object] = {**train_metrics, **test_metrics, **structure}
        if hasattr(model, "concepts"):
            row["concepts"] = int(model.concepts)  # type: ignore[attr-defined]
        if isinstance(model, ContextDGM):
            row["edges"] = int(model.edges)
            row["accepted_context_edges"] = sum(rec["candidate"] == "c_hat" for rec in model.accepted)
            row["spurious_edges"] = sum(rec["candidate"] != "c_hat" for rec in model.accepted)
            row["accepted_gain_mean"] = (
                float(np.mean([float(rec["penalized_gain"]) for rec in model.accepted])) if model.accepted else 0.0
            )
            row["candidate_recall"] = row["accepted_context_edges"] / 4.0
            row["accepted_edges"] = model.accepted
        out[name] = row
    return out


def run_contextual_xor_ablation(seed: int, out_dir: Path, quick: bool = False) -> dict[str, object]:
    repeats = 4 if quick else 10
    runs = [run_one_contextual_xor_ablation(seed + 1000 * i, quick=quick) for i in range(repeats)]
    names = [
        "full_dgm_refine_repair_edges",
        "dgm_repair_only",
        "dgm_refine_only_no_repair",
        "no_edge_centroid_growth",
        "knn_cache_same_budget",
        "knn_cache_large_budget",
    ]
    summary: dict[str, object] = {
        "description": "Contextual XOR with initial (b1,b2) atoms; the useful new edge is the context distinction.",
        "repeats": repeats,
        "runs": runs,
    }
    for name in names:
        rows = [dict(run[name]) for run in runs]  # type: ignore[arg-type]
        keys = ["prequential_accuracy", "prequential_nll", "prequential_clipped_nll", "test_accuracy", "test_nll", "concepts"]
        if all("edges" in row for row in rows):
            keys.extend(["edges", "accepted_context_edges", "spurious_edges", "candidate_recall", "accepted_gain_mean"])
        if all(row.get("concept_purity") is not None for row in rows):
            keys.extend(["concept_purity", "ari"])
        summary[name] = summarize_rows(rows, keys)

    write_contextual_xor_csv(out_dir / "contextual_xor_ablation_summary.csv", summary, names)
    plot_contextual_xor(out_dir, summary, names)
    return summary


def write_contextual_xor_csv(path: Path, summary: dict[str, object], names: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "prequential_accuracy_mean",
                "prequential_nll_mean",
                "test_accuracy_mean",
                "concepts_mean",
                "edges_mean",
                "accepted_context_edges_mean",
                "spurious_edges_mean",
                "candidate_recall_mean",
                "concept_purity_mean",
                "ari_mean",
            ]
        )
        for name in names:
            row = dict(summary[name])  # type: ignore[arg-type]
            writer.writerow(
                [
                    name,
                    row.get("prequential_accuracy_mean"),
                    row.get("prequential_nll_mean"),
                    row.get("test_accuracy_mean"),
                    row.get("concepts_mean"),
                    row.get("edges_mean"),
                    row.get("accepted_context_edges_mean"),
                    row.get("spurious_edges_mean"),
                    row.get("candidate_recall_mean"),
                    row.get("concept_purity_mean"),
                    row.get("ari_mean"),
                ]
            )


def plot_bayes_gap(out_dir: Path, runs: list[dict[str, object]], t_steps: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    deltas = sorted({float(row["delta"]) for row in runs})
    theory = []
    slope_mean = []
    slope_std = []
    bayes_mean = []
    for delta in deltas:
        rows = [row for row in runs if float(row["delta"]) == delta]
        theory.append(float(rows[0]["theory_gap"]))
        slopes = np.asarray([float(row["repair_vs_refined_slope"]) for row in rows])
        bayes = np.asarray([float(row["bayes_oracle_slope"]) for row in rows])
        slope_mean.append(float(slopes.mean()))
        slope_std.append(float(slopes.std(ddof=0)))
        bayes_mean.append(float(bayes.mean()))

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.errorbar(theory, slope_mean, yerr=slope_std, marker="o", capsize=3, label="repair vs refined counts")
    ax.plot(theory, bayes_mean, marker="s", linestyle="--", label="Bayes oracle empirical")
    lim = max(theory) * 1.08 + 1e-9
    ax.plot([0.0, lim], [0.0, lim], color="black", linewidth=1.0, label="y=x")
    ax.set_xlabel("theoretical gap I(Y;U|Z) (nats)")
    ax.set_ylabel("estimated regret slope (nats/sample)")
    ax.set_title("Regret slope matches refinement gap")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "bayes_gap_slope_vs_theory.png", dpi=180)
    plt.close(fig)

    checkpoints = np.arange(1000, t_steps + 1, 1000)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for delta in deltas:
        if delta == 0.0:
            continue
        rows = [row for row in runs if float(row["delta"]) == delta]
        curves = np.asarray([row["regret_counts_checkpoints"] for row in rows], dtype=float)
        ax.plot(checkpoints[: curves.shape[1]], curves.mean(axis=0), label=f"delta={delta:.1f}")
    ax.set_xlabel("time")
    ax.set_ylabel("cumulative regret vs refined counts")
    ax.set_title("Repair-only regret is linear when U is useful")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "bayes_gap_cumulative_regret.png", dpi=180)
    plt.close(fig)

    zero_rows = [row for row in runs if float(row["delta"]) == 0.0]
    if zero_rows:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        curves = np.asarray([row["regret_counts_checkpoints"] for row in zero_rows], dtype=float)
        ax.plot(checkpoints[: curves.shape[1]], curves.mean(axis=0), label="repair - refined")
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel("time")
        ax.set_ylabel("cumulative regret")
        ax.set_title("Delta=0 negative control")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "bayes_gap_delta_zero_control.png", dpi=180)
        plt.close(fig)


def plot_guarded_refinement(out_dir: Path, runs: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m_values = sorted({int(row["m_score"]) for row in runs})
    true_recall = []
    false_rate = []
    spurious = []
    repair_acc = []
    for m in m_values:
        rows = [row for row in runs if int(row["m_score"]) == m]
        repair_rows = [row for row in rows if row["cell_type"] == "repair"]
        refine_rows = [row for row in rows if row["cell_type"] == "refine"]
        true_recall.append(sum(row["decision"] == "refine" and row["accepted"] == "u_true" for row in refine_rows) / len(refine_rows))
        false_rate.append(sum(row["decision"] == "refine" for row in repair_rows) / len(repair_rows))
        spurious.append(
            sum(row["decision"] == "refine" and row["accepted"] != "u_true" for row in rows) / len(rows)
        )
        repair_acc.append(sum(row["decision"] == "repair" for row in repair_rows) / len(repair_rows))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(m_values, true_recall, marker="o", label="true refinement recall")
    ax.plot(m_values, false_rate, marker="s", label="false refinement rate")
    ax.plot(m_values, spurious, marker="^", label="spurious acceptance")
    ax.plot(m_values, repair_acc, marker="d", label="repair decision accuracy")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("scoring buffer size")
    ax.set_ylabel("rate")
    ax.set_title("Guarded refinement rejects zero-value distinctions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "guarded_sample_size_curve.png", dpi=180)
    plt.close(fig)

    rows_256 = [row for row in runs if int(row["m_score"]) == 256]
    accepted = [float(row["best_penalized_gain"]) for row in rows_256 if row["decision"] == "refine"]
    rejected = [float(row["best_penalized_gain"]) for row in rows_256 if row["decision"] != "refine"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(rejected, bins=35, alpha=0.65, label="rejected best candidates")
    ax.hist(accepted, bins=35, alpha=0.65, label="accepted refinements")
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("best penalized validation gain")
    ax.set_ylabel("count")
    ax.set_title("Accepted vs rejected validation gains (m=256)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "guarded_gain_histogram.png", dpi=180)
    plt.close(fig)


def plot_contextual_xor(out_dir: Path, summary: dict[str, object], names: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [
        "Full DGM",
        "Repair only",
        "Refine only",
        "Centroid",
        "kNN-8",
        "kNN-64",
    ]
    acc = [float(dict(summary[name])["prequential_accuracy_mean"]) for name in names]  # type: ignore[arg-type]
    acc_std = [float(dict(summary[name])["prequential_accuracy_std"]) for name in names]  # type: ignore[arg-type]
    purity = [dict(summary[name]).get("concept_purity_mean") for name in names]  # type: ignore[arg-type]
    concepts = [float(dict(summary[name])["concepts_mean"]) for name in names]  # type: ignore[arg-type]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    axes[0].bar(x, acc, yerr=acc_std, capsize=3)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("prequential accuracy")
    axes[0].set_title("Online prediction")

    pur = [float(v) if v is not None else 0.0 for v in purity]
    axes[1].bar(x, pur)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("concept purity")
    axes[1].set_title("Recovered structure")

    axes[2].bar(x, concepts)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")
    axes[2].set_ylabel("concepts / examples")
    axes[2].set_title("Growth")
    fig.tight_layout()
    fig.savefig(out_dir / "contextual_xor_ablation.png", dpi=180)
    plt.close(fig)


def write_markdown_summary(path: Path, results: dict[str, object]) -> None:
    def metric(row: dict[str, object], key: str, precision: int = 3) -> str:
        value = row.get(key)
        if value is None:
            return "--"
        return f"{float(value):.{precision}f}"

    bayes = dict(results["known_bayes_gap"])  # type: ignore[arg-type]
    guarded = dict(results["guarded_refinement"])  # type: ignore[arg-type]
    xor = dict(results["contextual_xor_ablation"])  # type: ignore[arg-type]
    lines = [
        "# Refinement Theory Experiment Summary",
        "",
        "## 1. Known Bayes-gap / regret-slope calibration",
        "",
        "| delta | theory gap | repair/refined slope | Bayes-oracle slope | DGM U accepts | DGM noise accepts |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for delta, row_obj in dict(bayes["by_delta"]).items():  # type: ignore[arg-type]
        row = dict(row_obj)  # type: ignore[arg-type]
        lines.append(
            f"| {delta} | {row['theory_gap']:.4f} | "
            f"{row['repair_vs_refined_slope_mean']:.4f} +/- {row['repair_vs_refined_slope_std']:.4f} | "
            f"{row['bayes_oracle_slope_mean']:.4f} +/- {row['bayes_oracle_slope_std']:.4f} | "
            f"{row['dgm_true_accepted_mean']:.2f} | {row['dgm_noise_accepted_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 2. Guarded refine-or-repair with spurious distinctions",
            "",
            "| scoring m | true recall | false refine | spurious accept | repair accuracy |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    guarded_by_m = dict(guarded["by_m"])  # type: ignore[arg-type]
    for m in sorted(guarded_by_m, key=lambda value: int(value)):
        row = dict(guarded_by_m[m])  # type: ignore[arg-type]
        lines.append(
            f"| {m} | {row['true_refinement_recall']:.3f} | {row['false_refinement_rate']:.3f} | "
            f"{row['spurious_distinction_acceptance']:.3f} | {row['repair_decision_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 3. Contextual XOR graph-memory ablation",
            "",
            "| model | preq acc | preq NLL | test acc | concepts | edges | context edges | spurious edges | cand. recall | purity | ARI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    model_labels = {
        "full_dgm_refine_repair_edges": "Full DGM refine+repair+edges",
        "dgm_repair_only": "DGM repair-only",
        "dgm_refine_only_no_repair": "DGM refine-only",
        "no_edge_centroid_growth": "No-edge centroid",
        "knn_cache_same_budget": "kNN/cache same budget",
        "knn_cache_large_budget": "kNN/cache large budget",
    }
    for name, label in model_labels.items():
        row = dict(xor[name])  # type: ignore[index,arg-type]
        lines.append(
            f"| {label} | {row['prequential_accuracy_mean']:.3f} +/- {row['prequential_accuracy_std']:.3f} | "
            f"{row['prequential_nll_mean']:.3f} | {row['test_accuracy_mean']:.3f} | "
            f"{row['concepts_mean']:.1f} | {metric(row, 'edges_mean', 1)} | "
            f"{metric(row, 'accepted_context_edges_mean', 1)} | {metric(row, 'spurious_edges_mean', 1)} | "
            f"{metric(row, 'candidate_recall_mean', 3)} | {metric(row, 'concept_purity_mean', 3)} | "
            f"{metric(row, 'ari_mean', 3)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/refinement_theory_results"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "known_bayes_gap": run_bayes_gap_experiment(args.seed + 100, out_dir, quick=args.quick),
        "guarded_refinement": run_guarded_refinement_experiment(args.seed + 200, out_dir, quick=args.quick),
        "contextual_xor_ablation": run_contextual_xor_ablation(args.seed + 300, out_dir, quick=args.quick),
    }
    json_path = out_dir / "results_refinement_theory.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_summary(out_dir / "summary.md", results)

    print("Known Bayes-gap / regret-slope calibration")
    for delta, row_obj in dict(results["known_bayes_gap"]["by_delta"]).items():  # type: ignore[index]
        row = dict(row_obj)  # type: ignore[arg-type]
        print(
            f"  delta={delta}: theory={row['theory_gap']:.4f}, "
            f"slope={row['repair_vs_refined_slope_mean']:.4f}+/-{row['repair_vs_refined_slope_std']:.4f}, "
            f"bayes={row['bayes_oracle_slope_mean']:.4f}"
        )
    print("Guarded refine-or-repair")
    for m, row_obj in dict(results["guarded_refinement"]["by_m"]).items():  # type: ignore[index]
        row = dict(row_obj)  # type: ignore[arg-type]
        print(
            f"  m={m}: true_recall={row['true_refinement_recall']:.3f}, "
            f"false_refine={row['false_refinement_rate']:.3f}, "
            f"spurious={row['spurious_distinction_acceptance']:.3f}, "
            f"repair_acc={row['repair_decision_accuracy']:.3f}"
        )
    print("Contextual XOR graph-memory ablation")
    for name in [
        "full_dgm_refine_repair_edges",
        "dgm_repair_only",
        "dgm_refine_only_no_repair",
        "no_edge_centroid_growth",
        "knn_cache_same_budget",
        "knn_cache_large_budget",
    ]:
        row = dict(results["contextual_xor_ablation"][name])  # type: ignore[index,arg-type]
        print(
            f"  {name:30s} preq_acc={row['prequential_accuracy_mean']:.3f}+/-{row['prequential_accuracy_std']:.3f} "
            f"test_acc={row['test_accuracy_mean']:.3f} concepts={row['concepts_mean']:.1f}"
        )
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
