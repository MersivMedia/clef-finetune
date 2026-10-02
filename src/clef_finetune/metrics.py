"""Evaluation metrics, including the business metric
'auto-decide coverage at target precision'."""

from __future__ import annotations

import math
from typing import Any, Sequence


def accuracy(preds: Sequence[int], targets: Sequence[int]) -> float:
    return sum(p == t for p, t in zip(preds, targets)) / len(targets) if targets else float("nan")


def macro_f1(preds: Sequence[int], targets: Sequence[int], n_classes: int) -> float:
    """Macro-F1 over classes that appear in targets or predictions."""
    f1s = []
    for c in range(n_classes):
        tp = sum(p == c and t == c for p, t in zip(preds, targets))
        fp = sum(p == c and t != c for p, t in zip(preds, targets))
        fn = sum(p != c and t == c for p, t in zip(preds, targets))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s) if f1s else float("nan")


def nll(probs: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    return sum(-math.log(max(p[t], 1e-12)) for p, t in zip(probs, targets)) / len(targets)


def brier_score(probs: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    total = 0.0
    for p, t in zip(probs, targets):
        total += sum((pi - (1.0 if i == t else 0.0)) ** 2 for i, pi in enumerate(p))
    return total / len(targets)


def ece(probs: Sequence[Sequence[float]], targets: Sequence[int], n_bins: int = 10) -> float:
    """Top-label expected calibration error with equal-width bins."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, t in zip(probs, targets):
        conf = max(p)
        pred = max(range(len(p)), key=p.__getitem__)
        bins[min(int(conf * n_bins), n_bins - 1)].append((conf, pred == t))
    n = len(targets)
    total = 0.0
    for b in bins:
        if b:
            total += len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b))
    return total


def coverage_at_precision(
    confidences: Sequence[float], correct: Sequence[bool], target_precision: float = 0.99
) -> dict[str, Any]:
    """Largest fraction of cases the model can decide alone (confidence >= threshold)
    while precision on those auto-decided cases stays >= target_precision.

    Returns coverage, the threshold, and achieved precision. If no threshold meets the
    target, coverage is 0 and threshold is None (everything goes to a human).
    Note: with small eval sets this is noisy -- 99% precision needs hundreds of cases
    to be meaningful.
    """
    pairs = sorted(zip(confidences, correct), key=lambda x: -x[0])
    n = len(pairs)
    best = {"coverage": 0.0, "threshold": None, "precision": None, "n_auto": 0, "n": n, "target_precision": target_precision}
    hits = 0
    for i, (conf, ok) in enumerate(pairs, 1):
        hits += bool(ok)
        # only cut between distinct confidence values
        if i < n and pairs[i][0] == conf:
            continue
        prec = hits / i
        if prec >= target_precision:
            best = {
                "coverage": i / n,
                "threshold": conf,
                "precision": prec,
                "n_auto": i,
                "n": n,
                "target_precision": target_precision,
            }
    return best


def summarize(rows: list[dict[str, Any]], target_precisions: Sequence[float] = (0.99, 0.95)) -> dict[str, Any]:
    """rows: dicts with question_id, n_options, probs (list), target (int)."""
    by_q: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_q.setdefault(r["question_id"], []).append(r)

    def block(items: list[dict[str, Any]]) -> dict[str, Any]:
        probs = [r["probs"] for r in items]
        targets = [r["target"] for r in items]
        preds = [max(range(len(p)), key=p.__getitem__) for p in probs]
        out = {
            "n": len(items),
            "accuracy": accuracy(preds, targets),
            "nll": nll(probs, targets),
            "brier": brier_score(probs, targets),
            "ece": ece(probs, targets),
        }
        if len({len(p) for p in probs}) == 1:
            out["macro_f1"] = macro_f1(preds, targets, len(probs[0]))
        conf = [max(p) for p in probs]
        correct = [a == b for a, b in zip(preds, targets)]
        out["auto_decide"] = {f"{tp:g}": coverage_at_precision(conf, correct, tp) for tp in target_precisions}
        return out

    return {"overall": block(rows) if rows else {}, "per_question": {q: block(v) for q, v in by_q.items()}}
