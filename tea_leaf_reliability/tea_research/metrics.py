"""Probability calibration and paired evaluation for a frozen test set."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def probabilities(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def nll(logits, labels, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(labels)), labels]))


def fit_temperature(logits, labels):
    """Fit one positive temperature on the dedicated source calibration set."""
    result = minimize_scalar(lambda log_t: nll(logits, labels, np.exp(log_t)),
                             method="bounded", bounds=(-3.0, 3.0))
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("Temperature fitting failed. Check exported logits.")
    candidate = float(np.exp(result.x))
    return candidate if result.fun < nll(logits, labels) else 1.0


def coverage_threshold(scores, target):
    if not 0 < target <= 1:
        raise ValueError("Coverage target must be greater than 0 and at most 1.")
    scores = np.asarray(scores)
    if not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("Valid calibration scores are required.")
    k = math.ceil(target * len(scores))
    return float(np.sort(scores)[-k])


def confusion(y, pred, k):
    return np.bincount(np.asarray(y, dtype=int) * k + np.asarray(pred, dtype=int),
                       minlength=k * k).reshape(k, k)


def macro_f1(y, pred, k):
    cm = confusion(y, pred, k)
    support = cm.sum(axis=1)
    denom = support + cm.sum(axis=0)
    active = support > 0
    return float(np.mean(2 * np.diag(cm)[active] / denom[active]))


def brier_values(y, probs):
    probs = np.asarray(probs, dtype=np.float64)
    return np.sum(probs ** 2, axis=1) - 2 * probs[np.arange(len(y)), y] + 1


def evaluate_probs(y, probs, classes, threshold, bins=10):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    pred, confidence = p.argmax(axis=1), p.max(axis=1)
    correct = pred == y
    accept = confidence >= threshold
    k = len(classes)
    cm = confusion(y, pred, k)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    reliability, ece = [], 0.0
    bin_ids = np.minimum(np.floor(confidence * bins).astype(int), bins - 1)
    for i in range(bins):
        mask = bin_ids == i
        n = int(mask.sum())
        avg_conf = float(confidence[mask].mean()) if n else None
        accuracy = float(correct[mask].mean()) if n else None
        if n:
            ece += n / len(y) * abs(avg_conf - accuracy)
        reliability.append({"low": i / bins, "high": (i + 1) / bins, "n": n,
                            "mean_confidence": avg_conf, "accuracy": accuracy})
    classes_report = {}
    for i, label in enumerate(classes):
        tp = int(cm[i, i])
        n = int(support[i])
        npred = int(predicted[i])
        classes_report[label] = {"support": n, "predicted": npred,
                                 "recall": tp / n if n else None,
                                 "precision": tp / npred if npred else None,
                                 "f1": 2 * tp / (n + npred) if n else None}
    return {
        "n": len(y), "accuracy": float(correct.mean()), "macro_f1": macro_f1(y, pred, k),
        "brier": float(brier_values(y, p).mean()), "ece": ece,
        "mean_confidence": float(confidence.mean()), "threshold": float(threshold),
        "coverage": float(accept.mean()), "accepted": int(accept.sum()),
        "retained_error": float((~correct[accept]).mean()) if accept.any() else None,
        "per_class": classes_report, "confusion_matrix": cm.tolist(), "reliability_bins": reliability,
    }


def risk_coverage(y, probs):
    """Evaluate confidence cutoffs descriptively; do not tune on this curve."""
    pred, confidence = probs.argmax(axis=1), probs.max(axis=1)
    order = np.argsort(-confidence, kind="stable")
    sorted_conf = confidence[order]
    cumulative_error = np.cumsum(pred[order] != y[order])
    # Equal scores must enter together because a threshold cannot split ties.
    ends = np.flatnonzero(np.r_[sorted_conf[:-1] != sorted_conf[1:], True])
    return [{"threshold": float(sorted_conf[i]), "coverage": (int(i) + 1) / len(y),
             "retained_error": float(cumulative_error[i] / (i + 1))}
            for i in ends]


def paired_intervals(y, groups, predictions, n_samples=500, seed=42):
    """Resample the same groups for every arm; intervals are conditional on this fold."""
    y = np.asarray(y)
    pa, pb, pc, pd = (predictions[key] for key in ("fp32_raw", "fp32_cal", "int8_raw", "int8_cal"))
    k = pa.shape[1]
    a, c = pa.argmax(axis=1), pc.argmax(axis=1)
    brier_fp = brier_values(y, pb) - brier_values(y, pa)
    brier_int = brier_values(y, pd) - brier_values(y, pc)
    group_rows = defaultdict(list)
    for i, group in enumerate(groups):
        group_rows[group].append(i)
    indices = [np.asarray(v, dtype=int) for v in group_rows.values()]
    rng = np.random.default_rng(seed)

    def differences(idx):
        return (macro_f1(y[idx], c[idx], k) - macro_f1(y[idx], a[idx], k),
                float(brier_fp[idx].mean()), float(brier_int[idx].mean()))

    observed = differences(np.arange(len(y)))
    draws = []
    if len(indices) > 1:
        for _ in range(n_samples):
            sample = np.concatenate([indices[i] for i in rng.integers(len(indices), size=len(indices))])
            if len(np.unique(y[sample])) == k:
                draws.append(differences(sample))
    names = ("int8_minus_fp32_macro_f1", "fp32_cal_minus_raw_brier", "int8_cal_minus_raw_brier")
    intervals = np.percentile(draws, [2.5, 97.5], axis=0) if len(draws) >= 20 else None
    return {
        name: {"difference": observed[i], "ci95": [float(x) for x in intervals[:, i]]
               if intervals is not None else None,
               "bootstrap_draws_used": len(draws), "groups": len(indices)}
        for i, name in enumerate(names)
    }
