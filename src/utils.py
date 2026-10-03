"""Utilitaires partagés : métriques, standardisation et fichiers."""

from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def fit_standardizer(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def standardize(X, mean, std):
    return (X - mean) / std


def regression_metrics(y, pred):
    err = pred - y
    mse = float(np.mean(err**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
    }


def confusion(y, pred, classes):
    matrix = np.zeros((classes, classes), dtype=int)
    for a, b in zip(y.astype(int), pred.astype(int)):
        matrix[a, b] += 1
    return matrix


def classification_metrics(y, probabilities):
    pred = probabilities.argmax(axis=1)
    k = probabilities.shape[1]
    cm = confusion(y, pred, k)
    precision, recall, f1 = [], [], []
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p); recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    ce = -np.mean(np.log(np.clip(probabilities[np.arange(len(y)), y.astype(int)], 1e-12, 1.0)))
    return {
        "accuracy": float(np.mean(pred == y)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "cross_entropy": float(ce),
        "f1_per_class": f1,
        "confusion": cm.tolist(),
    }


def save_rows(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys and not isinstance(row[key], (list, dict)):
                keys.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in keys} for r in rows])


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def central_difference_check(loss_fn, grad_fn, theta, coordinates=8, h=1e-5, seed=42):
    rng = np.random.default_rng(seed)
    ids = rng.choice(theta.size, size=min(coordinates, theta.size), replace=False)
    analytic = grad_fn(theta, None)
    errors = []
    for index in ids:
        plus, minus = theta.copy(), theta.copy()
        plus[index] += h; minus[index] -= h
        numeric = (loss_fn(plus) - loss_fn(minus)) / (2 * h)
        denominator = max(1e-12, abs(numeric) + abs(analytic[index]))
        errors.append(abs(numeric - analytic[index]) / denominator)
    return float(max(errors))

