"""Classifier evaluation metrics."""
import numpy as np


def accuracy(y, proba):
    return np.mean(np.argmax(proba, axis=1) == y)


def log_loss(y, proba, eps=1e-15):
    p = np.clip(proba[np.arange(len(y)), y], eps, 1.0)
    return -np.mean(np.log(p))


def brier(y, proba):
    onehot = np.eye(proba.shape[1])[y]
    return np.mean(np.sum((proba - onehot) ** 2, axis=1))


def ece(y, proba, n_bins=10):
    """Expected calibration error on the confidence of the predicted class."""
    conf = proba.max(axis=1)
    correct = (np.argmax(proba, axis=1) == y).astype(float)
    bins = np.linspace(0.5 if proba.shape[1] == 2 else 0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def reliability_curve(y, proba, n_bins=10):
    """Fraction of positives vs mean predicted P(class 1), for a reliability diagram."""
    p1 = proba[:, 1]
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p1, bins) - 1, 0, n_bins - 1)
    mean_pred, frac_pos = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 20:
            mean_pred.append(p1[m].mean())
            frac_pos.append((y[m] == 1).mean())
    return np.array(mean_pred), np.array(frac_pos)


ALL_METRICS = {"accuracy": accuracy, "log_loss": log_loss, "brier": brier, "ece": ece}
