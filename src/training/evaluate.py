"""
src/training/evaluate.py
OWNER: Person 5 (Training loop, eval, demo)

Evaluation metrics for the 3-class prediction task.
"""
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import numpy as np


LABEL_NAMES = ["down", "flat", "up"]


def compute_metrics(preds: list[int], labels: list[int]) -> dict:
    """
    Returns dict of standard classification metrics.
    """
    return {
        "accuracy":    accuracy_score(labels, preds),
        "f1_macro":    f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "f1_up":       f1_score(labels, preds, labels=[2], average="micro", zero_division=0),
        "f1_down":     f1_score(labels, preds, labels=[0], average="micro", zero_division=0),
    }


def print_report(preds: list[int], labels: list[int]):
    print(classification_report(labels, preds, target_names=LABEL_NAMES, zero_division=0))
    cm = confusion_matrix(labels, preds)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"       {'  '.join(LABEL_NAMES)}")
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:4s}  {row}")


def directional_accuracy(preds: list[int], labels: list[int]) -> float:
    """
    Accuracy on just the up/down calls (ignoring flat predictions and flat labels).
    More finance-relevant than overall accuracy.
    """
    preds   = np.array(preds)
    labels  = np.array(labels)
    mask    = (labels != 1) & (preds != 1)
    if mask.sum() == 0:
        return float("nan")
    return accuracy_score(labels[mask], preds[mask])
