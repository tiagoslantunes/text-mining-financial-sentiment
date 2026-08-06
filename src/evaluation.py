"""Standardised evaluation framework following the professor's Lab Extra approach."""

import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = {
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
}


def evaluate_model(model, X, y, name: str) -> dict:
    """5-fold stratified CV — Recall, Precision, Accuracy, F1.

    StratifiedKFold preserves class distribution (critical for imbalanced data).
    Returns all metrics as a flat dict for building comparison tables.
    """
    t0 = time.time()
    cv = cross_validate(
        model, X, y, cv=CV, scoring=SCORING,
        return_train_score=True, n_jobs=2,
    )
    elapsed = time.time() - t0

    result = {
        "Model": name,
        "F1-macro": cv["test_f1_macro"].mean(),
        "F1-macro_std": cv["test_f1_macro"].std(),
        "F1-weighted": cv["test_f1_weighted"].mean(),
        "Accuracy": cv["test_accuracy"].mean(),
        "Precision-macro": cv["test_precision_macro"].mean(),
        "Recall-macro": cv["test_recall_macro"].mean(),
        "Train_F1-macro": cv["train_f1_macro"].mean(),
        "Overfit_gap": cv["train_f1_macro"].mean() - cv["test_f1_macro"].mean(),
        "Time_s": elapsed,
    }

    print(
        f"\n{'='*55}\n{name}\n"
        f"F1-macro:  {result['F1-macro']:.4f} ± {result['F1-macro_std']:.4f}\n"
        f"Accuracy:  {result['Accuracy']:.4f}\n"
        f"Precision: {result['Precision-macro']:.4f} | Recall: {result['Recall-macro']:.4f}\n"
        f"Overfit gap: {result['Overfit_gap']:.4f} | Time: {elapsed:.1f}s"
    )
    return result


def plot_confusion_matrix(y_true, y_pred, class_names, title: str, save_path: str):
    """Normalized and absolute confusion matrix side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=axes[0],
        display_labels=class_names, cmap="Blues", normalize="true",
    )
    axes[0].set_title(f"{title} — Normalized")
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=axes[1],
        display_labels=class_names, cmap="Blues", normalize=None,
    )
    axes[1].set_title(f"{title} — Absolute")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_classification_report(y_true, y_pred):
    print(classification_report(
        y_true, y_pred,
        target_names=["Bearish", "Bullish", "Neutral"],
    ))
