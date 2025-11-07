import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader


matplotlib.use("Agg")


MetricDict = Dict[str, float]


@dataclass
class EvaluationDetails:
    """Container for detailed evaluation outputs."""

    metrics: MetricDict
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    average_loss: float


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> MetricDict:
    """Compute standard multi-class metrics."""
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    metrics: MetricDict = {
        "acc": accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
    }
    try:
        metrics["auc_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        metrics["auc_micro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="micro")
    except Exception:
        metrics["auc_macro"] = float("nan")
        metrics["auc_micro"] = float("nan")
    return metrics


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool = False,
) -> EvaluationDetails:
    """
    Run a full evaluation pass returning metrics and raw predictions.

    Parameters
    ----------
    model:
        PyTorch module in eval mode.
    loader:
        DataLoader that yields tensors and labels.
    device:
        Torch device to run inference on.
    criterion:
        Loss function (e.g., cross entropy).
    """
    model.eval()
    running_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.squeeze().long().to(device)
            with autocast(enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

            running_loss += loss.item() * inputs.size(0)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(torch.argmax(probs, dim=1).detach().cpu().numpy().tolist())
            y_prob.append(probs.detach().cpu().numpy())

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    y_prob_arr = np.concatenate(y_prob, axis=0) if y_prob else np.empty((0, 0))
    metrics = _compute_classification_metrics(y_true_arr, y_pred_arr, y_prob_arr)
    avg_loss = running_loss / len(loader.dataset)
    metrics["loss"] = avg_loss

    return EvaluationDetails(
        metrics=metrics,
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        y_prob=y_prob_arr,
        average_loss=avg_loss,
    )


def save_metrics_json(metrics: MetricDict, out_path: Path) -> None:
    """Persist metrics as JSON with sorted keys."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, sort_keys=True)


def append_metrics_row(metrics_csv: Path, row: Dict) -> None:
    """
    Append a metrics row to CSV. Creates file with header on first write.

    Parameters
    ----------
    metrics_csv:
        Path to CSV file.
    row:
        Dictionary with serialisable values.
    """
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if metrics_csv.exists():
        df_row.to_csv(metrics_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(metrics_csv, index=False)


def save_predictions_csv(
    out_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[Iterable[str]] = None,
) -> None:
    """Persist predictions with optional probability columns."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: Dict[str, Iterable] = {
        "index": np.arange(y_true.shape[0]),
        "label": y_true,
        "prediction": y_pred,
    }
    if y_prob.size > 0:
        prob_df = pd.DataFrame(y_prob)
        if class_names:
            prob_df.columns = [f"prob_{name}" for name in class_names]
        else:
            prob_df.columns = [f"prob_{i}" for i in range(prob_df.shape[1])]
        records.update(prob_df.to_dict(orient="list"))
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: Path,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    annot_fontsize: int = 12,
) -> None:
    """
    Plot and save confusion matrix heatmap.

    Parameters
    ----------
    normalize:
        When True, rows are normalised to probabilities.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize and cm.sum(axis=1).all():
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"fontsize": annot_fontsize},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.75},
    )
    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    title_suffix = " (normalised)" if normalize else ""
    plt.title(f"Confusion Matrix{title_suffix}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_training_curves(
    history: List[Dict],
    model_name: str,
    out_dir: Path,
) -> None:
    """
    Plot training vs validation loss/accuracy curves from history rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    df_numeric = df[df["epoch"].apply(lambda x: isinstance(x, (int, float)))]
    if df_numeric.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df_numeric["epoch"], df_numeric["train_loss"], label="Train Loss")
    if "val_loss" in df_numeric:
        axes[0].plot(df_numeric["epoch"], df_numeric["val_loss"], label="Val Loss")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(df_numeric["epoch"], df_numeric["train_acc"], label="Train Acc")
    if "val_acc" in df_numeric:
        axes[1].plot(df_numeric["epoch"], df_numeric["val_acc"], label="Val Acc")
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name.lower().replace(' ', '_')}_training_curves.png", dpi=300)
    plt.close(fig)


def save_metric_barplot(
    metrics: Dict[str, float],
    out_path: Path,
    metrics_to_plot: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> None:
    """Save a horizontal bar plot for selected metrics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_to_plot is None:
        metrics_to_plot = [
            "acc",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "auc_macro",
        ]

    plot_data = {k: metrics[k] for k in metrics_to_plot if k in metrics}
    if not plot_data:
        return

    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(plot_data.values()), y=list(plot_data.keys()), orient="h")
    plt.xlim(0, 1)
    if title:
        plt.title(title)
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
