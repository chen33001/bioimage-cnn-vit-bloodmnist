import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from medmnist import INFO
from torch.cuda.amp import GradScaler, autocast
from torchvision import models

from src.data_loader import DEFAULT_IMAGE_SIZE, get_dataloaders, set_seed
from src.eval_utils import (
    EvaluationDetails,
    evaluate_model,
    plot_training_curves,
    save_confusion_matrix,
    save_metric_barplot,
    save_metrics_json,
    save_predictions_csv,
)


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch.lower() == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif arch.lower() == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported CNN architecture: {arch}")


def train_one_epoch(model, loader, device, criterion, optimizer, scaler: GradScaler, use_amp: bool):
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    for x, y in loader:
        x = x.to(device)
        y = y.squeeze().long().to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())
    labels_arr = np.asarray(all_labels)
    preds_arr = np.asarray(all_preds)
    acc = float((preds_arr == labels_arr).mean()) if labels_arr.size > 0 else 0.0
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train CNN on BloodMNIST")
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0"], help="CNN architecture")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models/cnn")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader, val_loader, test_loader, n_classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augment=True,
        download=True,
        pin_memory=pin_memory,
    )
    class_names = list(INFO["bloodmnist"]["label"].values())

    model = build_model(args.arch, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val = -np.inf
    metrics_rows: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler, use_amp)
        val_details: EvaluationDetails = evaluate_model(model, val_loader, device, criterion)
        val_metrics = val_details.metrics
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        metrics_rows.append(row)
        val_score = val_metrics["f1_macro"]
        if val_score > best_val:
            best_val = val_score
            ckpt_path = Path(args.models_dir) / f"{args.arch}_best.pt"
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_f1_macro": best_val}, ckpt_path)

        print(
            f"Epoch {epoch}/{args.epochs} - Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"Val f1_macro {val_metrics['f1_macro']:.4f} acc {val_metrics['acc']:.4f}"
        )

    # Test evaluation using best weights (if saved)
    ckpt_path = Path(args.models_dir) / f"{args.arch}_best.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])

    val_details = evaluate_model(model, val_loader, device, criterion, use_amp=use_amp)
    test_details = evaluate_model(model, test_loader, device, criterion, use_amp=use_amp)

    metrics_rows.append(
        {"epoch": "val_best", **{f"val_{k}": v for k, v in val_details.metrics.items()}}
    )
    metrics_rows.append(
        {"epoch": "test", **{f"test_{k}": v for k, v in test_details.metrics.items()}}
    )

    # Save metrics CSV
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    out_csv = results_dir / "cnn_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(out_csv, index=False)
    print(f"Saved metrics: {out_csv}")

    save_metrics_json(val_details.metrics, results_dir / "cnn_val_metrics.json")
    save_metrics_json(test_details.metrics, results_dir / "cnn_test_metrics.json")
    save_predictions_csv(
        results_dir / "cnn_val_predictions.csv",
        val_details.y_true,
        val_details.y_pred,
        val_details.y_prob,
        class_names=class_names,
    )
    save_predictions_csv(
        results_dir / "cnn_test_predictions.csv",
        test_details.y_true,
        test_details.y_pred,
        test_details.y_prob,
        class_names=class_names,
    )
    save_confusion_matrix(
        val_details.y_true,
        val_details.y_pred,
        class_names=class_names,
        out_path=figures_dir / "cnn_confusion_val.png",
        normalize=True,
    )
    save_confusion_matrix(
        test_details.y_true,
        test_details.y_pred,
        class_names=class_names,
        out_path=figures_dir / "cnn_confusion_test.png",
        normalize=True,
    )
    plot_training_curves(metrics_rows, f"CNN ({args.arch})", figures_dir)
    save_metric_barplot(
        val_details.metrics,
        figures_dir / "cnn_val_metrics_bar.png",
        title=f"{args.arch.upper()} Validation Metrics",
    )
    save_metric_barplot(
        test_details.metrics,
        figures_dir / "cnn_test_metrics_bar.png",
        title=f"{args.arch.upper()} Test Metrics",
    )


if __name__ == "__main__":
    main()
