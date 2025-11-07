import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")

MODEL_KEYS = ("cnn", "vit")
SPLITS = ("val", "test")
METRICS_OF_INTEREST = [
    "acc",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "auc_macro",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "auc_micro",
    "loss",
]
SUMMARY_METRICS = [m for m in METRICS_OF_INTEREST if m != "loss"]
PLOT_METRICS = SUMMARY_METRICS


def load_metrics(results_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load metrics JSON artefacts into nested dict model -> split -> metrics."""
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in MODEL_KEYS:
        data[model] = {}
        for split in SPLITS:
            json_path = results_dir / f"{model}_{split}_metrics.json"
            if json_path.exists():
                with json_path.open("r", encoding="utf-8") as fp:
                    data[model][split] = json.load(fp)
            else:
                data[model][split] = {}
    return data


def to_dataframe(
    metrics_dict: Dict[str, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    """Convert nested metrics dict to tidy dataframe."""
    records: List[Dict[str, float]] = []
    for model, splits in metrics_dict.items():
        for split, metrics in splits.items():
            if not metrics:
                continue
            record = {"model": model, "split": split}
            record.update({metric: value for metric, value in metrics.items() if metric in METRICS_OF_INTEREST})
            records.append(record)
    return pd.DataFrame.from_records(records)


def plot_metric_comparison(df: pd.DataFrame, figures_dir: Path) -> None:
    """Create grouped bar chart comparing CNN vs ViT across metrics."""
    if df.empty:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    melted = df.melt(id_vars=["model", "split"], value_vars=PLOT_METRICS, var_name="metric", value_name="value")

    for split in SPLITS:
        split_df = melted[melted["split"] == split]
        if split_df.empty:
            continue

        plt.figure(figsize=(10, 5))
        sns.barplot(data=split_df, x="metric", y="value", hue="model")
        plt.ylim(0, 1)
        plt.title(f"CNN vs ViT Metrics ({split.title()} split)")
        plt.ylabel("Score")
        plt.xlabel("Metric")
        plt.tight_layout()
        plt.savefig(figures_dir / f"comparison_{split}_metrics.png", dpi=300)
        plt.close()


def write_summary(df: pd.DataFrame, out_path: Path, mode_label: str) -> None:
    """
    Generate a lightweight narrative summary highlighting stronger model per metric.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        out_path.write_text("No metrics available to summarise.\n", encoding="utf-8")
        return

    lines = ["# CNN vs ViT Summary", f"_Mode: {mode_label}_", ""]
    for split in SPLITS:
        split_df = df[df["split"] == split]
        if split_df.empty:
            continue
        lines.append(f"## {split.title()} Split")
        for metric in SUMMARY_METRICS:
            if metric not in split_df:
                continue
            best_row = split_df.sort_values(metric, ascending=False).iloc[0]
            lines.append(
                f"- **{metric}**: {best_row['model'].upper()} leads with {best_row[metric]:.4f}"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CNN vs ViT results.")
    parser.add_argument("--results-dir", default="results", help="Directory containing metrics JSON artefacts.")
    parser.add_argument("--figures-dir", default="figures", help="Directory to store generated plots.")
    parser.add_argument(
        "--output-csv",
        default="results/final_comparison.csv",
        help="Path to write combined metrics CSV.",
    )
    parser.add_argument(
        "--summary-path",
        default="results/final_summary.md",
        help="Path to write a markdown summary.",
    )
    parser.add_argument(
        "--mode-label",
        default="full",
        help="Label describing the run mode (e.g., full, reduced).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    metrics = load_metrics(results_dir)
    df = to_dataframe(metrics)
    if df.empty:
        print("No metrics found to aggregate.")
        return

    df.sort_values(["split", "model"], inplace=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote combined metrics to {output_csv}")

    plot_metric_comparison(df, figures_dir)
    write_summary(df, Path(args.summary_path), mode_label=args.mode_label)


if __name__ == "__main__":
    main()
