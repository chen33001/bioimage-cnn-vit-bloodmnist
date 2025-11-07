import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from medmnist import INFO
except Exception:  # pragma: no cover
    INFO = None

from src.eval_utils import save_confusion_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate confusion matrices from saved predictions.")
    parser.add_argument("--model", choices=["cnn", "vit"], required=True, help="Model key used in file naming.")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Dataset split to visualise.")
    parser.add_argument("--results-dir", default="results", help="Directory containing *_predictions.csv files.")
    parser.add_argument("--figures-dir", default="figures", help="Directory to write confusion matrices.")
    parser.add_argument("--normalize", action="store_true", help="Normalise rows to probabilities.")
    return parser


def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "label" not in df or "prediction" not in df:
        raise ValueError(f"{csv_path} is missing required columns 'label'/'prediction'")
    return df["label"].to_numpy(), df["prediction"].to_numpy()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"{args.model}_{args.split}_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {csv_path}. Run scripts/run_all.ps1 first.")

    y_true, y_pred = load_predictions(csv_path)
    if INFO is not None and "bloodmnist" in INFO:
        class_names = list(INFO["bloodmnist"]["label"].values())
    else:  # fallback
        class_names = [f"class_{i}" for i in range(len(np.unique(y_true)))]
    out_path = figures_dir / f"{args.model}_confusion_{args.split}.png"
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_path=out_path,
        normalize=args.normalize,
        figsize=(12, 9),
        cmap="BuPu",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
