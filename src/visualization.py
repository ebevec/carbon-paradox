from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("out")

def plot_predictions(predictions_file, title, save_path=None):
    df = pd.read_csv(predictions_file)
    plt.plot(df["actual"].values, label="Actual")
    plt.plot(df["prediction"].values, label="Prediction")
    plt.title(title)
    plt.xlabel("Observation")
    plt.ylabel("Carbon intensity")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_model_comparison(save_path=None):
    baseline = pd.read_csv(OUT_DIR / "baseline_metrics.csv")
    gnn = pd.read_csv(OUT_DIR / "gnn_metrics.csv")
    comparison = baseline.merge(gnn, on="metric", suffixes=("_baseline", "_gnn"))

    x = range(len(comparison))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        comparison["value_baseline"],
        width=width,
        label="Baseline",
    )
    plt.bar(
        [i + width / 2 for i in x],
        comparison["value_gnn"],
        width=width,
        label="GNN",
    )
    plt.xticks(list(x), comparison["metric"])
    plt.ylabel("Value")
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_codecarbon(file_path, title, save_path=None):
    df = pd.read_csv(file_path)
    plt.figure(figsize=(6, 4))
    plt.bar([title], [df["emissions"].sum()])
    plt.ylabel("kg CO2eq")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
