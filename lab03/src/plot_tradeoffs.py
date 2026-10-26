import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

MODEL_COLORS = {
    "MLP": "#377eb8",
    "CNN": "#4daf4a",
    "Attention": "#e41a1c",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def build_summary() -> list[dict[str, object]]:
    accuracy_rows = {
        row["model"]: row
        for row in read_csv(RESULTS_DIR / "test_accuracy.csv")
    }
    cost_rows = {
        row["model"]: row
        for row in read_csv(RESULTS_DIR / "cost_analysis.csv")
    }
    latency_rows = read_csv(RESULTS_DIR / "latency.csv")
    latency_lookup = {
        (row["model"], int(row["batch_size"])): float(row["median_ms"])
        for row in latency_rows
    }

    summary_rows = []
    for model_name in ["MLP", "CNN", "Attention"]:
        summary_rows.append(
            {
                "model": model_name,
                "test_accuracy": float(
                    accuracy_rows[model_name]["test_accuracy"]
                ),
                "parameters": int(
                    cost_rows[model_name]["torchinfo_parameters"]
                ),
                "hand_flops": int(cost_rows[model_name]["hand_flops"]),
                "batch_1_latency_ms": latency_lookup[(model_name, 1)],
                "batch_128_latency_ms": latency_lookup[(model_name, 128)],
            }
        )

    return summary_rows


def save_summary(rows: list[dict[str, object]]) -> None:
    with (RESULTS_DIR / "tradeoff_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def annotate_points(
    axes: plt.Axes,
    rows: list[dict[str, object]],
    x_key: str,
) -> None:
    for row in rows:
        model_name = str(row["model"])
        x_value = float(row[x_key])
        accuracy_percent = float(row["test_accuracy"]) * 100
        axes.scatter(
            x_value,
            accuracy_percent,
            s=90,
            color=MODEL_COLORS[model_name],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        axes.annotate(
            f"{model_name}\n{accuracy_percent:.2f}%",
            (x_value, accuracy_percent),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )


def plot_accuracy_vs_parameters(rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(figsize=(8, 5))
    annotate_points(axes, rows, "parameters")
    axes.set_xlabel("Trainable Parameters")
    axes.set_ylabel("Test Accuracy (%)")
    axes.set_title("Fashion-MNIST Accuracy vs Parameter Count")
    axes.grid(True, alpha=0.3)
    axes.set_ylim(84.5, 91.0)
    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / "accuracy_vs_parameters.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_accuracy_vs_latency(rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for axis, batch_size, key in [
        (axes[0], 1, "batch_1_latency_ms"),
        (axes[1], 128, "batch_128_latency_ms"),
    ]:
        annotate_points(axis, rows, key)
        axis.set_xscale("log")
        axis.set_xlabel("Median Inference Latency (ms, log scale)")
        axis.set_title(f"Batch Size {batch_size}")
        axis.grid(True, which="both", alpha=0.3)
        axis.set_ylim(84.5, 91.0)

    axes[0].set_ylabel("Test Accuracy (%)")
    figure.suptitle("Fashion-MNIST Accuracy vs CPU Inference Latency")
    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / "accuracy_vs_latency.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    rows = build_summary()
    save_summary(rows)
    plot_accuracy_vs_parameters(rows)
    plot_accuracy_vs_latency(rows)

    for row in rows:
        print(
            f"{row['model']}: accuracy={float(row['test_accuracy']) * 100:.2f}%, "
            f"parameters={int(row['parameters']):,}, "
            f"batch-1 latency={float(row['batch_1_latency_ms']):.4f} ms, "
            f"batch-128 latency={float(row['batch_128_latency_ms']):.4f} ms"
        )
    print(f"Saved trade-off plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
