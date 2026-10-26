import csv
import platform
import statistics
import time
from pathlib import Path

import torch

from models import create_models


SEED = 42
WARMUP_RUNS = 10
DEVICE = torch.device("cpu")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def measure_latency(
    model: torch.nn.Module,
    batch_size: int,
    measured_runs: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(SEED + batch_size)
    inputs = torch.randn(
        batch_size,
        1,
        28,
        28,
        generator=generator,
        device=DEVICE,
    )

    model = model.to(DEVICE)
    model.eval()

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(inputs)

        durations_ms = []
        for _ in range(measured_runs):
            start = time.perf_counter()
            model(inputs)
            end = time.perf_counter()
            durations_ms.append((end - start) * 1_000)

    return {
        "median_ms": statistics.median(durations_ms),
        "minimum_ms": min(durations_ms),
        "maximum_ms": max(durations_ms),
    }


def main() -> None:
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_cases = [
        (1, 100),
        (128, 20),
    ]
    rows = []

    for model_name, model in create_models().items():
        for batch_size, measured_runs in benchmark_cases:
            measurements = measure_latency(
                model=model,
                batch_size=batch_size,
                measured_runs=measured_runs,
            )
            rows.append(
                {
                    "model": model_name,
                    "device": str(DEVICE),
                    "batch_size": batch_size,
                    "warmup_runs": WARMUP_RUNS,
                    "measured_runs": measured_runs,
                    **measurements,
                }
            )

    with (RESULTS_DIR / "latency.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    environment_lines = [
        f"Platform: {platform.platform()}",
        f"PyTorch: {torch.__version__}",
        f"Device: {DEVICE}",
        f"PyTorch CPU threads: {torch.get_num_threads()}",
        f"Warm-up runs per case: {WARMUP_RUNS}",
        "Batch 1 measured runs: 100",
        "Batch 128 measured runs: 20",
    ]
    (RESULTS_DIR / "latency_environment.txt").write_text(
        "\n".join(environment_lines) + "\n",
        encoding="utf-8",
    )

    for row in rows:
        print(
            f"{row['model']}, batch={row['batch_size']}: "
            f"median={row['median_ms']:.4f} ms "
            f"(min={row['minimum_ms']:.4f}, max={row['maximum_ms']:.4f})"
        )

    print(f"Saved latency evidence to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
