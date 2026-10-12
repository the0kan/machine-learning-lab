import csv
import statistics
import threading
import time
from pathlib import Path

import joblib
import psutil
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from train_baselines import SEED, build_models, set_seeds

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TRAINING_RUNS = 5
INFERENCE_RUNS = 100


class PeakMemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.peak_bytes = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _record(self):
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def _sample(self):
        while not self.stop_event.wait(0.001):
            self._record()

    def __enter__(self):
        self._record()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._record()
        self.stop_event.set()
        self.thread.join()

    @property
    def peak_mb(self):
        return self.peak_bytes / (1024**2)


def fit_clone(model, X_train, y_train):
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)
    return fitted_model


def main():
    set_seeds()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )
    single_sample = X_test[:1]
    rows = []

    for model_name, model in build_models().items():
        fit_clone(model, X_train, y_train)

        training_times_ms = []
        fitted_model = None
        for _ in range(TRAINING_RUNS):
            start = time.perf_counter_ns()
            fitted_model = fit_clone(model, X_train, y_train)
            end = time.perf_counter_ns()
            training_times_ms.append((end - start) / 1_000_000)

        with PeakMemoryMonitor() as training_memory:
            fit_clone(model, X_train, y_train)

        fitted_model.predict(single_sample)
        inference_times_ms = []
        for _ in range(INFERENCE_RUNS):
            start = time.perf_counter_ns()
            fitted_model.predict(single_sample)
            end = time.perf_counter_ns()
            inference_times_ms.append((end - start) / 1_000_000)

        with PeakMemoryMonitor() as inference_memory:
            for _ in range(INFERENCE_RUNS):
                fitted_model.predict(single_sample)

        model_file = RESULTS_DIR / f"{model_name}.joblib"
        joblib.dump(fitted_model, model_file)
        model_size_bytes = model_file.stat().st_size

        row = {
            "model": model_name,
            "training_time_median_ms": f"{statistics.median(training_times_ms):.6f}",
            "inference_time_median_ms": f"{statistics.median(inference_times_ms):.6f}",
            "model_size_bytes": model_size_bytes,
            "model_size_kb": f"{model_size_bytes / 1024:.3f}",
            "training_peak_memory_mb": f"{training_memory.peak_mb:.3f}",
            "inference_peak_memory_mb": f"{inference_memory.peak_mb:.3f}",
        }
        rows.append(row)
        print(row)

    output_file = RESULTS_DIR / "system_measurements.csv"
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
