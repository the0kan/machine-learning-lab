import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def main():
    set_seeds()

    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )

    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=SEED,
        ),
    }

    rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        rows.append(
            {
                "model": model_name,
                "test_accuracy": f"{accuracy:.4f}",
            }
        )

        print(f"{model_name}: {accuracy:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "baseline_accuracy.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "test_accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
