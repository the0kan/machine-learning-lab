import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_loader import create_dataloaders
from models import create_models, trainable_parameter_count


SEED = 42
EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = torch.device("cpu")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    set_seeds()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = create_models()
    criterion = nn.CrossEntropyLoss()
    history_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    for model_name, model in models.items():
        train_loader, validation_loader, test_loader = create_dataloaders()
        model = model.to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

        print(f"\nTraining {model_name} on {DEVICE}")
        model_start = time.perf_counter()

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            train_loss, train_accuracy = run_training_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
            validation_loss, validation_accuracy = evaluate(
                model,
                validation_loader,
                criterion,
            )
            epoch_seconds = time.perf_counter() - epoch_start

            history_rows.append(
                {
                    "model": model_name,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "epoch_seconds": epoch_seconds,
                }
            )
            save_rows(RESULTS_DIR / "training_history.csv", history_rows)

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"train loss={train_loss:.4f}, "
                f"train acc={train_accuracy * 100:.2f}% | "
                f"val loss={validation_loss:.4f}, "
                f"val acc={validation_accuracy * 100:.2f}% | "
                f"{epoch_seconds:.2f}s"
            )

        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        training_seconds = time.perf_counter() - model_start
        test_rows.append(
            {
                "model": model_name,
                "parameters": trainable_parameter_count(model),
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "training_seconds": training_seconds,
                "device": str(DEVICE),
            }
        )
        save_rows(RESULTS_DIR / "test_accuracy.csv", test_rows)

        print(
            f"{model_name} final test accuracy: "
            f"{test_accuracy * 100:.2f}% "
            f"(test loss={test_loss:.4f})"
        )

    print(f"\nSaved training evidence to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
