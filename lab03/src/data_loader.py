from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 128
TRAIN_SIZE = 55_000
VALIDATION_SIZE = 5_000

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def create_dataloaders(
    data_dir: Path = DEFAULT_DATA_DIR,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    full_train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    split_generator = torch.Generator().manual_seed(SEED)
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [TRAIN_SIZE, VALIDATION_SIZE],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    train_loader, validation_loader, test_loader = create_dataloaders()
    images, labels = next(iter(train_loader))

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Image batch shape: {tuple(images.shape)}")
    print(f"Label batch shape: {tuple(labels.shape)}")
    print(f"Image dtype: {images.dtype}")
