import torch
from torch import nn
from torchinfo import summary


class MLPClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class CNNClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)


class AttentionClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_extractor = nn.Unfold(kernel_size=7, stride=7)
        self.patch_projection = nn.Linear(7 * 7, 64)
        self.position_embedding = nn.Parameter(torch.zeros(1, 16, 64))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_extractor(images).transpose(1, 2)
        tokens = self.patch_projection(patches)
        tokens = tokens + self.position_embedding
        encoded_tokens = self.encoder(tokens)
        pooled_tokens = encoded_tokens.mean(dim=1)
        return self.classifier(pooled_tokens)


def create_models() -> dict[str, nn.Module]:
    return {
        "MLP": MLPClassifier(),
        "CNN": CNNClassifier(),
        "Attention": AttentionClassifier(),
    }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    sample_batch = torch.zeros(2, 1, 28, 28)

    for name, model in create_models().items():
        model.eval()
        with torch.inference_mode():
            output = model(sample_batch)

        model_summary = summary(
            model,
            input_size=(1, 1, 28, 28),
            verbose=0,
        )

        print(f"\n{name}")
        print(f"Output shape: {tuple(output.shape)}")
        print(f"Trainable parameters: {trainable_parameter_count(model):,}")
        print(f"torchinfo parameters: {model_summary.total_params:,}")
