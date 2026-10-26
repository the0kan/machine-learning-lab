import csv
from pathlib import Path

import torch
from thop import profile
from torchinfo import summary

from models import create_models, trainable_parameter_count


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def linear_flops(input_features: int, output_features: int) -> int:
    return 2 * input_features * output_features


def convolution_flops(
    kernel_size: int,
    input_channels: int,
    output_channels: int,
    output_height: int,
    output_width: int,
) -> int:
    return (
        2
        * kernel_size
        * kernel_size
        * input_channels
        * output_channels
        * output_height
        * output_width
    )


def attention_encoder_layer_flops(
    sequence_length: int,
    model_dimension: int,
    feedforward_dimension: int,
) -> dict[str, int]:
    qkv_projections = (
        2 * sequence_length * model_dimension * 3 * model_dimension
    )
    attention_scores = (
        2 * sequence_length * sequence_length * model_dimension
    )
    attention_weighted_values = (
        2 * sequence_length * sequence_length * model_dimension
    )
    output_projection = (
        2 * sequence_length * model_dimension * model_dimension
    )
    feedforward_network = (
        2 * sequence_length * model_dimension * feedforward_dimension
        + 2 * sequence_length * feedforward_dimension * model_dimension
    )

    return {
        "QKV projections": qkv_projections,
        "Attention scores": attention_scores,
        "Attention weighted values": attention_weighted_values,
        "Output projection": output_projection,
        "Feedforward network": feedforward_network,
    }


def calculate_hand_flops() -> tuple[dict[str, int], dict[str, int]]:
    mlp_flops = (
        linear_flops(784, 256)
        + linear_flops(256, 128)
        + linear_flops(128, 10)
    )

    cnn_flops = (
        convolution_flops(3, 1, 16, 28, 28)
        + convolution_flops(3, 16, 32, 14, 14)
        + linear_flops(32 * 7 * 7, 10)
    )

    attention_components = attention_encoder_layer_flops(
        sequence_length=16,
        model_dimension=64,
        feedforward_dimension=128,
    )
    patch_projection = 2 * 16 * 49 * 64
    classifier = linear_flops(64, 10)
    attention_flops = (
        patch_projection
        + 2 * sum(attention_components.values())
        + classifier
    )

    return (
        {
            "MLP": mlp_flops,
            "CNN": cnn_flops,
            "Attention": attention_flops,
        },
        attention_components,
    )


def main() -> None:
    torch.manual_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = create_models()
    sample = torch.zeros(1, 1, 28, 28)
    hand_flops, attention_components = calculate_hand_flops()

    rows = []
    summaries = []

    for name, model in models.items():
        model.eval()
        model_summary = summary(
            model,
            input_size=(1, 1, 28, 28),
            verbose=0,
        )
        thop_macs, thop_parameters = profile(
            model,
            inputs=(sample,),
            verbose=False,
        )

        parameter_count = trainable_parameter_count(model)
        thop_flops_estimate = int(2 * thop_macs)
        rows.append(
            {
                "model": name,
                "torchinfo_parameters": parameter_count,
                "thop_parameters": int(thop_parameters),
                "hand_flops": hand_flops[name],
                "thop_macs": int(thop_macs),
                "thop_macs_times_2": thop_flops_estimate,
            }
        )

        summaries.append(f"===== {name} =====\n{model_summary}\n")

    with (RESULTS_DIR / "cost_analysis.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    (RESULTS_DIR / "model_summaries.txt").write_text(
        "\n".join(summaries),
        encoding="utf-8",
    )

    attention_layer_total = sum(attention_components.values())
    calculation_lines = [
        "Representative hand-calculated FLOPs",
        "=====================================",
        "",
        "Linear layer (MLP 784 -> 256):",
        f"2 x 784 x 256 = {linear_flops(784, 256):,} FLOPs",
        "",
        "Conv2d layer (CNN first convolution):",
        "2 x 3 x 3 x 1 x 16 x 28 x 28 "
        f"= {convolution_flops(3, 1, 16, 28, 28):,} FLOPs",
        "",
        "Attention encoder layer (sequence=16, d_model=64, feedforward=128):",
    ]
    calculation_lines.extend(
        f"- {component}: {value:,} FLOPs"
        for component, value in attention_components.items()
    )
    calculation_lines.extend(
        [
            f"- One encoder layer total: {attention_layer_total:,} FLOPs",
            "",
            "Approximate full-model totals",
            "-----------------------------",
            *(f"- {name}: {value:,} FLOPs" for name, value in hand_flops.items()),
            "",
            "Counting convention and tool gap",
            "--------------------------------",
            "The hand calculation counts a multiply and an add as two FLOPs.",
            "THOP reports multiply-accumulate operations (MACs), so 2 x MACs",
            "matches the hand calculation for the supported MLP and CNN layers.",
            "THOP does not profile PyTorch TransformerEncoder internals by default.",
            "It therefore reports only the patch projection and classifier for the",
            "Attention model and also omits most of that model's parameters.",
            "Bias additions, activations, pooling, normalization, softmax, and",
            "positional additions are omitted from the representative hand totals.",
        ]
    )
    (RESULTS_DIR / "hand_flops.txt").write_text(
        "\n".join(calculation_lines) + "\n",
        encoding="utf-8",
    )

    for row in rows:
        print(
            f"{row['model']}: params={row['torchinfo_parameters']:,}, "
            f"hand FLOPs={row['hand_flops']:,}, "
            f"THOP MACs={row['thop_macs']:,}, "
            f"2x MACs={row['thop_macs_times_2']:,}"
        )
    print(f"Saved analysis to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
