"""
Evaluation script for FER-2013 emotion classifier.
Computes accuracy, per-class metrics, and confusion matrix.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.config import FER_LABELS, get_device, ensure_directories
from src.fer_dataset import FERDataset
from src.train_fer import create_model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """
    Evaluate model and return predictions and ground truth.

    Returns:
        (all_predictions, all_labels)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label_idx"].to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    """
    Plot and save confusion matrix as PNG.

    Args:
        cm: Confusion matrix array
        labels: Class label names
        output_path: Path to save PNG
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet18 on FER test set")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/archive",
        help="Path to data root directory (containing train/ and test/ folders)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/best_fer_resnet.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )

    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = FERDataset(args.data_root, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    print(f"Test samples: {len(test_dataset)}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = create_model(num_classes=len(FER_LABELS), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Evaluate
    print("\nEvaluating model...")
    all_preds, all_labels = evaluate_model(model, test_loader, device)

    # Compute metrics
    accuracy = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")

    # Classification report
    print("Per-Class Metrics:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=FER_LABELS,
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Save confusion matrix as .npy
    cm_npy_path = Path("artifacts") / "confusion_matrix.npy"
    np.save(cm_npy_path, cm)
    print(f"Confusion matrix saved to: {cm_npy_path}")

    # Plot and save confusion matrix as .png
    cm_png_path = Path("artifacts") / "confusion_matrix.png"
    plot_confusion_matrix(cm, FER_LABELS, cm_png_path)

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
