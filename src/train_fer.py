"""
Training script for FER-2013 emotion classifier using ResNet18.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from src.config import FER_LABELS, get_device, ensure_directories
from src.fer_dataset import FERDataset


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create ResNet18 model with modified final layer.

    Args:
        num_classes: Number of emotion classes
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        ResNet18 model
    """
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)

    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label_idx"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Validate for one epoch.

    Returns:
        (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label_idx"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 on FER dataset")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/archive",
        help="Path to data root directory (containing train/ and test/ folders)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts",
        help="Output directory for model checkpoint"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    ensure_directories()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Datasets
    print("Loading datasets...")
    train_dataset = FERDataset(args.data_root, split="train")
    val_dataset = FERDataset(args.data_root, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Model
    print("Creating model...")
    model = create_model(num_classes=len(FER_LABELS), pretrained=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0.0
    best_model_path = out_dir / "best_fer_resnet.pt"

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "labels": FER_LABELS,
            }, best_model_path)
            print(f"Saved best model with val acc: {val_acc:.2f}%")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
