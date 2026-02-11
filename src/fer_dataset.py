"""
FER dataset loader for folder-based structure.
Supports train/val/test splits from directory layout.
"""

from typing import Dict, Optional, Literal
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.config import FER_LABELS


SplitType = Literal["train", "val", "test"]


class FERDataset(Dataset):
    """
    FER dataset loader for folder-based structure.

    Expected directory structure:
        data_root/
            train/
                angry/*.jpg
                disgust/*.jpg
                fear/*.jpg
                happy/*.jpg
                neutral/*.jpg
                sad/*.jpg
                surprise/*.jpg
            test/
                angry/*.jpg
                ...

    This implementation can be easily adapted for AffectNet by changing
    the root directory and ensuring the same folder structure.
    """

    def __init__(
        self,
        data_root: str,
        split: SplitType,
        val_fraction: float = 0.15,
        transform: Optional[transforms.Compose] = None,
        seed: int = 42,
    ):
        """
        Initialize FER dataset from folder structure.

        Args:
            data_root: Path to data root (e.g., "data/archive")
            split: One of "train", "val", "test"
            val_fraction: Fraction of train data to use for validation (0.0-1.0)
            transform: Optional torchvision transforms
            seed: Random seed for train/val split
        """
        self.data_root = Path(data_root)
        self.split = split
        self.val_fraction = val_fraction
        self.labels = FER_LABELS
        self.seed = seed

        # Build file list
        self.samples = self._build_sample_list()

        # Default transform: resize to 224x224, convert to tensor, ImageNet normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

    def _build_sample_list(self) -> list[tuple[Path, int, str]]:
        """
        Build list of (image_path, label_idx, label_name) tuples.

        Returns:
            List of samples
        """
        samples = []

        if self.split == "test":
            # Test split: use data_root/test/<class>/*.jpg
            split_dir = self.data_root / "test"
        else:
            # Train/val splits: use data_root/train/<class>/*.jpg
            split_dir = self.data_root / "train"

        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        # Iterate through each class folder
        for label_name in self.labels:
            class_dir = split_dir / label_name
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue

            # Get all image files
            image_files = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
            label_idx = self.labels.index(label_name)

            # For train/val split, we need to partition the train folder
            if self.split in ["train", "val"]:
                # Set seed for reproducible split
                random.seed(self.seed)
                random.shuffle(image_files)

                # Calculate split point
                n_val = int(len(image_files) * self.val_fraction)

                if self.split == "val":
                    # Take first n_val images for validation
                    image_files = image_files[:n_val]
                else:  # train
                    # Take remaining images for training
                    image_files = image_files[n_val:]

            # Add to samples list
            for img_path in image_files:
                samples.append((img_path, label_idx, label_name))

        if len(samples) == 0:
            raise ValueError(f"No samples found for split '{self.split}' in {split_dir}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single sample.

        Returns:
            Dictionary with keys:
                - image: (3, 224, 224) tensor
                - label_idx: int (0-6)
                - label_name: str
        """
        img_path, label_idx, label_name = self.samples[idx]

        # Load image as RGB
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label_idx": label_idx,
            "label_name": label_name,
        }


# For future AffectNet support, you would create:
# class AffectNetDataset(Dataset):
#     """
#     AffectNet dataset loader with the same interface.
#     """
#     def __init__(self, data_root: str, split: SplitType, transform: Optional = None):
#         # Use the same folder structure approach
#         # Just point to AffectNet root directory
#         pass
