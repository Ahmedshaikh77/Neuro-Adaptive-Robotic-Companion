"""
Train the gesture model on a Jester-style dataset and save a checkpoint that
src/gesture_engine.py can load directly.

Expected folder layout (arrange Jester, or a subset, into this):
    data_gesture/
        train/<gesture_name>/<sample_id>/frame_0001.jpg ...
        test/<gesture_name>/<sample_id>/frame_0001.jpg ...
Each <sample_id> folder holds the ordered frames of one short clip.

NOTE ON DATA: the 20BN-Jester set has required registration and has moved hosts.
If you cannot obtain it, point --data-root at any folder-structured gesture set
in the same layout, or reduce --gestures to a subset. The training code does not
care which gestures, only that the folder layout matches.

Example:
  python -m src.train_gesture --data-root data_gesture --epochs 15 --out artifacts/gesture.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import get_device
from src.gesture_engine import build_gesture_model, DEFAULT_GESTURES


def sample_frames(folder: Path, n_frames: int, size: int) -> np.ndarray:
    frames = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
    if not frames:
        raise ValueError(f"no frames in {folder}")
    idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
    stack = []
    for j in idx:
        img = Image.open(frames[j]).convert("L").resize((size, size))
        stack.append(np.asarray(img, dtype=np.float32))
    return np.stack(stack, 0) / 255.0  # (T, H, W)


class GestureDataset(Dataset):
    def __init__(self, root, split, gestures, n_frames=8, size=96):
        self.n_frames, self.size = n_frames, size
        self.gestures = gestures
        self.samples = []
        base = Path(root) / split
        for gi, g in enumerate(gestures):
            gdir = base / g
            if not gdir.exists():
                print(f"warning: missing class dir {gdir}")
                continue
            for sample in sorted(p for p in gdir.iterdir() if p.is_dir()):
                self.samples.append((sample, gi))
        if not self.samples:
            raise ValueError(f"no samples under {base}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        folder, gi = self.samples[i]
        x = sample_frames(folder, self.n_frames, self.size)
        return torch.from_numpy(x), gi


def run_epoch(model, loader, device, crit, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot = correct = 0; loss_sum = 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item(); tot += x.size(0)
    return loss_sum / tot, 100.0 * correct / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data_gesture")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--gestures", nargs="*", default=DEFAULT_GESTURES)
    ap.add_argument("--out", default="artifacts/gesture.pt")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device(); print("device:", device)
    tr = DataLoader(GestureDataset(args.data_root, "train", args.gestures, args.n_frames, args.size),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    te = DataLoader(GestureDataset(args.data_root, "test", args.gestures, args.n_frames, args.size),
                    batch_size=args.batch_size, num_workers=args.num_workers)
    model = build_gesture_model(len(args.gestures), n_frames=args.n_frames).to(device)
    crit = nn.CrossEntropyLoss(); opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(args.epochs):
        tl, ta = run_epoch(model, tr, device, crit, opt)
        vl, vacc = run_epoch(model, te, device, crit)
        print(f"epoch {ep+1}/{args.epochs}  train {ta:.1f}%  test {vacc:.1f}%")
        if vacc > best:
            best = vacc
            torch.save({"model_state_dict": model.state_dict(), "gestures": args.gestures,
                        "n_frames": args.n_frames, "test_acc": vacc}, args.out)
            print(f"  saved {args.out} (test {vacc:.1f}%)")
    print(f"done. best test acc {best:.1f}%. checkpoint: {args.out}")


if __name__ == "__main__":
    main()
