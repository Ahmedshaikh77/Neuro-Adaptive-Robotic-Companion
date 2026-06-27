"""
Train the on-device keyword-spotting model (audio modality) on Google Speech
Commands, and save a checkpoint that src/audio_kws.py can load directly.

Speech Commands downloads automatically via torchaudio (~2.3 GB the first time).
Features use the SAME log-mel function as inference (src.audio_kws.log_mel), so
train-time and run-time features match.

Example:
  python -m src.train_kws --epochs 20 --batch-size 128 --out artifacts/kws.pt
"""
from __future__ import annotations
import argparse, os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import get_device, ensure_directories
from src.audio_kws import build_kws_model, log_mel, DEFAULT_KEYWORDS

TARGET = [k for k in DEFAULT_KEYWORDS if not k.startswith("_")]
LABELS = TARGET + ["_unknown_"]
SR = 16000
N_MELS = 40
FIXED_FRAMES = 101  # ~1 s at hop 160


def pad_or_fix(feat: np.ndarray, frames: int = FIXED_FRAMES) -> np.ndarray:
    if feat.shape[1] < frames:
        feat = np.pad(feat, ((0, 0), (0, frames - feat.shape[1])))
    return feat[:, :frames]


class SCDataset(Dataset):
    def __init__(self, subset: str, root: str = "data"):
        import torchaudio
        ensure_directories()
        os.makedirs(root, exist_ok=True)
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True, subset=subset)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        wav, sr, label, *_ = self.ds[i]
        wav = wav.squeeze(0).numpy().astype(np.float32)
        feat = pad_or_fix(log_mel(wav, sr=sr, n_mels=N_MELS))
        idx = LABELS.index(label) if label in TARGET else LABELS.index("_unknown_")
        return torch.from_numpy(feat).unsqueeze(0), idx


def run_epoch(model, loader, device, crit, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot = correct = 0
    loss_sum = 0.0
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
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="artifacts/kws.pt")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device(); print("device:", device)
    tr = DataLoader(SCDataset("training", args.data_root), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers)
    va = DataLoader(SCDataset("validation", args.data_root), batch_size=args.batch_size,
                    num_workers=args.num_workers)
    model = build_kws_model(len(LABELS), n_mels=N_MELS).to(device)
    crit = nn.CrossEntropyLoss(); opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(args.epochs):
        tl, ta = run_epoch(model, tr, device, crit, opt)
        vl, vacc = run_epoch(model, va, device, crit)
        print(f"epoch {ep+1}/{args.epochs}  train {ta:.1f}%  val {vacc:.1f}%")
        if vacc > best:
            best = vacc
            torch.save({"model_state_dict": model.state_dict(), "keywords": LABELS,
                        "val_acc": vacc}, args.out)
            print(f"  saved {args.out} (val {vacc:.1f}%)")
    print(f"done. best val acc {best:.1f}%. checkpoint: {args.out}")


if __name__ == "__main__":
    main()
