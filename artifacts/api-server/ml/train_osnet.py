from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import DEVICE, OSNET_WEIGHTS
from datasets import PARSERS
from hashing import sha256_file
from osnet import OSNet, load_osnet
from preprocess import decode_image, osnet_tensor


class ReIdFolder(Dataset):
    def __init__(self, samples, identity_to_index: dict[str, int]):
        self.samples = samples
        self.identity_to_index = identity_to_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = decode_image(Path(sample.path).read_bytes())
        return osnet_tensor(image).squeeze(0), self.identity_to_index[sample.identity_id]


def resource_check(batch_size: int, sample_count: int, output: Path) -> list[str]:
    warnings: list[str] = []
    disk = shutil.disk_usage(output.parent if output.parent.exists() else Path("."))
    if disk.free < 512 * 1024 * 1024:
        warnings.append("Less than 512 MB disk is free for checkpoints.")
    if batch_size > 16 and DEVICE == "cpu":
        warnings.append("CPU training with batch size > 16 is likely to be slow or memory-heavy.")
    estimated = sample_count * 256 * 128 * 3 * 4
    if estimated > 2 * 1024 * 1024 * 1024:
        warnings.append("Dataset tensors may not fit in memory; training streams from disk per batch.")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune OSNet on an authorized local Re-ID dataset.")
    parser.add_argument("--dataset", required=True, choices=sorted(PARSERS))
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="data/training/osnet-finetune.pth")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    report = PARSERS[args.dataset](Path(args.root))
    if not report.layout_ok:
        raise SystemExit("; ".join(report.errors) or "Dataset layout is invalid.")
    train_samples = [sample for sample in report.samples if sample.split == "train"][: args.max_train]
    val_samples = [sample for sample in report.samples if sample.split in {"query", "gallery", "test"}][: max(32, args.max_train // 8)]
    if not train_samples:
        raise SystemExit("No training samples were parsed. Test identities were not used.")
    identities = sorted({sample.identity_id for sample in train_samples})
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    warnings = resource_check(args.batch_size, len(train_samples), output)
    for warning in warnings:
        print(json.dumps({"warning": warning}))

    if OSNET_WEIGHTS.is_file():
        model = load_osnet(OSNET_WEIGHTS, DEVICE)
        model.classifier = nn.Linear(model.feature_dim, len(identities))
        model.classifier.to(DEVICE)
    else:
        model = OSNet(num_classes=len(identities)).to(DEVICE)
    model.train()
    loader = DataLoader(
        ReIdFolder(train_samples, identity_to_index),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    history = []
    start = datetime.now(timezone.utc)
    for epoch in range(args.epochs):
        running = 0.0
        count = 0
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * images.size(0)
            count += images.size(0)
        epoch_loss = running / max(count, 1)
        history.append({"epoch": epoch + 1, "trainLoss": epoch_loss})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"state_dict": model.state_dict(), "identities": identities}, output)
    payload = {
        "status": "completed",
        "dataset": args.dataset,
        "root": args.root,
        "startedAt": start.isoformat(),
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "epochs": args.epochs,
        "batchSize": args.batch_size,
        "seed": args.seed,
        "device": DEVICE,
        "startingCheckpointSha256": sha256_file(OSNET_WEIGHTS),
        "selectedCheckpoint": str(output.resolve()),
        "selectedCheckpointSha256": sha256_file(output),
        "trainIdentities": len(identities),
        "trainSamples": len(train_samples),
        "heldOutSamples": len(val_samples),
        "history": history,
        "warnings": warnings,
        "activation": "not_activated",
        "note": "Training finished without changing the active production model.",
    }
    log_path = output.with_suffix(".json")
    log_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
