"""Training and validation loops.

Every arm runs this identical code. The only thing that differs is the
`apply_masking` flag on the training dataset, which is what makes the comparison
a controlled experiment rather than two loosely related runs.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from landmarks.eval.gap import global_average_precision, predictions_from_logits
from landmarks.utils.io import ensure_dir


def cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to zero.

    Warmup matters here: ArcFace logits are large at initialisation, and a cold
    start at full learning rate frequently diverges in the first few hundred
    steps.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    grad_clip: float = 5.0,
    epoch: int = 0,
) -> float:
    model.train()
    running, n = 0.0, 0

    bar = tqdm(loader, desc=f"train {epoch}", leave=False)
    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=scaler.is_enabled()):
            logits = model(images, labels)          # labels drive the ArcFace margin
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running += loss.item() * labels.size(0)
        n += labels.size(0)
        bar.set_postfix(loss=f"{running / max(1, n):.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return running / max(1, n)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, desc: str = "eval"):
    """Return (labels, logits) for a whole loader, as float32 numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            # No labels: the head returns plain scaled cosine logits.
            logits = model(images)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_logits)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, desc: str = "eval") -> dict:
    labels, logits = predict(model, loader, device, desc)
    preds, confs = predictions_from_logits(logits)
    return {
        "gap": global_average_precision(labels, preds, confs),
        "top1": float((preds == labels).mean()),
        "n": int(len(labels)),
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg,
    device: torch.device,
    run_dir: str | Path,
) -> dict:
    """Train one arm, selecting the checkpoint by validation GAP.

    GAP rather than accuracy for model selection, because GAP is the reported
    metric and rewards calibrated confidence -- selecting on accuracy would
    optimise something the study does not measure.
    """
    run_dir = ensure_dir(run_dir)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    steps_per_epoch = len(train_loader)
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=cfg.train.warmup_epochs * steps_per_epoch,
        total_steps=cfg.train.epochs * steps_per_epoch,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and torch.cuda.is_available())

    history: list[dict] = []
    best_gap, best_epoch, since_improved = -1.0, -1, 0

    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, cfg.train.grad_clip, epoch,
        )
        metrics = evaluate(model, val_loader, device, desc=f"val {epoch}")
        elapsed = time.time() - t0

        row = {"epoch": epoch, "train_loss": loss, "val_gap": metrics["gap"],
               "val_top1": metrics["top1"], "seconds": round(elapsed, 1)}
        history.append(row)
        print(f"epoch {epoch:2d} | loss {loss:.4f} | val GAP {metrics['gap']:.4f} "
              f"| val top1 {metrics['top1']:.4f} | {elapsed:.0f}s")

        if metrics["gap"] > best_gap:
            best_gap, best_epoch, since_improved = metrics["gap"], epoch, 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch,
                 "val_gap": best_gap, "arm": cfg.arm},
                Path(run_dir) / "best.pt",
            )
        else:
            since_improved += 1
            if since_improved >= cfg.train.early_stop_patience:
                print(f"early stop: no val GAP improvement for "
                      f"{cfg.train.early_stop_patience} epochs")
                break

        with open(Path(run_dir) / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nbest val GAP {best_gap:.4f} at epoch {best_epoch}")
    return {"best_gap": best_gap, "best_epoch": best_epoch, "history": history}
