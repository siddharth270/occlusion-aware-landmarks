"""Step 4: train one experimental arm.

    python scripts/train.py --arm baseline
    python scripts/train.py --arm masked
    python scripts/train.py --arm maskaug        # optional third arm

Every arm shares this entry point, this config, and this seed. The only thing
that varies is whether the training set has transient regions removed, which is
what licenses attributing any difference in GAP to masking.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landmarks.config import ARMS, get_config
from landmarks.data.dataset import LandmarkDataset, build_frame, load_occlusion_index
from landmarks.data.subset import load_class_map, load_subset
from landmarks.data.transforms import build_transforms
from landmarks.engine.train import fit
from landmarks.models.build import build_model, count_parameters
from landmarks.utils.seed import set_seed, worker_init_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one arm of the occlusion study.")
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--cache", type=str, default="/kaggle/working/cache")
    p.add_argument("--index", type=str, default=None,
                   help="occlusion_index.csv (defaults to <cache>/occlusion_index.csv)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=str, default=None, help="override the run directory")
    return p.parse_args()


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle,                 # stable BatchNorm stats during training
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        worker_init_fn=worker_init_fn,
    )


def main() -> None:
    args = parse_args()

    cfg = get_config(args.arm)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.backbone is not None:
        cfg.model.backbone = args.backbone
    if args.workers is not None:
        cfg.train.num_workers = args.workers

    set_seed(cfg.seed, cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ---------------------------------------------------------------
    cache_root = Path(args.cache)
    index_path = Path(args.index) if args.index else cache_root / "occlusion_index.csv"

    subset = load_subset(cfg)
    class_map = load_class_map(cfg)
    occlusion = load_occlusion_index(index_path)

    train_frame = build_frame(subset, occlusion, "train")
    val_frame = build_frame(subset, occlusion, "val")
    num_classes = len(class_map)

    train_ds = LandmarkDataset(
        train_frame, cache_root,
        transform=build_transforms(cfg.train.image_size, train=True),
        apply_masking=cfg.masking.enabled,
        strategy=cfg.masking.strategy,
        apply_prob=cfg.masking.apply_prob,
        max_mask_fraction=cfg.masking.max_mask_fraction,
    )
    # Validation matches the training condition, so early stopping is measured
    # under the same distribution the model was trained on. The 2x2
    # cross-condition table is built separately in scripts/evaluate.py.
    val_ds = LandmarkDataset(
        val_frame, cache_root,
        transform=build_transforms(cfg.train.image_size, train=False),
        apply_masking=cfg.masking.enabled,
        strategy=cfg.masking.strategy,
        apply_prob=1.0 if cfg.masking.enabled else 0.0,
        max_mask_fraction=cfg.masking.max_mask_fraction,
    )

    train_loader = make_loader(train_ds, cfg.train.batch_size, cfg.train.num_workers, True)
    val_loader = make_loader(val_ds, cfg.eval.batch_size, cfg.train.num_workers, False)

    # ---- model --------------------------------------------------------------
    model = build_model(cfg, num_classes)

    run_dir = Path(args.out) if args.out else cfg.run_dir
    cfg.save(Path(run_dir) / "config.json")

    print(f"arm={cfg.arm} masking={cfg.masking.enabled} "
          f"strategy={cfg.masking.strategy} p={cfg.masking.apply_prob}")
    print(f"train={len(train_ds):,}  val={len(val_ds):,}  classes={num_classes:,}")
    print(f"model={cfg.model.backbone} head={cfg.model.head} "
          f"params={count_parameters(model):,}")
    print(f"device={device}  run_dir={run_dir}")

    fit(model, train_loader, val_loader, cfg, device, run_dir)


if __name__ == "__main__":
    main()
