# Siddharth Mehta, CS5330 PRCV, Final Project
# Loads each trained model and scores it on the test split twice, once on raw
# images and once on masked ones. Writes the cross condition table, the results
# broken down by occlusion level, and the significance tests.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landmarks.config import ARMS, get_config
from landmarks.data.dataset import LandmarkDataset, build_frame, load_occlusion_index
from landmarks.data.subset import load_class_map, load_subset
from landmarks.data.transforms import build_transforms
from landmarks.engine.train import predict
from landmarks.eval.gap import predictions_from_logits
from landmarks.eval.stats import bootstrap_gap_ci, mcnemar, paired_bootstrap_delta
from landmarks.eval.stratified import cross_condition_table, stratified_gap
from landmarks.models.build import build_model
from landmarks.utils.io import ensure_dir
from landmarks.utils.seed import set_seed, worker_init_fn


# Reads the command line options for the evaluation run.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate all arms and build result tables.")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--cache", type=str, default="/kaggle/working/cache")
    p.add_argument("--index", type=str, default=None)
    p.add_argument("--runs", type=str, default=None, help="root holding <arm>/best.pt")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


# Scores one trained model on one input condition and returns its
# predictions with the confidence attached to each.
def eval_one(cfg, ckpt_path: Path, frame, cache_root, num_classes, mask_eval: bool,
             device, workers: int):
    from torch.utils.data import DataLoader

    dataset = LandmarkDataset(
        frame, cache_root,
        transform=build_transforms(cfg.train.image_size, train=False),
        apply_masking=mask_eval,
        strategy=cfg.masking.strategy,
        apply_prob=1.0 if mask_eval else 0.0,
        max_mask_fraction=cfg.masking.max_mask_fraction,
    )
    loader = DataLoader(dataset, batch_size=cfg.eval.batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True,
                        worker_init_fn=worker_init_fn)

    model = build_model(cfg, num_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device)

    labels, logits = predict(model, loader, device,
                             desc=f"{cfg.arm}/{'masked' if mask_eval else 'raw'}")
    if not np.array_equal(labels, frame.label.values):
        raise RuntimeError("prediction order does not match frame order")

    del model
    torch.cuda.empty_cache()
    return predictions_from_logits(logits)


# Evaluates every arm on both raw and masked input, then writes the
# result tables and the significance tests.
def main() -> None:
    args = parse_args()

    base = get_config("baseline")
    set_seed(base.seed, base.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_root = Path(args.cache)
    index_path = Path(args.index) if args.index else cache_root / "occlusion_index.csv"
    runs_root = Path(args.runs) if args.runs else Path(base.paths.artifacts) / "runs"
    out_dir = ensure_dir(args.out or Path(base.paths.artifacts) / "results")

    subset = load_subset(base)
    class_map = load_class_map(base)
    occlusion = load_occlusion_index(index_path)
    frame = build_frame(subset, occlusion, args.split)
    num_classes = len(class_map)

    bins, labels_ = base.occlusion.bins, base.occlusion.bin_labels
    print(f"evaluating on {args.split}: {len(frame):,} images, {num_classes:,} classes")

    predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    stratified: dict[tuple[str, str], pd.DataFrame] = {}

    for arm in args.arms:
        ckpt = runs_root / arm / "best.pt"
        if not ckpt.exists():
            print(f"skip {arm}: no checkpoint at {ckpt}")
            continue

        cfg = get_config(arm)
        for condition, mask_eval in [("raw", False), ("masked", True)]:
            preds, confs = eval_one(cfg, ckpt, frame, cache_root, num_classes,
                                    mask_eval, device, args.workers)
            predictions[(arm, condition)] = (preds, confs)

            df = stratified_gap(frame, preds, confs, bins, labels_)
            df.insert(0, "eval_condition", condition)
            df.insert(0, "train_arm", arm)
            stratified[(arm, condition)] = df

            overall = df[df.occlusion_bin == "ALL"].iloc[0]
            print(f"{arm:9s} / {condition:6s}  GAP={overall.gap:.4f}  top1={overall.top1:.4f}")

    if not stratified:
        raise RuntimeError(f"no checkpoints found under {runs_root}")

    strat_all = pd.concat(stratified.values(), ignore_index=True)
    strat_all.to_csv(out_dir / "stratified_gap.csv", index=False)

    matrix = cross_condition_table(stratified)
    matrix.to_csv(out_dir / "cross_condition_gap.csv")
    print("\ncross-condition GAP (rows=train arm, cols=eval input):")
    print(matrix.round(4).to_string())

    y = frame.label.values
    report: dict = {"split": args.split, "n": int(len(frame)), "arms": {}, "tests": {}}

    for (arm, condition), (preds, confs) in predictions.items():
        point, lo, hi = bootstrap_gap_ci(y, preds, confs,
                                         iters=base.eval.bootstrap_iters, seed=base.seed)
        report["arms"][f"{arm}/{condition}"] = {
            "gap": point, "ci_low": lo, "ci_high": hi,
            "top1": float((preds == y).mean()),
        }
        print(f"{arm:9s}/{condition:6s} GAP {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    # Gives the input condition that an arm was trained on.
    def matched(arm: str) -> str:
        return "masked" if get_config(arm).masking.enabled else "raw"

    if ("baseline", "raw") in predictions:
        pa, ca = predictions[("baseline", "raw")]
        for arm in args.arms:
            key = (arm, matched(arm))
            if arm == "baseline" or key not in predictions:
                continue
            pb, cb = predictions[key]
            boot = paired_bootstrap_delta(y, pa, ca, pb, cb,
                                          iters=base.eval.bootstrap_iters, seed=base.seed)
            mc = mcnemar(pa == y, pb == y)
            report["tests"][f"{arm} vs baseline"] = {"bootstrap": boot, "mcnemar": mc}
            print(f"\n{arm} vs baseline (matched conditions):")
            print(f"  dGAP {boot['delta']:+.4f}  95% CI [{boot['ci_low']:+.4f}, "
                  f"{boot['ci_high']:+.4f}]  p={boot['p_value']:.4f}  "
                  f"significant={boot['significant']}")
            print(f"  McNemar: {arm} fixed {mc['b_only']}, broke {mc['a_only']}, "
                  f"p={mc['p_value']:.4g}")

    with open(out_dir / "significance.json", "w") as f:
        json.dump(report, f, indent=2)

    for (arm, condition), (preds, confs) in predictions.items():
        pd.DataFrame({
            "id": frame.id.values, "label": y, "pred": preds, "conf": confs,
            "occlusion_ratio": frame.occlusion_ratio.values,
        }).to_parquet(out_dir / f"preds_{arm}_{condition}.parquet", index=False)

    print(f"\nwrote results to {out_dir}")


if __name__ == "__main__":
    main()
