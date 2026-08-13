# Siddharth Mehta, CS5330 PRCV, Final Project
# Picks the 1000 most photographed landmarks out of the full dataset, samples up
# to 80 images from each, and splits them into train, validation and test. The
# manifests it writes are fixed after this and every later step reads them.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landmarks.config import get_config
from landmarks.data.subset import build_subset
from landmarks.utils.seed import set_seed


# Reads the command line options, all of which override a config default.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the GLDv2 experimental subset.")
    p.add_argument("--n-classes", type=int, default=None)
    p.add_argument("--min-per-class", type=int, default=None)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-verify", action="store_true",
                   help="skip the on-disk file existence check (faster)")
    return p.parse_args()


# Builds the frozen subset and writes the split manifests.
def main() -> None:
    args = parse_args()

    cfg = get_config("baseline")
    if args.seed is not None:
        cfg.seed = args.seed
    if args.n_classes is not None:
        cfg.subset.n_classes = args.n_classes
    if args.min_per_class is not None:
        cfg.subset.min_images_per_class = args.min_per_class
    if args.max_per_class is not None:
        cfg.subset.max_images_per_class = args.max_per_class
    if args.no_verify:
        cfg.subset.verify_files_exist = False

    set_seed(cfg.seed, cfg.deterministic)
    build_subset(cfg)


if __name__ == "__main__":
    main()
