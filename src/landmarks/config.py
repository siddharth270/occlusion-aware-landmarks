# Siddharth Mehta
# CS5330 PRCV

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

from pathlib import Path

ARMS = ("baseline", "masked", "maskaug")

ON_KAGGLE = Path("/kaggle/input").exists()

REPO_ROOT = Path(__file__).resolve().parents[2]

def _find_competition_root() -> Path:

    candidates = [
        Path("/kaggle/input/competitions/landmark-recognition-2021"),
        Path("/kaggle/input/landmark-recognition-2021"),
    ]
    for c in candidates:
        if (c / "train.csv").exists():
            return c
    return candidates[0]


@dataclass
class Paths:
    competition: Path = field(default_factory=_find_competition_root)
    cache: Path = Path("/kaggle/input/gld21-subset-cache")
    detections: Path = Path("/kaggle/input/gld21-detections")
    manifests: Path = REPO_ROOT / "manifests"
    manifests_out: Path = (
        Path("/kaggle/working/manifests") if ON_KAGGLE else REPO_ROOT / "manifests"
    )
    artifacts: Path = Path("/kaggle/working/artifacts") if ON_KAGGLE else REPO_ROOT / "artifacts"


@dataclass
class SubsetConfig:
    n_classes: int = 1000
    min_images_per_class: int = 30
    max_images_per_class: int = 80
    split_ratios: tuple[float, float, float] = (0.80, 0.10, 0.10) # train, val, test
    verify_files_exist: bool = True


@dataclass
class CacheConfig:
    short_side: int = 256
    jpeg_quality: int = 90

@dataclass
class DetectionConfig:

    weights: str = "yolov8s-seg.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.70
    batch_size: int = 32
    use_segmentation: bool = True
    taxonomy_tiers: tuple[str, ...] = ("people", "vehicles", "animals", "portable_objects")

@dataclass
class OcclusionConfig:

    # occlusion_ratio = union area of transient regions / total image area.

    bins: tuple[float, ...] = (0.0, 0.02, 0.10, 0.25, 1.01)
    bin_labels: tuple[str, ...] = ("none", "low", "medium", "high")


@dataclass
class MaskingConfig:
    enabled: bool = False
    strategy: str = "mean_fill"        # black | mean_fill | blur | inpaint_telea
    dilate_px: int = 4                 # grow masks to catch boundary pixels
    max_mask_fraction: float = 0.85    # skip if masking would erase the image
    apply_prob: float = 1.0            # 1.0 = deterministic; <1.0 = augmentation
    subject_guard: bool = True         # skip large centred detections (statues, facades)
    subject_min_area: float = 0.25     # area fraction above which the guard applies



@dataclass
class ArcFaceConfig:
    scale: float = 30.0
    margin: float = 0.30


@dataclass
class ModelConfig:
    backbone: str = "tf_efficientnet_b0_ns"
    pretrained: bool = True
    embedding_dim: int = 512
    head: str = "arcface"
    arcface: ArcFaceConfig = field(default_factory=ArcFaceConfig)
    dropout: float = 0.2


@dataclass
class TrainConfig:
    image_size: int = 224
    epochs: int = 15
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1 
    label_smoothing: float = 0.0
    amp: bool = True
    grad_clip: float = 5.0
    early_stop_patience: int = 5


@dataclass
class EvalConfig:
    batch_size: int = 128
    cross_condition: bool = True
    bootstrap_iters: int = 1000


@dataclass
class Config:
    arm: str = "baseline"
    seed: int = 42
    deterministic: bool = True

    paths: Paths = field(default_factory=Paths)
    subset: SubsetConfig = field(default_factory=SubsetConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.paths.artifacts) / "runs" / self.arm

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=str))

    def save(self, path: str | Path | None = None) -> Path:
        # Dump the resolved config beside the run so results are traceable.
        path = Path(path) if path else self.run_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def get_config(arm: str = "baseline", **overrides) -> Config:

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")

    cfg = Config(arm=arm, **overrides)

    if arm == "baseline":
        cfg.masking.enabled = False
    elif arm == "masked":
        cfg.masking.enabled = True
        cfg.masking.apply_prob = 1.0
    elif arm == "maskaug":
        cfg.masking.enabled = True
        cfg.masking.apply_prob = 0.5

    return cfg