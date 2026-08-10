from __future__ import annotations

# COCO-80 in class-id order, as used by all Ultralytics YOLO detection weights.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(COCO_CLASSES)}

TIERS: dict[str, tuple[str, ...]] = {
    # Tier 1 -- the dominant occluder in tourist photography.
    "people": ("person",),

    # Tier 2 -- transient by definition; parked or passing, never part of the landmark.
    "vehicles": ("bicycle", "car", "motorcycle", "bus", "truck"),

    # Tier 3 -- pets, pigeons, horse-drawn carriages, tourist camels.
    "animals": (
        "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe",
    ),

    # Tier 4 -- carried by visitors. Deliberately narrow: the wider COCO
    # sports/tableware classes (frisbee, kite, sports ball, bottle) were
    # observed firing on architectural features and were removed.
    "portable_objects": (
        "backpack", "umbrella", "handbag", "suitcase",
    ),

    # OPT-IN. Excluded by default: high false-positive rate on landmark imagery.
    "misc_objects": (
        "tie", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "cell phone",
        "laptop", "book",
    ),


    # Tier 5 -- OPT-IN. Semi-permanent street furniture. Arguably part of the
    # scene rather than noise; included as an ablation, off by default.
    "street_furniture": (
        "bench", "chair", "potted plant", "dining table",
        "traffic light", "fire hydrant", "parking meter", "stop sign",
    ),
}

# Deliberately never masked: each of these can BE the landmark.
#   airplane -> museum/monument aircraft
#   train    -> preserved locomotives, station exhibits
#   clock    -> Big Ben, Prague Astronomical Clock, Musee d'Orsay
NEVER_MASK: tuple[str, ...] = ("airplane", "train", "clock", "boat")

DEFAULT_TIERS: tuple[str, ...] = ("people", "vehicles", "animals", "portable_objects")


def transient_class_ids(tiers: tuple[str, ...] = DEFAULT_TIERS) -> set[int]:
    """Resolve tier names to the COCO class ids that should be masked."""
    unknown = set(tiers) - set(TIERS)
    if unknown:
        raise ValueError(f"unknown tiers {sorted(unknown)}; valid: {sorted(TIERS)}")

    names: set[str] = set()
    for tier in tiers:
        names.update(TIERS[tier])

    names -= set(NEVER_MASK)          # belt and braces
    return {NAME_TO_ID[n] for n in names}


def describe_taxonomy(tiers: tuple[str, ...] = DEFAULT_TIERS) -> str:
    """Human-readable summary, for logging into the report."""
    lines = [f"transient taxonomy ({len(transient_class_ids(tiers))} COCO classes):"]
    for tier in tiers:
        lines.append(f"  {tier:18s} {', '.join(TIERS[tier])}")
    lines.append(f"  {'never masked':18s} {', '.join(NEVER_MASK)}")
    return "\n".join(lines)


# Per-class keep thresholds. `person` is the class COCO detectors are best at
# and the occluder that matters most, so it gets a permissive threshold.
# Everything else must clear a high bar: the observed failure mode is rare,
# low-confidence classes (boat, kite, bird) firing on architecture and
# fireworks, which then masks the landmark itself.
CONF_THRESHOLDS: dict[str, float] = {
    "person": 0.30,
    "umbrella": 0.40,
    "car": 0.40,
    "bus": 0.40,
    "truck": 0.40,
    "bicycle": 0.40,
    "motorcycle": 0.40,
}
DEFAULT_CONF_THRESHOLD: float = 0.55


def conf_threshold_for(class_name: str) -> float:
    """Confidence a detection must reach before it is allowed to mask pixels."""
    return CONF_THRESHOLDS.get(class_name, DEFAULT_CONF_THRESHOLD)
