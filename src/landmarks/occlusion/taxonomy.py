# Siddharth Mehta, CS5330 PRCV, Final Project
# Decides which object classes count as transient. A few classes are never masked
# because they can be the landmark themselves, like a clock tower or a museum plane.

from __future__ import annotations

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
    "people": ("person",),

    "vehicles": ("bicycle", "car", "motorcycle", "bus", "truck"),

    "animals": (
        "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe",
    ),

    "portable_objects": (
        "backpack", "umbrella", "handbag", "suitcase",
    ),

    "misc_objects": (
        "tie", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "cell phone",
        "laptop", "book",
    ),


    "street_furniture": (
        "bench", "chair", "potted plant", "dining table",
        "traffic light", "fire hydrant", "parking meter", "stop sign",
    ),
}

NEVER_MASK: tuple[str, ...] = ("airplane", "train", "clock", "boat")

DEFAULT_TIERS: tuple[str, ...] = ("people", "vehicles", "animals", "portable_objects")


# Turns tier names into the set of COCO class ids that may be masked.
def transient_class_ids(tiers: tuple[str, ...] = DEFAULT_TIERS) -> set[int]:
    unknown = set(tiers) - set(TIERS)
    if unknown:
        raise ValueError(f"unknown tiers {sorted(unknown)}; valid: {sorted(TIERS)}")

    names: set[str] = set()
    for tier in tiers:
        names.update(TIERS[tier])

    names -= set(NEVER_MASK)
    return {NAME_TO_ID[n] for n in names}


# Readable summary of the taxonomy, for logging into the report.
def describe_taxonomy(tiers: tuple[str, ...] = DEFAULT_TIERS) -> str:
    lines = [f"transient taxonomy ({len(transient_class_ids(tiers))} COCO classes):"]
    for tier in tiers:
        lines.append(f"  {tier:18s} {', '.join(TIERS[tier])}")
    lines.append(f"  {'never masked':18s} {', '.join(NEVER_MASK)}")
    return "\n".join(lines)


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


# Confidence a detection of this class must reach before it is
# allowed to mask anything.
def conf_threshold_for(class_name: str) -> float:
    return CONF_THRESHOLDS.get(class_name, DEFAULT_CONF_THRESHOLD)
