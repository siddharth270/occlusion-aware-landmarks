# Siddharth Mehta, CS5330 PRCV, Final Project
# Wraps the YOLO segmentation model. It keeps a wide set of detections at a low
# threshold so the filtering choices can be changed later without having to run
# the detector over every image again.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from landmarks.occlusion.taxonomy import COCO_CLASSES, transient_class_ids, NEVER_MASK


@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    box: tuple[float, float, float, float]
    polygon: np.ndarray | None = None


@dataclass
class ImageDetections:
    image_id: str
    orig_h: int
    orig_w: int
    detections: list[Detection] = field(default_factory=list)
    orig_img: np.ndarray | None = None

    # How many transient objects were kept for this image.
    @property
    def n_transient(self) -> int:
        return len(self.detections)


class TransientDetector:

    # Loads the weights and works out which classes may ever be masked.
    def __init__(
        self,
        weights: str = "yolov8m-seg.pt",
        imgsz: int = 640,
        conf: float = 0.10,
        iou: float = 0.70,
        device: int | str = 0,
        use_segmentation: bool = True,
        max_det: int = 100,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.use_segmentation = use_segmentation
        self.max_det = max_det
        self.keep_ids = {
            i for i, n in enumerate(COCO_CLASSES) if n not in NEVER_MASK
        }


    # Runs detection in small batches and yields one result per image.
    def predict(
        self,
        paths: Sequence[str | Path],
        batch_size: int = 4,
        keep_images: bool = False,
        verbose: bool = False,
    ) -> Iterator[ImageDetections]:
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            results = self.model.predict(
                source=[str(p) for p in batch],
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                max_det=self.max_det,
                retina_masks=False,
                stream=False,
                verbose=verbose,
            )

            if len(results) != len(batch):
                raise RuntimeError(
                    f"detector returned {len(results)} results for {len(batch)} "
                    "inputs; positional pairing would misalign ids"
                )

            for p, r in zip(batch, results):
                yield self._parse(Path(p).stem, r, keep_images)
            del results


    # Turns one raw detection result into normalised boxes and polygons.
    def _parse(self, image_id: str, r, keep_image: bool) -> ImageDetections:
        h, w = r.orig_shape
        out = ImageDetections(
            image_id=image_id,
            orig_h=int(h),
            orig_w=int(w),
            orig_img=r.orig_img if keep_image else None,
        )

        if r.boxes is None or len(r.boxes) == 0:
            return out

        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        xyxyn = r.boxes.xyxyn.cpu().numpy()

        polys = None
        if self.use_segmentation and r.masks is not None:
            polys = r.masks.xyn

        for i, cid in enumerate(cls):
            if cid not in self.keep_ids:
                continue
            poly = None
            if polys is not None and i < len(polys) and len(polys[i]) >= 3:
                poly = np.asarray(polys[i], dtype=np.float32)
            out.detections.append(
                Detection(
                    class_id=int(cid),
                    class_name=COCO_CLASSES[cid],
                    conf=float(conf[i]),
                    box=tuple(float(v) for v in xyxyn[i]),
                    polygon=poly,
                )
            )
        return out


# Releases cached GPU memory between chunks.
def free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
