# YOLOv8-seg wrapper for transient occluder detection.

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
    box: tuple[float, float, float, float]      # normalised xyxy
    polygon: np.ndarray | None = None           # normalised (N, 2), or None if box-only


@dataclass
class ImageDetections:
    image_id: str
    orig_h: int
    orig_w: int
    detections: list[Detection] = field(default_factory=list)
    orig_img: np.ndarray | None = None          # BGR; reused so we never re-read the JPEG

    @property
    def n_transient(self) -> int:
        return len(self.detections)


class TransientDetector:
    """Batched YOLO inference over image paths.

    Args:
        weights: Ultralytics checkpoint name, e.g. "yolov8s-seg.pt".
        imgsz: inference resolution. 640 on ~800px source keeps small,
            distant tourists detectable.
        conf: keep threshold. Set this LOW (0.10) and filter later --
            re-thresholding stored detections is free, re-running YOLO is not.
        tiers: taxonomy tiers defining which classes are transient.
    """

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
        # A low `conf` is deliberate (store a superset, filter offline), but it
        # lets NMS return up to `max_det` instances per image, each carrying a
        # full-resolution mask tensor. 300 x 640 x 640 floats per image OOMs a
        # T4 at any useful batch size. 100 is far above the real occluder count
        # in landmark photos (observed ~3/image) while capping memory.
        self.max_det = max_det
        self.keep_ids = {
            i for i, n in enumerate(COCO_CLASSES) if n not in NEVER_MASK
        }



    def predict(
        self,
        paths: Sequence[str | Path],
        batch_size: int = 4,
        keep_images: bool = False,
        verbose: bool = False,
    ) -> Iterator[ImageDetections]:
        """Stream detections, one ImageDetections per input path.

        Batching is explicit rather than delegated to Ultralytics' `batch=`
        argument. That path warms the model up with a batch-sized dummy tensor,
        and in fp32 a yolov8m-seg forward at 640px costs ~0.5GB per image, which
        OOMs a 15GB T4 long before the requested batch size is reached. Slicing
        here bounds peak memory exactly and lets each batch's Results objects be
        released before the next batch is built.
        """
        for start in range(0, len(paths), batch_size):
            batch = [str(p) for p in paths[start : start + batch_size]]
            results = self.model.predict(
                source=batch,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                max_det=self.max_det,
                retina_masks=False,
                stream=False,
                verbose=verbose,
            )
            for r in results:
                # Id comes from the result, not a zipped path list: if
                # Ultralytics skips an unreadable JPEG, zip() would desync and
                # every subsequent image would be written under the wrong id.
                yield self._parse(Path(r.path).stem, r, keep_images)
            del results


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

        # Ultralytics gives one polygon per detection, in the same order as boxes.
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


def free_gpu() -> None:
    """Drop cached allocations. Kaggle kernels hold every model you instantiate,
    and the cache build calls this between chunks to keep memory flat."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
