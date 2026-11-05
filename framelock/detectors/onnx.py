"""Local ONNX YOLOv8 detector (optional dependency: onnxruntime).

Export a model with:  yolo export model=yolov8n.pt format=onnx imgsz=640
then point OnnxYoloDetector at the .onnx file. Pre/post-processing is kept in
pure functions so it can be tested without a model file or onnxruntime.
"""

from __future__ import annotations

import numpy as np

from framelock.types import BBox, Detection

COCO_NAMES = (
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
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


def letterbox(
    frame: np.ndarray, size: int = 640
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize with unchanged aspect ratio + gray padding. Returns (img, scale, pad)."""
    import cv2

    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    x0, y0 = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, scale, (pad_x, pad_y)


def postprocess(
    pred: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    scale: float,
    pad: tuple[float, float],
    orig_w: int,
    orig_h: int,
    names: tuple[str, ...] = COCO_NAMES,
    label_allow: tuple[str, ...] | None = None,
) -> list[Detection]:
    """Decode a YOLOv8 head ([1,84,N] or [1,N,84]) into image-space detections."""
    import cv2

    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:  # [84, N] -> [N, 84]
        pred = pred.T
    if pred.shape[1] < 5:
        return []

    boxes_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(class_ids)), class_ids]

    keep = confs >= conf_thresh
    boxes_cxcywh, class_ids, confs = boxes_cxcywh[keep], class_ids[keep], confs[keep]
    if len(confs) == 0:
        return []

    # cxcywh (letterbox space) -> xywh (original image space).
    cx, cy, bw, bh = boxes_cxcywh.T
    pad_x, pad_y = pad
    x = (cx - bw / 2 - pad_x) / scale
    y = (cy - bh / 2 - pad_y) / scale
    w = bw / scale
    h = bh / scale

    labels = [names[int(c)] if int(c) < len(names) else str(int(c)) for c in class_ids]
    if label_allow:
        mask = np.array([lab in label_allow for lab in labels])
        x, y, w, h = x[mask], y[mask], w[mask], h[mask]
        confs, labels = confs[mask], [lab for lab, m in zip(labels, mask, strict=True) if m]
        if len(confs) == 0:
            return []

    boxes_xywh = np.stack([x, y, w, h], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, confs.tolist(), conf_thresh, iou_thresh)
    idxs = np.array(idxs).flatten().tolist()

    out: list[Detection] = []
    for i in idxs:
        bbox = BBox(float(x[i]), float(y[i]), float(w[i]), float(h[i])).clipped(orig_w, orig_h)
        if bbox.area > 0:
            out.append(Detection(bbox, float(confs[i]), labels[i]))
    return out


class OnnxYoloDetector:
    name = "onnx-yolov8"

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        label_allow: tuple[str, ...] | None = ("person",),
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "onnxruntime is required for local detection: pip install framelock[onnx]"
            ) from exc
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.label_allow = label_allow

    def detect(self, frame: np.ndarray) -> list[Detection]:
        orig_h, orig_w = frame.shape[:2]
        canvas, scale, pad = letterbox(frame, self.imgsz)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)
        pred = self.session.run(None, {self.input_name: blob})[0]
        return postprocess(
            pred, self.conf_thresh, self.iou_thresh, scale, pad, orig_w, orig_h,
            label_allow=self.label_allow,
        )
