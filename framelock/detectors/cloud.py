"""Cloud detection backend over a small, provider-agnostic JSON contract.

Request (POST JSON):
    {
      "image": "<base64 JPEG>",
      "width": 1280, "height": 720,
      "prompt": "the person in the red jacket",   # optional, open-vocabulary
      "model": "yolo-world"                        # optional
    }

Response:
    {
      "detections": [
        {"bbox": [x, y, w, h], "score": 0.93, "label": "person"}
      ],
      "format": "xywh",          # optional: xywh | xyxy | cxcywh
      "normalized": false         # optional: coords in 0..1 instead of pixels
    }

This works with a self-hosted detector, Roboflow, or any vision service wrapped
to this contract. On any network/parse failure it returns [] so the local
tracker coasts on prediction instead of crashing the control loop.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

import cv2
import httpx
import numpy as np

from framelock.types import BBox, Detection

log = logging.getLogger(__name__)


@dataclass
class CloudConfig:
    endpoint: str
    api_key: str | None = None
    model: str | None = None
    prompt: str | None = None
    timeout: float = 2.0
    jpeg_quality: int = 80
    min_score: float = 0.35
    bbox_format: str = "xywh"  # default expected format; response may override
    normalized: bool = False
    label_allow: tuple[str, ...] | None = None


class CloudDetector:
    name = "cloud"

    def __init__(self, config: CloudConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self.last_error: str | None = None
        self._headers = {"Content-Type": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = client or httpx.Client(timeout=config.timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _encode(self, frame: np.ndarray) -> str:
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        )
        if not ok:
            raise ValueError("failed to JPEG-encode frame")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        payload = {"image": self._encode(frame), "width": w, "height": h}
        if self.config.prompt:
            payload["prompt"] = self.config.prompt
        if self.config.model:
            payload["model"] = self.config.model
        try:
            resp = self._client.post(self.config.endpoint, json=payload, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            self.last_error = str(exc)
            log.warning("cloud detector failed: %s", exc)
            return []
        self.last_error = None
        return parse_detections(data, w, h, self.config)


def parse_detections(
    data: dict, width: int, height: int, config: CloudConfig
) -> list[Detection]:
    fmt = data.get("format", config.bbox_format)
    normalized = bool(data.get("normalized", config.normalized))
    out: list[Detection] = []
    for item in data.get("detections", []):
        try:
            raw = item["bbox"]
            score = float(item.get("score", 0.0))
            label = str(item.get("label", "object"))
        except (KeyError, TypeError, ValueError):
            continue
        if score < config.min_score:
            continue
        if config.label_allow and label not in config.label_allow:
            continue
        bbox = _to_xywh(raw, fmt, normalized, width, height)
        if bbox is None or bbox.area <= 0:
            continue
        out.append(Detection(bbox=bbox, score=score, label=label))
    return out


def _to_xywh(
    raw: list, fmt: str, normalized: bool, width: int, height: int
) -> BBox | None:
    if len(raw) != 4:
        return None
    a, b, c, d = (float(v) for v in raw)
    if normalized:
        a, c = a * width, c * width
        b, d = b * height, d * height
    if fmt == "xywh":
        x, y, w, h = a, b, c, d
    elif fmt == "xyxy":
        x, y, w, h = a, b, c - a, d - b
    elif fmt == "cxcywh":
        x, y, w, h = a - c / 2, b - d / 2, c, d
    else:
        return None
    if w < 0 or h < 0:
        return None
    return BBox(x, y, w, h).clipped(width, height)
