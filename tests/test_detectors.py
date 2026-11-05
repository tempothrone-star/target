"""Behavior tests for detector backends (no network, no model file needed)."""

import base64

import cv2
import httpx
import numpy as np

from framelock.detectors.cloud import CloudConfig, CloudDetector, parse_detections
from framelock.detectors.motion import MotionDetector
from framelock.detectors.onnx import letterbox, postprocess


def _frame(w=64, h=48, color=(0, 0, 0)):
    return np.full((h, w, 3), color, dtype=np.uint8)


# ---- cloud ----------------------------------------------------------------

def test_cloud_parses_xywh_pixels():
    cfg = CloudConfig(endpoint="http://x")
    data = {"detections": [{"bbox": [10, 20, 30, 40], "score": 0.9, "label": "person"}]}
    dets = parse_detections(data, 640, 480, cfg)
    assert len(dets) == 1
    assert (dets[0].bbox.x, dets[0].bbox.y, dets[0].bbox.w, dets[0].bbox.h) == (10, 20, 30, 40)


def test_cloud_parses_normalized_xyxy():
    cfg = CloudConfig(endpoint="http://x")
    data = {
        "format": "xyxy",
        "normalized": True,
        "detections": [{"bbox": [0.1, 0.2, 0.5, 0.6], "score": 0.8, "label": "car"}],
    }
    dets = parse_detections(data, 100, 100, cfg)
    b = dets[0].bbox
    assert (b.x, b.y, b.w, b.h) == (10.0, 20.0, 40.0, 40.0)


def test_cloud_filters_low_score_and_label():
    cfg = CloudConfig(endpoint="http://x", min_score=0.5, label_allow=("person",))
    data = {
        "detections": [
            {"bbox": [0, 0, 5, 5], "score": 0.2, "label": "person"},
            {"bbox": [0, 0, 5, 5], "score": 0.9, "label": "car"},
            {"bbox": [0, 0, 5, 5], "score": 0.9, "label": "person"},
        ]
    }
    assert len(parse_detections(data, 100, 100, cfg)) == 1


def test_cloud_sends_image_and_auth_and_handles_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        body = request.read()
        seen["has_image"] = b"image" in body
        return httpx.Response(200, json={
            "detections": [{"bbox": [1, 2, 3, 4], "score": 0.9, "label": "person"}]
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    det = CloudDetector(CloudConfig(endpoint="http://x", api_key="secret"), client=client)
    dets = det.detect(_frame())
    assert seen["auth"] == "Bearer secret"
    assert seen["has_image"]
    assert len(dets) == 1


def test_cloud_network_failure_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    det = CloudDetector(CloudConfig(endpoint="http://x"), client=client)
    assert det.detect(_frame()) == []
    assert det.last_error is not None


def test_cloud_encodes_valid_jpeg():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["image"] = json.loads(request.read())["image"]
        return httpx.Response(200, json={"detections": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    det = CloudDetector(CloudConfig(endpoint="http://x"), client=client)
    det.detect(_frame(color=(10, 120, 200)))
    raw = base64.b64decode(captured["image"])
    assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is not None


# ---- motion ---------------------------------------------------------------

def test_motion_detects_moving_blob_after_warmup():
    det = MotionDetector(min_area=100)
    for _ in range(30):
        det.detect(_frame(320, 240))  # learn empty background
    frame = _frame(320, 240)
    cv2.rectangle(frame, (140, 100), (180, 140), (255, 255, 255), -1)
    dets = det.detect(frame)
    assert len(dets) >= 1
    assert any(120 < d.bbox.cx < 200 and 80 < d.bbox.cy < 160 for d in dets)


# ---- onnx postprocess -----------------------------------------------------

def _yolo_pred(cx, cy, w, h, class_id, conf, n=8400, classes=80):
    pred = np.zeros((4 + classes, n), dtype=np.float32)
    pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0] = cx, cy, w, h
    pred[4 + class_id, 0] = conf
    return pred[None]


def test_postprocess_decodes_single_detection():
    pred = _yolo_pred(320, 320, 100, 100, class_id=0, conf=0.9)
    dets = postprocess(pred, 0.35, 0.45, scale=1.0, pad=(0, 0), orig_w=640, orig_h=640)
    assert len(dets) == 1
    assert dets[0].label == "person"
    assert abs(dets[0].bbox.cx - 320) < 1 and abs(dets[0].bbox.cy - 320) < 1


def test_postprocess_drops_low_confidence():
    pred = _yolo_pred(320, 320, 100, 100, 0, conf=0.1)
    assert postprocess(pred, 0.35, 0.45, 1.0, (0, 0), 640, 640) == []


def test_postprocess_label_allow_filters():
    pred = _yolo_pred(320, 320, 100, 100, class_id=2, conf=0.9)  # car
    dets = postprocess(pred, 0.35, 0.45, 1.0, (0, 0), 640, 640, label_allow=("person",))
    assert dets == []


def test_postprocess_undoes_letterbox():
    # 1280x720 letterboxed into 640: scale=0.5, pad_y=140.
    frame = np.zeros((720, 1280, 3), np.uint8)
    canvas, scale, pad = letterbox(frame, 640)
    assert canvas.shape == (640, 640, 3)
    assert abs(scale - 0.5) < 1e-6
    # A box centered in the letterbox image should map to the image center.
    pred = _yolo_pred(320, 320, 100, 100, 0, 0.9)
    dets = postprocess(pred, 0.35, 0.45, scale, pad, 1280, 720)
    assert abs(dets[0].bbox.cx - 640) < 2 and abs(dets[0].bbox.cy - 360) < 2


def test_postprocess_nms_suppresses_duplicates():
    pred = np.zeros((1, 84, 8400), np.float32)
    for i in range(3):
        pred[0, 0, i] = 320 + i
        pred[0, 1, i] = 320
        pred[0, 2, i] = 100
        pred[0, 3, i] = 100
        pred[0, 4, i] = 0.9 - i * 0.05
    dets = postprocess(pred, 0.35, 0.45, 1.0, (0, 0), 640, 640)
    assert len(dets) == 1
