"""Detector backends. All implement the Detector protocol and return Detections."""

from framelock.detectors.base import Detector
from framelock.detectors.cloud import CloudConfig, CloudDetector
from framelock.detectors.motion import MotionDetector
from framelock.detectors.onnx import OnnxYoloDetector

__all__ = ["Detector", "CloudConfig", "CloudDetector", "MotionDetector", "OnnxYoloDetector"]
