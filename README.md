# FrameLock

**Operator-designated camera target lock and auto-framing for civilian camera drones.**

You click (or otherwise designate) a subject in the camera view; FrameLock keeps it
centered automatically by emitting bounded gimbal-framing commands, so you don't have
to fly the gimbal by hand. Detection can run in the **cloud** or **locally**; a local
Kalman filter predicts motion at frame rate so framing stays smooth even when the
detector is slow or briefly unavailable.

> **Scope and safety.** This is camera-framing assistance only — the same idea as DJI
> ActiveTrack or webcam auto-framing. It has **no** payload, firing, or autonomous
> interception capability. A lock only ever starts from an explicit operator
> designation; on target loss it coasts briefly on prediction, then **stops** and drops
> the lock. It never autonomously searches for or re-acquires a new subject.

---

## How it works

```
 frame ──▶ Detector ──▶ Tracker ──▶ Lock manager ──▶ Framing controller ──▶ Gimbal adapter
           (cloud /     (Kalman,    (operator-       (PID, deadband,        (sim / UDP /
            ONNX /       IoU, ego-   designated,      clamp, smoothing)      MAVLink)
            motion)      motion)     coast→stop)
```

- **Detector** finds subjects in each frame. Cloud for heavy/open-vocabulary AI; ONNX
  YOLO for onboard inference; a model-free motion detector for zero-setup use.
- **Tracker** associates detections to tracks with a per-track constant-velocity Kalman
  filter and **camera ego-motion compensation**, so tracks survive fast camera pans.
- **Lock manager** holds the operator-chosen track, predicts through short occlusions,
  and fails safe (stop) on loss.
- **Framing controller** turns the target's image-plane error into bounded pan/tilt
  angular rates (PID with deadband, anti-windup, rate clamp, and smoothing).
- **Gimbal adapter** delivers those rates to a simulated gimbal, a UDP JSON bridge, or a
  MAVLink gimbal manager.

The cloud is used for *detection*; prediction for the control loop runs *locally* because
cloud round-trips are too slow to close a smooth framing loop. An optional `latency_comp`
lead aims the camera where the subject is *now*, not where it was when the frame was sent.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .            # core (numpy, opencv-headless, httpx)
pip install -e ".[web]"     # + live demo (fastapi, uvicorn)
pip install -e ".[onnx]"    # + local ONNX detector (onnxruntime)
pip install -e ".[mavlink]" # + MAVLink gimbal (pymavlink)
pip install -e ".[dev]"     # + pytest, ruff
```

## Quick start

**Laptop camera (default):** run the web app, open it in your browser, click **Start camera**,
allow camera access, then click any subject in the scene to lock onto it.

```bash
framelock demo --host 0.0.0.0 --port 8000
```

A laptop camera is fixed, so framing is done with a **digital PTZ**: the software steers a crop
window to keep your subject centered (the "Gimbal output" view), emitting the same bounded rates
a real drone gimbal would use. Click-to-lock uses **CamShift** (color histogram), so it works on
static or moving subjects with no model download. The browser captures the webcam via
`getUserMedia` and posts frames to the pipeline; this requires a secure context (HTTPS or
`localhost`) and camera permission. A **Simulated scene** tab is included for demos without a camera.

**Simulated scene (no hardware):** switch to the *Simulated scene* tab; click a subject to lock.

**Render a proof video** of the closed loop (auto-locks a subject after warmup):

```bash
framelock render --output demo.mp4 --snapshot demo_frame.png --frames 360
# prints lock retention and mean/settled framing error in pixels
```

**Run on real footage** (video file, webcam index, or RTSP URL):

```bash
framelock run --source clip.mp4 --detector motion --gimbal sim \
  --auto-lock --no-display --output out.mp4
```

`--auto-lock` is an explicit operator opt-in that locks the subject nearest frame center
after warmup. Without it, nothing locks until you designate a target.

## Detector backends

| Backend | Flag | Notes |
| --- | --- | --- |
| Motion | `--detector motion` | Background subtraction; no model, good for demos/static cameras. |
| Cloud | `--detector cloud --cloud-endpoint URL` | Provider-agnostic JSON contract (below). |
| Local ONNX | `--detector onnx --model yolov8n.onnx` | `yolo export model=yolov8n.pt format=onnx imgsz=640`. |

**Cloud contract** — `POST` JSON:

```json
{ "image": "<base64 JPEG>", "width": 1280, "height": 720,
  "prompt": "the person in the red jacket", "model": "yolo-world" }
```

Response (any of `xywh` / `xyxy` / `cxcywh`, pixel or normalized):

```json
{ "format": "xywh", "normalized": false,
  "detections": [ { "bbox": [10, 20, 30, 40], "score": 0.93, "label": "person" } ] }
```

Set the API key with `--cloud-api-key` or `FRAMELOCK_CLOUD_KEY` (sent as a Bearer token).
On any network/parse failure the detector returns `[]` and the tracker coasts on
prediction. For a slow cloud model, run detection every Nth frame and allow a longer coast:

```bash
framelock run --source rtsp://drone/cam --detector cloud \
  --cloud-endpoint https://my-detector.example/detect \
  --detect-every 5 --latency-comp 0.25 --max-coast 2.0 --gimbal udp --udp-port 14550
```

## Gimbal adapters

| Adapter | Flag | Output |
| --- | --- | --- |
| Simulated | `--gimbal sim` | Integrates rates in software (tests/demo). |
| UDP JSON | `--gimbal udp --udp-host H --udp-port P` | `{"type":"gimbal_rate","pan_rate_dps":..,"tilt_rate_dps":..,"valid":..}` datagrams to a companion-computer bridge. |
| MAVLink | `--gimbal mavlink --mavlink-conn udpin:0.0.0.0:14550` | `GIMBAL_MANAGER_SET_PITCHYAW` angular velocities (rad/s) for PX4/ArduPilot. |

**Command convention:** `pan_rate > 0` moves the camera toward image right; `tilt_rate > 0`
toward image bottom. Rates are degrees/second, clamped to `--max-rate`. On loss the
controller emits a zero-rate hold (MAVLink sends zero rates, never a retract/neutral mode
change). Wire the adapter that matches your aircraft; the tracking core is hardware-independent.

## Tuning

`FramingConfig` (see `framelock/control.py`): `kp`, `ki`, `kd`, `deadband`, `max_rate`,
`smoothing`, `integral_limit`. Defaults are well-damped for both static and moving subjects.
Higher `max_rate` acquires faster but needs ego-motion compensation (already wired in the
pipeline) to keep tracks associated during fast pans.

## Tests

```bash
pytest          # 83 behavior tests: Kalman, tracker, lock, control, detectors, gimbal, pipeline, sim, webcam, web
ruff check .    # lint
```

Tests use no network, no model file, and no hardware (cloud via `httpx.MockTransport`,
MAVLink via an in-memory serializer, gimbal via simulation).

## Layout

```
framelock/
  types.py kalman.py tracker.py lock.py control.py pipeline.py   # core
  detectors/{base,cloud,onnx,motion,camshift}.py                 # perception
  gimbal.py ptz.py sources.py overlay.py                         # I/O + digital PTZ + HUD
  sim.py render.py webcam.py web.py cli.py                       # demo + webcam + entry points
  static/index.html                                              # demo UI (webcam + sim)
tests/                                                           # behavior tests
```

## Limitations and responsible use

- Association is IoU + Kalman (no appearance re-ID); long full occlusions or many similar
  subjects can cause ID switches. Appearance embeddings are a natural next step.
- The MAVLink adapter is verified at the message-construction level, not against a physical
  aircraft. Validate rates, limits, and failsafes on your hardware before flight.
- Operate within local aviation and privacy laws. This tool is for framing subjects you are
  authorized to film; it is not a targeting or weapon system.
