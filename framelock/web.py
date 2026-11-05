"""Interactive live demo: click a subject to lock; the camera auto-frames it.

Runs a synthetic world in a background thread and streams annotated frames as
MJPEG. Binding to 0.0.0.0 and using only relative URLs makes it work behind the
Arena live-preview proxy. No hardware or network AI is required for the demo.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from framelock import overlay
from framelock.sim import SimRunner
from framelock.webcam import WebcamEngine


class DemoEngine:
    def __init__(self, fps: float = 30.0) -> None:
        self.dt = 1.0 / fps
        self.runner = SimRunner()
        self.world = self.runner.world
        self._mutex = threading.Lock()
        self._jpeg: bytes | None = None
        self._state: dict = {"lock_state": "idle"}
        self._fps = fps
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        last = time.monotonic()
        while self._running:
            t0 = time.monotonic()
            with self._mutex:
                frame, result = self.runner.step(self.dt)
                annotated = overlay.draw(
                    frame, result, self.world.camera.pan, self.world.camera.tilt,
                    self._fps, "sim",
                )
                ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok:
                    self._jpeg = buf.tobytes()
                self._state = self._state_from(result)
            now = time.monotonic()
            period = now - last
            last = now
            if period > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / period)
            sleep = self.dt - (now - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _state_from(self, result) -> dict:
        cam = self.world.camera
        return {
            "lock_state": result.lock.state.value,
            "track_id": result.lock.track_id,
            "pan": round(cam.pan, 2),
            "tilt": round(cam.tilt, 2),
            "pan_rate": round(result.command.pan_rate, 2),
            "tilt_rate": round(result.command.tilt_rate, 2),
            "fps": round(self._fps, 1),
            "dropout": self.world.dropout_prob,
            "tracks": [
                {
                    "id": t.track_id, "label": t.label, "state": t.state.value,
                    "cx": round(t.bbox.cx / self.world.view_w, 4),
                    "cy": round(t.bbox.cy / self.world.view_h, 4),
                }
                for t in result.tracks
            ],
        }

    def frame(self) -> bytes | None:
        with self._mutex:
            return self._jpeg

    def state(self) -> dict:
        with self._mutex:
            return dict(self._state)

    def designate(self, nx: float, ny: float) -> dict:
        with self._mutex:
            res = self.runner.designate(nx * self.world.view_w, ny * self.world.view_h)
            return {"lock_state": res.state.value, "track_id": res.track_id}

    def release(self) -> dict:
        with self._mutex:
            res = self.runner.release()
            return {"lock_state": res.state.value}

    def toggle_dropout(self) -> dict:
        with self._mutex:
            self.world.dropout_prob = 0.0 if self.world.dropout_prob > 0 else 0.6
            return {"dropout": self.world.dropout_prob}


_PAGE = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


def create_app() -> FastAPI:
    engine = DemoEngine()
    webcam = WebcamEngine()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        engine.start()
        try:
            yield
        finally:
            engine.stop()

    app = FastAPI(title="FrameLock demo", lifespan=lifespan)

    @app.post("/api/webcam/frame")
    async def webcam_frame(request: Request) -> JSONResponse:
        body = await request.body()
        try:
            result = await run_in_threadpool(webcam.process, body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(result)

    @app.post("/api/webcam/lock")
    async def webcam_lock(request: Request) -> JSONResponse:
        body = await request.json()
        x = min(max(float(body.get("x", 0.5)), 0.0), 1.0)
        y = min(max(float(body.get("y", 0.5)), 0.0), 1.0)
        return JSONResponse(await run_in_threadpool(webcam.designate, x, y))

    @app.post("/api/webcam/release")
    def webcam_release() -> JSONResponse:
        return JSONResponse(webcam.release())

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

    @app.get("/state")
    def state() -> JSONResponse:
        return JSONResponse(engine.state())

    @app.post("/lock")
    async def lock(request: Request) -> JSONResponse:
        body = await request.json()
        x = float(body.get("x", 0.5))
        y = float(body.get("y", 0.5))
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        return JSONResponse(engine.designate(x, y))

    @app.post("/release")
    def release() -> JSONResponse:
        return JSONResponse(engine.release())

    @app.post("/dropout")
    def dropout() -> JSONResponse:
        return JSONResponse(engine.toggle_dropout())

    @app.get("/snapshot.jpg")
    def snapshot() -> StreamingResponse:
        jpeg = engine.frame() or b""
        return StreamingResponse(iter([jpeg]), media_type="image/jpeg")

    @app.get("/stream")
    def stream() -> StreamingResponse:
        def gen():
            boundary = b"--frame\r\n"
            while True:
                jpeg = engine.frame()
                if jpeg:
                    yield (
                        boundary
                        + b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        + jpeg
                        + b"\r\n"
                    )
                time.sleep(1 / 30)

        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    app.state.engine = engine
    return app


def run_demo(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
