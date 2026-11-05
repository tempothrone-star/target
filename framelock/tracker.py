"""Greedy IoU multi-object tracker with a Kalman filter per track.

Detections are associated to predicted track boxes. Tracks are born tentative,
become confirmed after a few hits, coast on prediction when unmatched, and are
removed after a bounded number of misses. IDs are stable and monotonic.
"""

from __future__ import annotations

from framelock.kalman import BoxKalmanFilter
from framelock.types import BBox, Detection, Track, TrackState


class Tracker:
    def __init__(
        self,
        min_hits: int = 3,
        max_misses: int = 30,
        iou_threshold: float = 0.15,
        process_noise: float = 1.0,
        measurement_noise: float = 4.0,
    ) -> None:
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._next_id = 1
        self._filters: dict[int, BoxKalmanFilter] = {}
        self._tracks: dict[int, Track] = {}

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def _spawn(self, det: Detection) -> Track:
        track_id = self._next_id
        self._next_id += 1
        self._filters[track_id] = BoxKalmanFilter(
            det.bbox, self._process_noise, self._measurement_noise
        )
        track = Track(track_id=track_id, bbox=det.bbox, label=det.label, score=det.score)
        self._tracks[track_id] = track
        return track

    def _associate(
        self, track_ids: list[int], detections: list[Detection]
    ) -> tuple[dict[int, int], list[int]]:
        """Return matched {track_id: det_index} and unmatched detection indices."""
        pairs: list[tuple[float, int, int]] = []
        for tid in track_ids:
            predicted = self._filters[tid].box()
            for di, det in enumerate(detections):
                iou = predicted.iou(det.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched: dict[int, int] = {}
        used_dets: set[int] = set()
        for _iou, tid, di in pairs:
            if tid in matched or di in used_dets:
                continue
            matched[tid] = di
            used_dets.add(di)
        unmatched = [di for di in range(len(detections)) if di not in used_dets]
        return matched, unmatched

    def update(
        self, detections: list[Detection], dt: float, ego_shift: tuple[float, float] = (0.0, 0.0)
    ) -> list[Track]:
        # Compensate camera motion so static-world points stay aligned, then
        # predict every existing track forward to this frame's timestamp.
        ex, ey = ego_shift
        for tid in self._tracks:
            if ex or ey:
                self._filters[tid].shift(-ex, -ey)
            self._filters[tid].predict(dt)

        track_ids = list(self._tracks.keys())
        matched, unmatched_dets = self._associate(track_ids, detections)

        for tid, track in self._tracks.items():
            track.age += 1
            if tid in matched:
                det = detections[matched[tid]]
                box = self._filters[tid].update(det.bbox)
                track.bbox = box
                track.label = det.label
                track.score = det.score
                track.hits += 1
                track.misses = 0
                track.velocity = self._filters[tid].velocity()
                if track.hits >= self.min_hits:
                    track.state = TrackState.CONFIRMED
                else:
                    track.state = TrackState.TENTATIVE
            else:
                track.misses += 1
                track.bbox = self._filters[tid].box()
                track.velocity = self._filters[tid].velocity()
                track.state = TrackState.COASTING

        for di in unmatched_dets:
            self._spawn(detections[di])

        # Drop tracks that have coasted too long.
        for tid in [t for t, tr in self._tracks.items() if tr.misses > self.max_misses]:
            del self._tracks[tid]
            del self._filters[tid]

        return self.tracks

    def get(self, track_id: int) -> Track | None:
        return self._tracks.get(track_id)

    def filter_for(self, track_id: int) -> BoxKalmanFilter | None:
        return self._filters.get(track_id)

    @staticmethod
    def nearest_to_point(tracks: list[Track], x: float, y: float) -> Track | None:
        """Pick the track whose box contains the point, else the nearest center."""
        containing = [t for t in tracks if t.bbox.contains(x, y)]
        if containing:
            return min(containing, key=lambda t: t.bbox.area)
        if not tracks:
            return None
        return min(tracks, key=lambda t: (t.bbox.cx - x) ** 2 + (t.bbox.cy - y) ** 2)


def box_distance(a: BBox, b: BBox) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5
