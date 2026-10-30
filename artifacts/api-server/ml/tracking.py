from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Track:
    track_id: int
    bbox: list[float]
    missed: int = 0
    last_frame: int = 0


@dataclass
class IoUTracker:
    """Video-scoped IoU tracker. Track IDs are not identities and are not global."""

    iou_threshold: float = 0.3
    max_missed: int = 8
    tracks: dict[int, Track] = field(default_factory=dict)
    next_id: int = 1

    @staticmethod
    def iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list[dict], frame_number: int) -> list[dict]:
        assigned: set[int] = set()
        used_tracks: set[int] = set()
        scored: list[tuple[float, int, int]] = []
        for det_index, detection in enumerate(detections):
            for track_id, track in self.tracks.items():
                scored.append((self.iou(detection["bbox"], track.bbox), det_index, track_id))
        scored.sort(reverse=True)
        for score, det_index, track_id in scored:
            if score < self.iou_threshold or det_index in assigned or track_id in used_tracks:
                continue
            detections[det_index]["trackletId"] = f"t{track_id}"
            self.tracks[track_id].bbox = detections[det_index]["bbox"]
            self.tracks[track_id].missed = 0
            self.tracks[track_id].last_frame = frame_number
            assigned.add(det_index)
            used_tracks.add(track_id)
        for det_index, detection in enumerate(detections):
            if det_index in assigned:
                continue
            track_id = self.next_id
            self.next_id += 1
            detection["trackletId"] = f"t{track_id}"
            self.tracks[track_id] = Track(track_id=track_id, bbox=detection["bbox"], last_frame=frame_number)
        expired = []
        for track_id, track in self.tracks.items():
            if track_id not in used_tracks and track_id not in {
                int(item["trackletId"][1:]) for item in detections if item.get("trackletId")
            }:
                track.missed += 1
                if track.missed > self.max_missed:
                    expired.append(track_id)
        for track_id in expired:
            self.tracks.pop(track_id, None)
        return detections
