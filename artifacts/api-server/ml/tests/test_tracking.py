from tracking import IoUTracker


def test_iou_associates_overlapping_boxes_within_one_video() -> None:
    tracker = IoUTracker()
    first = tracker.update([{"bbox": [10, 10, 50, 80], "detectionIndex": 0}], 0)
    second = tracker.update([{"bbox": [12, 12, 52, 82], "detectionIndex": 0}], 1)
    assert first[0]["trackletId"] == second[0]["trackletId"]


def test_track_ids_restart_per_tracker_instance() -> None:
    first = IoUTracker().update([{"bbox": [0, 0, 20, 40]}], 0)
    second = IoUTracker().update([{"bbox": [0, 0, 20, 40]}], 0)
    assert first[0]["trackletId"] == "t1"
    assert second[0]["trackletId"] == "t1"
