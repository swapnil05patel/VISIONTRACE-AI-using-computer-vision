import os
import sys
import cv2

sys.path.insert(0, os.path.join(os.getcwd(), 'artifacts', 'api-server', 'ml'))
from inference import detect_people

names = ['test_video_person_A.mp4', 'test_video_multi_person.mp4', 'test_video_mixed_scenes.mp4']
for name in names:
    path = os.path.join('data', 'test images', name)
    cap = cv2.VideoCapture(path)
    total = 0
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += len(detect_people(frame, require_verification=False))
        frames += 1
    cap.release()
    print(name, 'frames=', frames, 'total_dets=', total, 'avg=', (total / frames if frames else 0))
