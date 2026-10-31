# VisionTrace demo test assets

This folder contains a diverse set of synthetic but realistic computer-vision test assets for validating the VisionTrace pipeline.

## Included videos

1. `test_video_cam_A.mp4`  
   Indoor corridor, warm lighting, 2 people, low motion complexity.

2. `test_video_cam_B.mp4`  
   Outdoor plaza, daylight, 3 people, medium complexity.

3. `test_video_cam_C.mp4`  
   Parking lot, low contrast, 4 people, more occlusion.

4. `test_video_multi_person.mp4`  
   Busy lobby, multiple crossing people, crowded scene.

5. `test_video_mixed_scenes.mp4`  
   Stairwell, strong shadows, high contrast, mixed backgrounds.

6. `test_video_person_A.mp4`  
   Close-up tracking, single person and stable follow path.

## Included query images

- `query_person_A.jpg` — clean frontal target for identity match.
- `query_person_B.jpg` — side-view variant with different pose.
- `query_person_C.jpg` — partial occlusion test.
- `query_cam_A_person.jpg` — sample frame from corridor camera.
- `query_cam_B_person.jpg` — sample frame from plaza camera.
- `query_reid_target.jpg` — scene frame with clear Re-ID target.
- `query_crowd_scene.jpg` — crowded scene test.
- `query_multi_person.jpg` — multi-person scene and search stress case.

## Why this set is useful

These files cover:
- indoor and outdoor scenes
- single-person and multi-person tracking
- varied lighting and contrast
- occlusion and crowding
- different camera viewpoints and movement patterns

This gives you a practical demo set for testing:
- detector confidence
- Re-ID matching
- similarity ranking
- multi-camera search behavior
- query robustness under lighting and pose differences

## Recommended test flow

### 1) Verify the ML models first

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
Get-Content .env | Where-Object { $_ -notmatch '^#' -and $_ -match '=' } | ForEach-Object {
  $parts = $_ -split '=', 2
  [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1])
}
python artifacts\api-server\ml\verify_models.py --image "data\test images\query_person_A.jpg"
```

### 2) Run the app

```powershell
$env:PORT = "3000"
$env:BASE_PATH = "/"
pnpm --filter @workspace/visiontrace dev --host 0.0.0.0
```

### 3) Upload or test image search

Use these queries in order:

1. `query_person_A.jpg` — baseline identity search
2. `query_person_B.jpg` — pose variation
3. `query_person_C.jpg` — occlusion / partial visibility
4. `query_crowd_scene.jpg` — crowd test
5. `query_multi_person.jpg` — complex multi-person scene

### 4) Register and index videos

Register these in sequence:

1. `test_video_person_A.mp4` — single-person easy case
2. `test_video_cam_A.mp4` — corridor, 2 persons
3. `test_video_cam_B.mp4` — plaza, 3 persons
4. `test_video_cam_C.mp4` — parking lot, 4 persons
5. `test_video_multi_person.mp4` — dense crowd
6. `test_video_mixed_scenes.mp4` — high-contrast staircase

### 5) Validate expected behavior

Look for:
- proper person detection in each video
- correct tracklet generation within a single video
- image search returning visually similar candidates
- search ranking that favors same clothing / pose / scene context
- no crash on crowd or occluded images

## Regeneration

If you want to rebuild the synthetic assets:

```bash
python generate_realistic_assets.py
```
