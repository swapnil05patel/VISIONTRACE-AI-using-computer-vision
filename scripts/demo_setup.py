#!/usr/bin/env python3
"""
VisionTrace Demo Setup Script
==============================
Bulk-imports local video files into VisionTrace for one-time demo preparation.

Usage:
    .venv/bin/python scripts/demo_setup.py --dir /path/to/demo/videos

Features:
  - Scans a local directory for .mp4/.mov/.webm/.avi files
  - Computes SHA-256 hash of each video to detect duplicates
  - Skips files already registered and indexed in the database
  - Copies new videos into the local object store (data/objects/)
  - Inserts upload record + source_video record + queues video_index job
  - Never re-processes completed videos after restart
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if not os.environ.get("DATABASE_URL"):
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def require_env(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        print(f"[error] {key} is not set. Run: export $(grep -v '^#' .env | xargs)", file=sys.stderr)
        sys.exit(1)
    return val


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


def collect_videos(directory: Path) -> list[Path]:
    videos = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            videos.append(path)
    return videos


def setup_demo(video_dir: Path, owner_id: str, object_root: Path, camera_id, dry_run: bool) -> None:
    try:
        import psycopg2
    except ImportError:
        print("[error] psycopg2 is not installed. Run: .venv/bin/pip install psycopg2-binary")
        sys.exit(1)

    database_url = require_env("DATABASE_URL")
    conn = psycopg2.connect(database_url)
    conn.autocommit = False
    cur = conn.cursor()

    videos = collect_videos(video_dir)
    if not videos:
        print(f"[warn] No video files found in {video_dir}")
        cur.close(); conn.close()
        return

    print(f"[info] Found {len(videos)} video file(s) in {video_dir}")

    cur.execute("SELECT sha256, id, status FROM vision_source_videos WHERE owner_id = %s", (owner_id,))
    existing_by_hash = {row[0]: (row[1], row[2]) for row in cur.fetchall() if row[0]}

    queued = 0
    skipped = 0

    for video_path in videos:
        print(f"\n[video] {video_path.name}")

        file_hash = sha256_path(video_path)
        size_bytes = video_path.stat().st_size
        suffix = video_path.suffix.lower()
        content_type = CONTENT_TYPES.get(suffix, "video/mp4")
        video_name = video_path.stem.replace("_", " ").replace("-", " ").title()

        if file_hash in existing_by_hash:
            vid_id, status = existing_by_hash[file_hash]
            print(f"  [skip] Already in database (id={vid_id[:8]}, status={status})")
            skipped += 1
            continue

        new_uuid = str(uuid.uuid4())
        object_path = f"/objects/uploads/{new_uuid}"

        if dry_run:
            print(f"  [dry-run] Would copy to {object_path} and queue indexing job")
            queued += 1
            continue

        dest = object_root / "uploads" / new_uuid
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [copy] {video_path} -> {dest}")
        shutil.copy2(video_path, dest)

        meta = {"size": size_bytes, "contentType": content_type, "name": video_path.name, "metadata": {"source": "demo_setup"}}
        Path(str(dest) + ".meta.json").write_text(json.dumps(meta))

        try:
            cur.execute(
                "INSERT INTO vision_private_uploads (id, owner_id, object_path, name, content_type, size_bytes, verified, created_at) VALUES (%s, %s, %s, %s, %s, %s, TRUE, NOW()) ON CONFLICT (object_path) DO NOTHING",
                (str(uuid.uuid4()), owner_id, object_path, video_path.name, content_type, size_bytes),
            )
            video_id = str(uuid.uuid4())
            cur.execute(
                "INSERT INTO vision_source_videos (id, owner_id, name, object_path, camera_id, status, sha256, size_bytes, content_type, frame_count, person_count, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, 'queued', %s, %s, %s, 0, 0, NOW(), NOW()) ON CONFLICT DO NOTHING RETURNING id",
                (video_id, owner_id, video_name, object_path, camera_id, file_hash, size_bytes, content_type),
            )
            returned = cur.fetchone()
            if not returned:
                print("  [skip] Conflict on insert - already exists")
                conn.rollback()
                skipped += 1
                continue

            job_id = str(uuid.uuid4())
            cur.execute(
                "INSERT INTO vision_processing_jobs (id, owner_id, kind, status, progress, stage, source_name, video_id, created_at, updated_at) VALUES (%s, %s, 'video_index', 'queued', 0, 'queued', %s, %s, NOW(), NOW())",
                (job_id, owner_id, video_name, video_id),
            )
            conn.commit()
            print(f"  [ok] video_id={video_id[:8]}, job_id={job_id[:8]}")
            queued += 1
        except Exception as exc:
            conn.rollback()
            print(f"  [error] {exc}", file=sys.stderr)
            try:
                dest.unlink(missing_ok=True)
                Path(str(dest) + ".meta.json").unlink(missing_ok=True)
            except Exception:
                pass

    cur.close()
    conn.close()
    print(f"\n[done] Queued={queued}, Skipped={skipped}")
    if queued and not dry_run:
        print("\n[next] Start services to begin processing:")
        print("  Terminal 1 (FastAPI ML):  .venv/bin/python -m uvicorn app:app --app-dir artifacts/api-server/ml --host 127.0.0.1 --port 18777 --workers 1")
        print("  Terminal 2 (Express API): export $(grep -v '^#' .env | xargs) && pnpm --filter @workspace/api-server run dev")
        print("  Terminal 3 (Frontend):    PORT=3000 BASE_PATH=/ pnpm --filter @workspace/visiontrace run dev")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionTrace demo setup: bulk-import videos.")
    parser.add_argument("--dir", required=True, type=Path, help="Directory with video files")
    parser.add_argument("--owner", default=None, help="Owner user ID (default: VISIONTRACE_DEV_USER_ID)")
    parser.add_argument("--camera", default=None, help="Camera ID label for imported videos")
    parser.add_argument("--object-dir", default=None, type=Path, help="Object storage root (default: data/objects)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    args = parser.parse_args()

    owner_id = args.owner or os.environ.get("VISIONTRACE_DEV_USER_ID", "dev-user")
    object_root = args.object_dir or (PROJECT_ROOT / "data" / "objects")
    video_dir = args.dir.resolve()

    if not video_dir.is_dir():
        print(f"[error] Not a directory: {video_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("VisionTrace Demo Setup")
    print(f"  Video directory : {video_dir}")
    print(f"  Owner ID        : {owner_id}")
    print(f"  Object storage  : {object_root}")
    print(f"  Camera ID       : {args.camera or '(none)'}")
    print(f"  Dry run         : {args.dry_run}")
    print("=" * 60)

    setup_demo(video_dir, owner_id, object_root, args.camera, args.dry_run)


if __name__ == "__main__":
    main()
