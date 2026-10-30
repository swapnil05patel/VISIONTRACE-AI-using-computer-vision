from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import EMBEDDING_SPACE, ML_SHARED_SECRET, YOLO_MODEL_LABEL
from faiss_index import vector_index
from inference import (
    detect_people,
    embed_person,
    model_card,
    runtime_status,
)
from preprocess import ImageError, decode_image
from tracking import IoUTracker
from video import encode_jpeg, iter_sampled_frames, probe_video

app = FastAPI(title="VisionTrace ML Runtime", docs_url=None, redoc_url=None)


class VectorBatch(BaseModel):
    vectorIds: list[int] = Field(min_length=1, max_length=5000)
    vectors: list[list[float]] = Field(min_length=1, max_length=5000)
    embeddingSpace: str | None = None


class VectorQuery(BaseModel):
    vector: list[float]
    topK: int = Field(default=10, ge=1, le=1000)


class VectorRemove(BaseModel):
    vectorIds: list[int] = Field(min_length=1, max_length=20000)


class VideoProcessRequest(BaseModel):
    path: str
    sampleFps: float = Field(default=1.0, gt=0, le=8)
    startFrame: int = Field(default=0, ge=0)
    maxFrames: int = Field(default=30, ge=1, le=120)
    track: bool = True


def _authorize(secret: str | None) -> None:
    if not ML_SHARED_SECRET:
        return
    if secret != ML_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Invalid ML runtime credential.")


def _raise(error: Exception) -> None:
    if isinstance(error, ImageError):
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    raise error


@app.middleware("http")
async def require_secret(request: Request, call_next):
    if ML_SHARED_SECRET and request.url.path != "/healthz":
        provided = request.headers.get("x-visiontrace-ml-secret")
        if provided != ML_SHARED_SECRET:
            return JSONResponse({"detail": "Invalid ML runtime credential."}, status_code=401)
    return await call_next(request)


@app.get("/healthz")
def health() -> dict[str, Any]:
    return {"status": "ok", "runtime": runtime_status(vector_index.size), "faissSize": vector_index.size, "model": model_card()}


@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read(32 * 1024 * 1024 + 1)
    try:
        image = decode_image(data)
        detections = detect_people(image)
    except ImageError as error:
        _raise(error)
        raise
    return {"detections": detections, "model": YOLO_MODEL_LABEL}


@app.post("/search-image")
async def search_image(
    file: UploadFile = File(...),
    detection_index: int = Form(..., ge=0),
    top_k: int = Form(default=10, ge=1, le=1000),
) -> dict[str, Any]:
    data = await file.read(32 * 1024 * 1024 + 1)
    try:
        image = decode_image(data)
        detections = detect_people(image)
        if not detections:
            raise HTTPException(status_code=422, detail="No person was detected in the query image.")
        if detection_index >= len(detections):
            raise HTTPException(status_code=400, detail="Selected person detection does not exist.")
        vector = embed_person(image, detections[detection_index])
    except ImageError as error:
        _raise(error)
        raise
    if vector_index.embedding_space and vector_index.embedding_space != EMBEDDING_SPACE:
        raise HTTPException(status_code=409, detail="The FAISS index belongs to a different embedding space.")
    return {
        "queryEmbedding": vector,
        "matches": vector_index.search(vector, top_k),
        "queryDetection": detections[detection_index],
    }


@app.post("/embed-image")
async def embed_image(
    file: UploadFile = File(...),
    detection_index: int = Form(default=0, ge=0),
) -> dict[str, Any]:
    data = await file.read(32 * 1024 * 1024 + 1)
    try:
        image = decode_image(data)
        detections = detect_people(image)
        if not detections:
            raise HTTPException(status_code=422, detail="No person was detected in the query image.")
        if detection_index >= len(detections):
            raise HTTPException(status_code=400, detail="Selected person detection does not exist.")
        vector = embed_person(image, detections[detection_index])
    except ImageError as error:
        _raise(error)
        raise
    return {"vector": vector, "detection": detections[detection_index], "detections": detections}


@app.post("/probe-video")
def probe(body: VideoProcessRequest) -> dict[str, Any]:
    from pathlib import Path

    try:
        return probe_video(Path(body.path))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/process-video-chunk")
def process_video_chunk(body: VideoProcessRequest) -> dict[str, Any]:
    from pathlib import Path

    tracker = IoUTracker() if body.track else None
    frames: list[dict[str, Any]] = []
    last_frame = body.startFrame
    exhausted = True
    try:
        for sampled in iter_sampled_frames(Path(body.path), sample_fps=body.sampleFps, start_frame=body.startFrame):
            last_frame = int(sampled["frameNumber"])
            image = sampled["image"]
            detections = detect_people(image)
            if tracker is not None:
                detections = tracker.update(detections, last_frame)
            people = []
            for detection in detections:
                vector = embed_person(image, detection)
                people.append({**detection, "embedding": vector})
            thumbnail = encode_jpeg(image)
            frames.append(
                {
                    "frameNumber": last_frame,
                    "timestampSeconds": sampled["timestampSeconds"],
                    "thumbnailJpegBase64": __import__("base64").b64encode(thumbnail).decode("ascii"),
                    "detections": people,
                }
            )
            if len(frames) >= body.maxFrames:
                exhausted = False
                break
    except (ImageError, ValueError) as error:
        if isinstance(error, ImageError):
            _raise(error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        "frames": frames,
        "lastFrame": last_frame,
        "exhausted": exhausted,
        "sampleFps": body.sampleFps,
        "embeddingSpace": EMBEDDING_SPACE,
    }


@app.post("/vectors/add")
def add_vectors(batch: VectorBatch) -> dict[str, int]:
    try:
        added = vector_index.add(batch.vectorIds, batch.vectors, batch.embeddingSpace)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"added": added, "size": vector_index.size}


@app.post("/vectors/search")
def search_vectors(query: VectorQuery) -> dict[str, Any]:
    try:
        return {"matches": vector_index.search(query.vector, query.topK)}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/vectors/remove")
def remove_vectors(body: VectorRemove) -> dict[str, int]:
    removed = vector_index.remove(body.vectorIds)
    return {"removed": removed, "size": vector_index.size}


@app.post("/vectors/rebuild")
def rebuild_vectors(batch: VectorBatch) -> dict[str, int]:
    try:
        rebuilt = vector_index.rebuild(batch.vectorIds, batch.vectors, batch.embeddingSpace)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"rebuilt": rebuilt, "size": vector_index.size}
