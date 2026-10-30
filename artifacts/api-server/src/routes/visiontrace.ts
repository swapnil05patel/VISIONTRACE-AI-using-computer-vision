import { and, count, desc, eq, inArray, isNotNull } from "drizzle-orm";
import {
  DetectPeopleInImageBody,
  DetectPeopleInImageResponse,
  GetProcessingJobBody,
  GetProcessingJobResponse,
  GetVideoBody,
  GetVideoResponse,
  GetVisionTraceOverviewResponse,
  ListVideoFramesBody,
  ListVideoFramesResponse,
  ListVideosResponse,
  RegisterVideoBody,
  RegisterVideoResponse,
  SearchFilters,
  SearchImageBody,
  SearchImageResponse,
  SearchVideoBody,
  StartDatasetImportBody,
  StartDatasetImportResponse,
  VideoFramesInput,
} from "@workspace/api-zod";
import {
  completedSearchesTable,
  db,
  indexedFramesTable,
  personDetectionsTable,
  processingJobsTable,
  modelActivationHistoryTable,
  modelArtifactsTable,
  reidEmbeddingsTable,
  sourceVideosTable,
} from "@workspace/db";
import { Router, type IRouter, type Request, type Response } from "express";
import { authenticatedUserId, requireAdmin } from "../lib/clerkAuth";
import { listDatasetAdapters, getDatasetRoot } from "../lib/datasetAdapters";
import { callMl, getVisionRuntime } from "../lib/visionRuntime";
import { getOwnedUpload } from "./storage";

const router: IRouter = Router();

function requestUser(req: Request, res: Response): string | null {
  return authenticatedUserId(req, res);
}

function serializeVideo(video: typeof sourceVideosTable.$inferSelect) {
  return {
    ...video,
    durationSeconds: video.durationSeconds ?? null,
    cameraId: video.cameraId ?? null,
  };
}

function serializeJob(job: typeof processingJobsTable.$inferSelect) {
  return {
    ...job,
    sourceName: job.sourceName ?? null,
    error: job.error ?? null,
    result: job.result ?? null,
  };
}

function filtersPass(
  filter: SearchFilters | undefined,
  item: {
    cameraId: string | null;
    recordedAt: Date | null;
    timestampSeconds: number;
  },
): boolean {
  if (!filter) return true;
  if (filter.cameraId && filter.cameraId !== item.cameraId) return false;
  const absoluteTime = item.recordedAt
    ? new Date(item.recordedAt.getTime() + item.timestampSeconds * 1000)
    : null;
  if ((filter.from || filter.to || filter.timeFrom || filter.timeTo) && !absoluteTime) {
    return false;
  }
  if (absoluteTime && filter.from && absoluteTime < new Date(filter.from)) return false;
  if (absoluteTime && filter.to && absoluteTime > new Date(filter.to)) return false;
  if (absoluteTime && filter.timeFrom) {
    const time = `${String(absoluteTime.getUTCHours()).padStart(2, "0")}:${String(
      absoluteTime.getUTCMinutes(),
    ).padStart(2, "0")}`;
    if (time < filter.timeFrom) return false;
  }
  if (absoluteTime && filter.timeTo) {
    const time = `${String(absoluteTime.getUTCHours()).padStart(2, "0")}:${String(
      absoluteTime.getUTCMinutes(),
    ).padStart(2, "0")}`;
    if (time > filter.timeTo) return false;
  }
  return true;
}

function runtimeError(runtime: Awaited<ReturnType<typeof getVisionRuntime>>): string {
  return runtime.limitations.length
    ? runtime.limitations.join(" ")
    : "Required computer-vision models are not ready.";
}

async function imageBytes(
  ownerId: string,
  objectPath: string,
): Promise<{ bytes: Buffer; name: string } | null> {
  const result = await getOwnedUpload(ownerId, objectPath, "image");
  if (!result.ok) return null;
  const [bytes] = await result.file.download();
  const [metadata] = await result.file.getMetadata();
  return { bytes, name: String(metadata.name ?? "query-image.jpg") };
}

router.get(
  "/visiontrace/runtime",
  async (_req: Request, res: Response): Promise<void> => {
    res.json(await getVisionRuntime());
  },
);

router.get(
  "/visiontrace/overview",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const [
      [videoCount],
      [frameCount],
      [personCount],
      [activeJobCount],
      [searchCount],
      recentJobs,
      datasets,
      runtime,
    ] = await Promise.all([
      db.select({ value: count() }).from(sourceVideosTable).where(eq(sourceVideosTable.ownerId, ownerId)),
      db
        .select({ value: count() })
        .from(indexedFramesTable)
        .innerJoin(sourceVideosTable, eq(indexedFramesTable.videoId, sourceVideosTable.id))
        .where(eq(sourceVideosTable.ownerId, ownerId)),
      db
        .select({ value: count() })
        .from(personDetectionsTable)
        .innerJoin(indexedFramesTable, eq(personDetectionsTable.frameId, indexedFramesTable.id))
        .innerJoin(sourceVideosTable, eq(indexedFramesTable.videoId, sourceVideosTable.id))
        .where(eq(sourceVideosTable.ownerId, ownerId)),
      db
        .select({ value: count() })
        .from(processingJobsTable)
        .where(
          and(
            eq(processingJobsTable.ownerId, ownerId),
            inArray(processingJobsTable.status, ["queued", "running"]),
          ),
        ),
      db
        .select({ value: count() })
        .from(completedSearchesTable)
        .where(eq(completedSearchesTable.ownerId, ownerId)),
      db
        .select()
        .from(processingJobsTable)
        .where(eq(processingJobsTable.ownerId, ownerId))
        .orderBy(desc(processingJobsTable.createdAt))
        .limit(5),
      listDatasetAdapters(ownerId),
      getVisionRuntime(),
    ]);

    res.json(
      GetVisionTraceOverviewResponse.parse({
        indexedVideos: Number(videoCount?.value ?? 0),
        indexedFrames: Number(frameCount?.value ?? 0),
        indexedPeople: Number(personCount?.value ?? 0),
        activeJobs: Number(activeJobCount?.value ?? 0),
        completedSearches: Number(searchCount?.value ?? 0),
        runtime,
        datasets,
        recentJobs: recentJobs.map(serializeJob),
      }),
    );
  },
);

router.get(
  "/visiontrace/datasets",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    res.json(await listDatasetAdapters(ownerId));
  },
);

router.post(
  "/visiontrace/datasets/imports",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const parsed = StartDatasetImportBody.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid dataset import request." });
      return;
    }
    const { datasetId, split, purpose, maxItems } = parsed.data;
    if (!getDatasetRoot(datasetId)) {
      res.status(409).json({
        error: `No authorized local copy of ${datasetId} is configured. Dataset downloads are not automatic.`,
      });
      return;
    }

    const [job] = await db
      .insert(processingJobsTable)
      .values({
        ownerId,
        kind: "dataset_import",
        status: "queued",
        progress: 0,
        stage: "queued",
        datasetId,
        sourceName: `${datasetId} · ${split} · ${purpose} (limit ${maxItems})`,
      })
      .returning();
    res.status(202).json(StartDatasetImportResponse.parse(serializeJob(job)));
  },
);

router.get(
  "/visiontrace/videos",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const videos = await db
      .select()
      .from(sourceVideosTable)
      .where(eq(sourceVideosTable.ownerId, ownerId))
      .orderBy(desc(sourceVideosTable.createdAt));
    res.json(ListVideosResponse.parse(videos.map(serializeVideo)));
  },
);

router.post(
  "/visiontrace/videos",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const parsed = RegisterVideoBody.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid video metadata." });
      return;
    }
    const body = parsed.data;
    const object = await getOwnedUpload(ownerId, body.objectPath, "video");
    if (!object.ok) {
      res.status(object.status).json({ error: object.error });
      return;
    }
    const [existing] = await db
      .select({ id: sourceVideosTable.id })
      .from(sourceVideosTable)
      .where(
        and(
          eq(sourceVideosTable.ownerId, ownerId),
          eq(sourceVideosTable.objectPath, body.objectPath),
        ),
      )
      .limit(1);
    if (existing) {
      res.status(409).json({ error: "This uploaded video is already registered." });
      return;
    }

    const runtime = await getVisionRuntime();
    if (runtime.status !== "ready") {
      res.status(503).json({ error: runtimeError(runtime) });
      return;
    }
    const now = new Date();
    try {
      const result = await db.transaction(async (tx) => {
        const [video] = await tx
          .insert(sourceVideosTable)
          .values({
            ownerId,
            name: body.name,
            objectPath: body.objectPath,
            cameraId: body.cameraId ?? null,
            recordedAt: body.recordedAt ? new Date(body.recordedAt) : null,
            status: "queued",
          })
          .returning();
        const [job] = await tx
          .insert(processingJobsTable)
          .values({
            ownerId,
            kind: "video_index",
            status: "queued",
            progress: 0,
            stage: "queued",
            sourceName: body.name,
            videoId: video.id,
            createdAt: now,
            updatedAt: now,
          })
          .returning();
        return { video, job };
      });
      res.status(202).json(
        RegisterVideoResponse.parse({
          video: serializeVideo(result.video),
          job: serializeJob(result.job),
        }),
      );
    } catch (error) {
      req.log.error({ err: error }, "Could not register source video");
      res.status(500).json({ error: "Could not register the source video." });
    }
  },
);

router.post(
  "/visiontrace/videos/details",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const parsed = GetVideoBody.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid video ID." });
      return;
    }
    const [video] = await db
      .select()
      .from(sourceVideosTable)
      .where(
        and(
          eq(sourceVideosTable.id, parsed.data.videoId),
          eq(sourceVideosTable.ownerId, ownerId),
        ),
      )
      .limit(1);
    if (!video) {
      res.status(404).json({ error: "Video not found." });
      return;
    }
    res.json(GetVideoResponse.parse(serializeVideo(video)));
  },
);

router.post(
  "/visiontrace/videos/frames",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const parsed = ListVideoFramesBody.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid video frame request." });
      return;
    }
    const body = parsed.data as VideoFramesInput;
    const [video] = await db
      .select({ id: sourceVideosTable.id })
      .from(sourceVideosTable)
      .where(
        and(
          eq(sourceVideosTable.id, body.videoId),
          eq(sourceVideosTable.ownerId, ownerId),
        ),
      )
      .limit(1);
    if (!video) {
      res.status(404).json({ error: "Video not found." });
      return;
    }
    const frames = await db
      .select()
      .from(indexedFramesTable)
      .where(eq(indexedFramesTable.videoId, body.videoId))
      .orderBy(indexedFramesTable.timestampSeconds)
      .limit(body.limit ?? 50);
    const detections = frames.length
      ? await db
          .select()
          .from(personDetectionsTable)
          .where(inArray(personDetectionsTable.frameId, frames.map((frame) => frame.id)))
      : [];
    const byFrame = new Map<string, typeof detections>();
    for (const detection of detections) {
      byFrame.set(detection.frameId, [
        ...(byFrame.get(detection.frameId) ?? []),
        detection,
      ]);
    }
    res.json(
      ListVideoFramesResponse.parse(
        frames.map((frame) => ({
          id: frame.id,
          videoId: frame.videoId,
          frameNumber: frame.frameNumber,
          timestampSeconds: frame.timestampSeconds,
          thumbnailPath: frame.thumbnailPath ?? null,
          detections: (byFrame.get(frame.id) ?? []).map((detection) => ({
            id: detection.id,
            detectionIndex: detection.detectionIndex,
            bbox: detection.bbox,
            detectorScore: detection.detectorScore,
            cropPath: detection.cropPath ?? null,
            trackletId: detection.trackletId ?? null,
          })),
        })),
      ),
    );
  },
);

router.get(
  "/visiontrace/cameras",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const rows = await db
      .select({
        id: sourceVideosTable.cameraId,
        videoCount: count(),
      })
      .from(sourceVideosTable)
      .where(
        and(
          eq(sourceVideosTable.ownerId, ownerId),
          isNotNull(sourceVideosTable.cameraId),
        ),
      )
      .groupBy(sourceVideosTable.cameraId);
    res.json(
      rows
        .filter((row): row is { id: string; videoCount: number } => Boolean(row.id))
        .map((row) => ({ id: row.id, videoCount: Number(row.videoCount) })),
    );
  },
);

router.post(
  "/visiontrace/images/detections",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const parsed = DetectPeopleInImageBody.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid image upload reference." });
      return;
    }
    const image = await imageBytes(ownerId, parsed.data.objectPath);
    if (!image) {
      res.status(404).json({ error: "Owned, valid query image was not found." });
      return;
    }
    const form = new FormData();
    form.append("file", new Blob([new Uint8Array(image.bytes)]), image.name);
    const result = await callMl<{
      detections: unknown[];
      model: string;
    }>("/detect-image", { method: "POST", body: form });
    if (!result.ok) {
      res.status(result.status).json({ error: result.error });
      return;
    }
    res.json(DetectPeopleInImageResponse.parse(result.data));
  },
);

router.post(
  "/visiontrace/search/image",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const parsed = SearchImageBody.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid image search request." });
      return;
    }
    const image = await imageBytes(ownerId, parsed.data.objectPath);
    if (!image) {
      res.status(404).json({ error: "Owned, valid query image was not found." });
      return;
    }
    const runtime = await getVisionRuntime();
    if (runtime.status !== "ready") {
      res.status(503).json({ error: runtimeError(runtime) });
      return;
    }
    const form = new FormData();
    form.append("file", new Blob([new Uint8Array(image.bytes)]), image.name);
    form.append("detection_index", String(parsed.data.detectionIndex));
    form.append(
      "top_k",
      String(Math.min(1000, (parsed.data.filters?.topK ?? 10) * 10)),
    );
    const mlResult = await callMl<{
      matches: { vectorId: number; similarity: number }[];
    }>("/search-image", { method: "POST", body: form });
    if (!mlResult.ok) {
      res.status(mlResult.status).json({ error: mlResult.error });
      return;
    }

    const matches = mlResult.data.matches;
    const rows = matches.length
      ? await db
          .select({
            vectorId: reidEmbeddingsTable.vectorId,
            videoId: sourceVideosTable.id,
            sourceName: sourceVideosTable.name,
            cameraId: sourceVideosTable.cameraId,
            recordedAt: sourceVideosTable.recordedAt,
            objectPath: sourceVideosTable.objectPath,
            frameNumber: indexedFramesTable.frameNumber,
            timestampSeconds: indexedFramesTable.timestampSeconds,
            thumbnailPath: indexedFramesTable.thumbnailPath,
          })
          .from(reidEmbeddingsTable)
          .innerJoin(
            personDetectionsTable,
            eq(reidEmbeddingsTable.detectionId, personDetectionsTable.id),
          )
          .innerJoin(
            indexedFramesTable,
            eq(personDetectionsTable.frameId, indexedFramesTable.id),
          )
          .innerJoin(
            sourceVideosTable,
            eq(indexedFramesTable.videoId, sourceVideosTable.id),
          )
          .where(
            and(
              eq(sourceVideosTable.ownerId, ownerId),
              inArray(
                reidEmbeddingsTable.vectorId,
                matches.map((match) => match.vectorId),
              ),
            ),
          )
      : [];
    const matchById = new Map(matches.map((match) => [match.vectorId, match]));
    const hits = rows
      .filter((row) =>
        filtersPass(parsed.data.filters, {
          cameraId: row.cameraId,
          recordedAt: row.recordedAt,
          timestampSeconds: row.timestampSeconds,
        }),
      )
      .map((row) => {
        const match = matchById.get(row.vectorId);
        return {
          id: crypto.randomUUID(),
          videoId: row.videoId,
          sourceName: row.sourceName,
          cameraId: row.cameraId,
          frameNumber: row.frameNumber,
          timestampSeconds: row.timestampSeconds,
          similarity: match?.similarity ?? 0,
          thumbnailPath: row.thumbnailPath,
          videoPath: `/api/storage/objects/${row.objectPath.replace(/^\/objects\//, "")}`,
        };
      })
      .sort((left, right) => right.similarity - left.similarity)
      .slice(0, parsed.data.filters?.topK ?? 10);
    await db.insert(completedSearchesTable).values({
      ownerId,
      mode: "image",
      queryObjectPath: parsed.data.objectPath,
      resultCount: hits.length,
    });
    res.json(
      SearchImageResponse.parse({
        queryMode: "image",
        results: hits,
        resultCount: hits.length,
        rankingNote:
          "Scores are raw cosine similarities from normalized embeddings, not identity probabilities.",
      }),
    );
  },
);

router.post(
  "/visiontrace/search/video",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const parsed = SearchVideoBody.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid video search request." });
      return;
    }
    const video = await getOwnedUpload(ownerId, parsed.data.objectPath, "video");
    if (!video.ok) {
      res.status(video.status).json({ error: video.error });
      return;
    }
    const runtime = await getVisionRuntime();
    if (runtime.status !== "ready") {
      res.status(503).json({ error: runtimeError(runtime) });
      return;
    }
    const [job] = await db
      .insert(processingJobsTable)
      .values({
        ownerId,
        kind: "video_search",
        status: "queued",
        progress: 0,
        stage: "queued",
        sourceName: parsed.data.objectPath,
      })
      .returning();
    res.status(202).json(serializeJob(job));
  },
);

router.post(
  "/visiontrace/jobs/status",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const parsed = GetProcessingJobBody.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid job ID." });
      return;
    }
    const [job] = await db
      .select()
      .from(processingJobsTable)
      .where(
        and(
          eq(processingJobsTable.id, parsed.data.jobId),
          eq(processingJobsTable.ownerId, ownerId),
        ),
      )
      .limit(1);
    if (!job) {
      res.status(404).json({ error: "Processing job not found." });
      return;
    }
    res.json(GetProcessingJobResponse.parse(serializeJob(job)));
  },
);

router.post(
  "/visiontrace/jobs/cancel",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const parsed = GetProcessingJobBody.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid job ID." });
      return;
    }
    const [job] = await db
      .update(processingJobsTable)
      .set({ status: "failed", stage: "cancelled", error: "Cancelled by the owner." })
      .where(
        and(
          eq(processingJobsTable.id, parsed.data.jobId),
          eq(processingJobsTable.ownerId, ownerId),
          inArray(processingJobsTable.status, ["queued", "running"]),
        ),
      )
      .returning();
    if (!job) {
      res.status(404).json({ error: "No cancellable job was found." });
      return;
    }
    res.json(serializeJob(job));
  },
);

router.post(
  "/visiontrace/search/tracklet",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const trackletId = typeof req.body?.trackletId === "string" ? req.body.trackletId : "";
    const topK = Math.min(100, Number(req.body?.topK) || 10);
    if (!trackletId) {
      res.status(400).json({ error: "A video-scoped trackletId is required." });
      return;
    }
    const runtime = await getVisionRuntime();
    if (runtime.status !== "ready") {
      res.status(503).json({ error: runtimeError(runtime) });
      return;
    }
    const [seed] = await db
      .select({
        vector: reidEmbeddingsTable.vector,
        ownerId: sourceVideosTable.ownerId,
      })
      .from(reidEmbeddingsTable)
      .innerJoin(personDetectionsTable, eq(reidEmbeddingsTable.detectionId, personDetectionsTable.id))
      .innerJoin(indexedFramesTable, eq(personDetectionsTable.frameId, indexedFramesTable.id))
      .innerJoin(sourceVideosTable, eq(indexedFramesTable.videoId, sourceVideosTable.id))
      .where(and(eq(personDetectionsTable.trackletId, trackletId), eq(sourceVideosTable.ownerId, ownerId)))
      .limit(1);
    if (!seed) {
      res.status(404).json({ error: "That tracklet was not found in your indexed footage." });
      return;
    }
    const searched = await callMl<{ matches: { vectorId: number; similarity: number }[] }>(
      "/vectors/search",
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ vector: seed.vector, topK: topK * 10 }),
      },
    );
    if (!searched.ok) {
      res.status(searched.status).json({ error: searched.error });
      return;
    }
    req.body = {
      objectPath: "tracklet",
      detectionIndex: 0,
      filters: { topK },
    };
    const matches = searched.data.matches;
    const rows = matches.length
      ? await db
          .select({
            vectorId: reidEmbeddingsTable.vectorId,
            videoId: sourceVideosTable.id,
            sourceName: sourceVideosTable.name,
            cameraId: sourceVideosTable.cameraId,
            objectPath: sourceVideosTable.objectPath,
            frameNumber: indexedFramesTable.frameNumber,
            timestampSeconds: indexedFramesTable.timestampSeconds,
            thumbnailPath: indexedFramesTable.thumbnailPath,
          })
          .from(reidEmbeddingsTable)
          .innerJoin(personDetectionsTable, eq(reidEmbeddingsTable.detectionId, personDetectionsTable.id))
          .innerJoin(indexedFramesTable, eq(personDetectionsTable.frameId, indexedFramesTable.id))
          .innerJoin(sourceVideosTable, eq(indexedFramesTable.videoId, sourceVideosTable.id))
          .where(
            and(
              eq(sourceVideosTable.ownerId, ownerId),
              inArray(
                reidEmbeddingsTable.vectorId,
                matches.map((match) => match.vectorId),
              ),
            ),
          )
      : [];
    const matchById = new Map(matches.map((match) => [match.vectorId, match]));
    const results = rows
      .map((row) => ({
        id: crypto.randomUUID(),
        videoId: row.videoId,
        sourceName: row.sourceName,
        cameraId: row.cameraId,
        frameNumber: row.frameNumber,
        timestampSeconds: row.timestampSeconds,
        similarity: matchById.get(row.vectorId)?.similarity ?? 0,
        thumbnailPath: row.thumbnailPath,
        videoPath: `/api/storage/objects/${row.objectPath.replace(/^\/objects\//, "")}`,
      }))
      .sort((left, right) => right.similarity - left.similarity)
      .slice(0, topK);
    res.json({
      queryMode: "video",
      results,
      resultCount: results.length,
      rankingNote:
        "Tracklet search ranks visually similar appearances. A track ID is not a real-world identity.",
    });
  },
);

router.post(
  "/visiontrace/videos/delete",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const parsed = GetVideoBody.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid video ID." });
      return;
    }
    const [video] = await db
      .select()
      .from(sourceVideosTable)
      .where(and(eq(sourceVideosTable.id, parsed.data.videoId), eq(sourceVideosTable.ownerId, ownerId)))
      .limit(1);
    if (!video) {
      res.status(404).json({ error: "Video not found." });
      return;
    }
    const vectorIds = await db
      .select({ vectorId: reidEmbeddingsTable.vectorId })
      .from(reidEmbeddingsTable)
      .innerJoin(personDetectionsTable, eq(reidEmbeddingsTable.detectionId, personDetectionsTable.id))
      .innerJoin(indexedFramesTable, eq(personDetectionsTable.frameId, indexedFramesTable.id))
      .where(eq(indexedFramesTable.videoId, video.id));
    if (vectorIds.length) {
      await callMl("/vectors/remove", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ vectorIds: vectorIds.map((row) => row.vectorId) }),
      });
    }
    await db.delete(sourceVideosTable).where(eq(sourceVideosTable.id, video.id));
    res.json({ ok: true });
  },
);

router.get("/visiontrace/models", async (req: Request, res: Response): Promise<void> => {
  const admin = requireAdmin(req, res);
  if (!admin) return;
  const models = await db.select().from(modelArtifactsTable).orderBy(desc(modelArtifactsTable.createdAt));
  res.json(models);
});

router.post("/visiontrace/models/activate", async (req: Request, res: Response): Promise<void> => {
  const admin = requireAdmin(req, res);
  if (!admin) return;
  const modelId = typeof req.body?.modelId === "string" ? req.body.modelId : "";
  const [model] = await db.select().from(modelArtifactsTable).where(eq(modelArtifactsTable.id, modelId)).limit(1);
  if (!model) {
    res.status(404).json({ error: "Model artifact was not found." });
    return;
  }
  if (!model.verified || (model.status !== "verified" && model.status !== "active")) {
    res.status(409).json({ error: "Only verified models can become active." });
    return;
  }
  const [previous] = await db
    .select()
    .from(modelArtifactsTable)
    .where(and(eq(modelArtifactsTable.type, model.type), eq(modelArtifactsTable.active, true)))
    .limit(1);
  await db.update(modelArtifactsTable).set({ active: false, status: "retired" }).where(
    and(eq(modelArtifactsTable.type, model.type), eq(modelArtifactsTable.active, true)),
  );
  await db.update(modelArtifactsTable).set({ active: true, status: "active" }).where(eq(modelArtifactsTable.id, model.id));
  await db.insert(modelActivationHistoryTable).values({
    modelId: model.id,
    action: "activate",
    actorId: admin,
    previousModelId: previous?.id,
  });
  res.json({ ok: true, activeId: model.id, previousId: previous?.id ?? null });
});

router.post("/visiontrace/models/rollback", async (req: Request, res: Response): Promise<void> => {
  const admin = requireAdmin(req, res);
  if (!admin) return;
  const [latest] = await db
    .select()
    .from(modelActivationHistoryTable)
    .orderBy(desc(modelActivationHistoryTable.createdAt))
    .limit(1);
  if (!latest?.previousModelId) {
    res.status(409).json({ error: "No previous active model is recorded." });
    return;
  }
  req.body = { modelId: latest.previousModelId };
  const [model] = await db.select().from(modelArtifactsTable).where(eq(modelArtifactsTable.id, latest.previousModelId)).limit(1);
  if (!model?.verified) {
    res.status(409).json({ error: "The previous model is no longer verified." });
    return;
  }
  await db.update(modelArtifactsTable).set({ active: false, status: "retired" }).where(eq(modelArtifactsTable.active, true));
  await db.update(modelArtifactsTable).set({ active: true, status: "active" }).where(eq(modelArtifactsTable.id, model.id));
  await db.insert(modelActivationHistoryTable).values({
    modelId: model.id,
    action: "rollback",
    actorId: admin,
    previousModelId: latest.modelId,
  });
  res.json({ ok: true, activeId: model.id });
});

// ─── Detected People Gallery ──────────────────────────────────────────────
// GET /api/visiontrace/gallery
// Returns person detections with source video info, paginated, grouped per video.
// Query params: page (1-based), pageSize (max 100), videoId (optional filter)
router.get(
  "/visiontrace/gallery",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;

    const page = Math.max(1, parseInt(String(req.query.page ?? "1"), 10) || 1);
    const pageSize = Math.min(100, Math.max(1, parseInt(String(req.query.pageSize ?? "48"), 10) || 48));
    const videoIdFilter = typeof req.query.videoId === "string" && req.query.videoId ? req.query.videoId : null;
    const offset = (page - 1) * pageSize;

    const baseWhere = and(
      eq(sourceVideosTable.ownerId, ownerId),
      eq(sourceVideosTable.status, "indexed"),
      videoIdFilter ? eq(sourceVideosTable.id, videoIdFilter) : undefined,
    );

    const [countRow] = await db
      .select({ total: count() })
      .from(personDetectionsTable)
      .innerJoin(indexedFramesTable, eq(personDetectionsTable.frameId, indexedFramesTable.id))
      .innerJoin(sourceVideosTable, eq(indexedFramesTable.videoId, sourceVideosTable.id))
      .where(baseWhere);

    const rows = await db
      .select({
        detectionId: personDetectionsTable.id,
        detectionIndex: personDetectionsTable.detectionIndex,
        bbox: personDetectionsTable.bbox,
        detectorScore: personDetectionsTable.detectorScore,
        cropPath: personDetectionsTable.cropPath,
        trackletId: personDetectionsTable.trackletId,
        frameNumber: indexedFramesTable.frameNumber,
        timestampSeconds: indexedFramesTable.timestampSeconds,
        thumbnailPath: indexedFramesTable.thumbnailPath,
        videoId: sourceVideosTable.id,
        videoName: sourceVideosTable.name,
        cameraId: sourceVideosTable.cameraId,
        objectPath: sourceVideosTable.objectPath,
        frameId: indexedFramesTable.id,
      })
      .from(personDetectionsTable)
      .innerJoin(indexedFramesTable, eq(personDetectionsTable.frameId, indexedFramesTable.id))
      .innerJoin(sourceVideosTable, eq(indexedFramesTable.videoId, sourceVideosTable.id))
      .where(baseWhere)
      .orderBy(desc(sourceVideosTable.createdAt), indexedFramesTable.frameNumber, personDetectionsTable.detectionIndex)
      .limit(pageSize)
      .offset(offset);

    const total = Number(countRow?.total ?? 0);
    res.json({
      detections: rows.map((r) => ({
        id: r.detectionId,
        detectionIndex: r.detectionIndex,
        bbox: r.bbox,
        detectorScore: r.detectorScore,
        cropPath: r.cropPath ?? null,
        trackletId: r.trackletId ?? null,
        frameNumber: r.frameNumber,
        timestampSeconds: r.timestampSeconds,
        thumbnailPath: r.thumbnailPath ?? null,
        videoId: r.videoId,
        videoName: r.videoName,
        cameraId: r.cameraId ?? null,
        videoPath: `/api/storage/objects/${r.objectPath.replace(/^\/objects\//, "")}`,
        frameId: r.frameId,
      })),
      total,
      page,
      pageSize,
      pages: Math.ceil(total / pageSize),
    });
  },
);

// ─── Indexed Videos list for gallery filter ───────────────────────────────
router.get(
  "/visiontrace/gallery/videos",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;
    const videos = await db
      .select({
        id: sourceVideosTable.id,
        name: sourceVideosTable.name,
        cameraId: sourceVideosTable.cameraId,
        personCount: sourceVideosTable.personCount,
        frameCount: sourceVideosTable.frameCount,
        durationSeconds: sourceVideosTable.durationSeconds,
        status: sourceVideosTable.status,
      })
      .from(sourceVideosTable)
      .where(and(eq(sourceVideosTable.ownerId, ownerId), eq(sourceVideosTable.status, "indexed")))
      .orderBy(desc(sourceVideosTable.createdAt));
    res.json({ videos });
  },
);

// ─── One-click Visual Search from Gallery Detection ───────────────────────
// POST /api/visiontrace/search/detection
// Body: { detectionId: string, topK?: number }
// Uses the stored 512-d embedding from the selected detection to run FAISS search.
router.post(
  "/visiontrace/search/detection",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;

    const detectionId = typeof req.body?.detectionId === "string" ? req.body.detectionId : "";
    const topK = Math.min(100, Math.max(1, Number(req.body?.topK) || 20));

    if (!detectionId) {
      res.status(400).json({ error: "detectionId is required." });
      return;
    }

    const runtime = await getVisionRuntime();
    if (runtime.status !== "ready") {
      res.status(503).json({ error: runtimeError(runtime) });
      return;
    }

    // Load the stored embedding for this detection
    const [seed] = await db
      .select({
        vector: reidEmbeddingsTable.vector,
        embeddingSpace: reidEmbeddingsTable.embeddingSpace,
        detectionIndex: personDetectionsTable.detectionIndex,
        bbox: personDetectionsTable.bbox,
        ownerId: sourceVideosTable.ownerId,
      })
      .from(reidEmbeddingsTable)
      .innerJoin(personDetectionsTable, eq(reidEmbeddingsTable.detectionId, personDetectionsTable.id))
      .innerJoin(indexedFramesTable, eq(personDetectionsTable.frameId, indexedFramesTable.id))
      .innerJoin(sourceVideosTable, eq(indexedFramesTable.videoId, sourceVideosTable.id))
      .where(and(eq(personDetectionsTable.id, detectionId), eq(sourceVideosTable.ownerId, ownerId)))
      .limit(1);

    if (!seed) {
      res.status(404).json({ error: "Detection not found or not authorized." });
      return;
    }

    // FAISS vector search using stored embedding
    const searched = await callMl<{ matches: { vectorId: number; similarity: number }[] }>(
      "/vectors/search",
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ vector: seed.vector, topK: topK * 10 }),
      },
      30_000,
    );

    if (!searched.ok) {
      res.status(searched.status).json({ error: searched.error });
      return;
    }

    const matches = searched.data.matches;
    if (!matches.length) {
      res.json({ results: [], resultCount: 0, queryMode: "detection", rankingNote: "No matches found in the current FAISS index." });
      return;
    }

    const rows = await db
      .select({
        vectorId: reidEmbeddingsTable.vectorId,
        videoId: sourceVideosTable.id,
        sourceName: sourceVideosTable.name,
        cameraId: sourceVideosTable.cameraId,
        objectPath: sourceVideosTable.objectPath,
        frameNumber: indexedFramesTable.frameNumber,
        timestampSeconds: indexedFramesTable.timestampSeconds,
        thumbnailPath: indexedFramesTable.thumbnailPath,
        detectionId: personDetectionsTable.id,
        bbox: personDetectionsTable.bbox,
        detectorScore: personDetectionsTable.detectorScore,
        trackletId: personDetectionsTable.trackletId,
        recordedAt: sourceVideosTable.recordedAt,
      })
      .from(reidEmbeddingsTable)
      .innerJoin(personDetectionsTable, eq(reidEmbeddingsTable.detectionId, personDetectionsTable.id))
      .innerJoin(indexedFramesTable, eq(personDetectionsTable.frameId, indexedFramesTable.id))
      .innerJoin(sourceVideosTable, eq(indexedFramesTable.videoId, sourceVideosTable.id))
      .where(
        and(
          eq(sourceVideosTable.ownerId, ownerId),
          inArray(reidEmbeddingsTable.vectorId, matches.map((m) => m.vectorId)),
        ),
      );

    const scoreById = new Map(matches.map((m) => [m.vectorId, m.similarity]));

    // Temporal window collapse (±1.5 seconds per video)
    const grouped = new Map<string, typeof rows>();
    for (const row of rows) {
      const key = `${row.videoId}:${Math.floor(row.timestampSeconds / 3)}`;
      grouped.set(key, [...(grouped.get(key) ?? []), row]);
    }

    const results = [];
    for (const group of grouped.values()) {
      group.sort((a, b) => (scoreById.get(b.vectorId) ?? 0) - (scoreById.get(a.vectorId) ?? 0));
      const best = group[0];
      const adjacent = group.filter((r) => Math.abs(r.timestampSeconds - best.timestampSeconds) <= 1.5);
      const score = adjacent.reduce((s, r) => s + (scoreById.get(r.vectorId) ?? 0), 0) / Math.max(adjacent.length, 1);
      results.push({
        id: crypto.randomUUID(),
        videoId: best.videoId,
        sourceName: best.sourceName,
        cameraId: best.cameraId ?? null,
        frameNumber: best.frameNumber,
        timestampSeconds: best.timestampSeconds,
        similarity: score,
        thumbnailPath: best.thumbnailPath ?? null,
        videoPath: `/api/storage/objects/${best.objectPath.replace(/^\/objects\//, "")}`,
        bbox: best.bbox,
        detectorScore: best.detectorScore,
        detectionId: best.detectionId,
        trackletId: best.trackletId ?? null,
      });
    }

    results.sort((a, b) => b.similarity - a.similarity);

    res.json({
      results: results.slice(0, topK),
      resultCount: results.length,
      queryMode: "detection",
      rankingNote:
        "Scores are cosine similarities (L2-normalised 512-d OSNet embeddings via FAISS IndexFlatIP). They are ranking signals, not identity probabilities. Verify each match against its source recording.",
    });
  },
);

// ─── FAISS Index Restore ──────────────────────────────────────────────────
// POST /api/visiontrace/faiss/restore
// Admin endpoint to rebuild the FAISS index from all indexed DB embeddings.
// Useful after FAISS file is lost or corrupted. Idempotent.
router.post(
  "/visiontrace/faiss/restore",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;

    const allEmbeddings = await db
      .select({
        vectorId: reidEmbeddingsTable.vectorId,
        vector: reidEmbeddingsTable.vector,
        embeddingSpace: reidEmbeddingsTable.embeddingSpace,
      })
      .from(reidEmbeddingsTable)
      .innerJoin(personDetectionsTable, eq(reidEmbeddingsTable.detectionId, personDetectionsTable.id))
      .innerJoin(indexedFramesTable, eq(personDetectionsTable.frameId, indexedFramesTable.id))
      .innerJoin(sourceVideosTable, eq(indexedFramesTable.videoId, sourceVideosTable.id))
      .where(eq(sourceVideosTable.ownerId, ownerId))
      .limit(50000);

    if (!allEmbeddings.length) {
      res.json({ ok: true, restored: 0, message: "No indexed embeddings found to restore." });
      return;
    }

    const embeddingSpace = allEmbeddings[0].embeddingSpace;
    const BATCH = 1000;
    let totalRestored = 0;

    for (let i = 0; i < allEmbeddings.length; i += BATCH) {
      const batch = allEmbeddings.slice(i, i + BATCH);
      const result = await callMl<{ rebuilt: number; size: number }>(
        "/vectors/rebuild",
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            vectorIds: batch.map((e) => e.vectorId),
            vectors: batch.map((e) => e.vector),
            embeddingSpace,
          }),
        },
        120_000,
      );
      if (result.ok) totalRestored += result.data.rebuilt;
    }

    // Mark all as indexed
    await db
      .update(reidEmbeddingsTable)
      .set({ indexStatus: "indexed" })
      .where(eq(reidEmbeddingsTable.indexStatus, "pending"));

    res.json({ ok: true, restored: totalRestored, total: allEmbeddings.length });
  },
);

// ─── Processing Pipeline Status for Live Visualization ───────────────────
// GET /api/visiontrace/pipeline/status
// Returns rich job + video status for the live processing visualization panel.
router.get(
  "/visiontrace/pipeline/status",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = requestUser(req, res);
    if (!ownerId) return;

    const [jobs, videos, runtime, [statsRow]] = await Promise.all([
      db
        .select()
        .from(processingJobsTable)
        .where(and(eq(processingJobsTable.ownerId, ownerId), inArray(processingJobsTable.kind, ["video_index"])))
        .orderBy(desc(processingJobsTable.createdAt))
        .limit(50),
      db
        .select({
          id: sourceVideosTable.id,
          name: sourceVideosTable.name,
          status: sourceVideosTable.status,
          frameCount: sourceVideosTable.frameCount,
          personCount: sourceVideosTable.personCount,
          durationSeconds: sourceVideosTable.durationSeconds,
          cameraId: sourceVideosTable.cameraId,
          createdAt: sourceVideosTable.createdAt,
          updatedAt: sourceVideosTable.updatedAt,
        })
        .from(sourceVideosTable)
        .where(eq(sourceVideosTable.ownerId, ownerId))
        .orderBy(desc(sourceVideosTable.createdAt))
        .limit(50),
      getVisionRuntime(),
      db
        .select({
          indexedVideos: count(),
        })
        .from(sourceVideosTable)
        .where(and(eq(sourceVideosTable.ownerId, ownerId), eq(sourceVideosTable.status, "indexed"))),
    ]);

    const jobsByVideoId = new Map<string, typeof jobs[0]>();
    for (const j of jobs) {
      if (j.videoId) jobsByVideoId.set(j.videoId, j);
    }

    res.json({
      runtime,
      indexedVideoCount: Number(statsRow?.indexedVideos ?? 0),
      videos: videos.map((v) => {
        const job = jobsByVideoId.get(v.id);
        return {
          ...v,
          job: job
            ? {
                id: job.id,
                status: job.status,
                stage: job.stage,
                progress: job.progress,
                error: job.error ?? null,
                updatedAt: job.updatedAt,
                // Pipeline math breakdown
                pipelineStages: [
                  { name: "Frame Sampling", description: `Sample @ ${process.env.VISIONTRACE_SAMPLE_FPS ?? 1} FPS`, done: job.progress >= 15 },
                  { name: "YOLO11n Detection", description: "conf≥0.25, 640×640 grid", done: job.progress >= 40 },
                  { name: "IoU Tracking", description: "threshold=0.3, max_missed=8", done: job.progress >= 60 },
                  { name: "OSNet Re-ID", description: "512-d L2 embedding, 256×128 crop", done: job.progress >= 85 },
                  { name: "FAISS Indexing", description: "IndexFlatIP cosine (inner product)", done: job.progress >= 95 },
                ],
              }
            : null,
        };
      }),
    });
  },
);

export default router;
