import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { eq, inArray } from "drizzle-orm";
import {
  db,
  datasetImportsTable,
  datasetSamplesTable,
  indexedFramesTable,
  personDetectionsTable,
  processingJobsTable,
  reidEmbeddingsTable,
  sourceVideosTable,
  completedSearchesTable,
} from "@workspace/db";
import { callMl, getVisionRuntime } from "./visionRuntime";
import { datasetAccessNote, getDatasetRoot } from "./datasetAdapters";
import { ObjectStorageService } from "./objectStorage";
import { logger } from "./logger";

const objectStorage = new ObjectStorageService();
const SAMPLE_FPS = Number(process.env.VISIONTRACE_SAMPLE_FPS ?? "1");

type JobRow = typeof processingJobsTable.$inferSelect;

function pythonBin() {
  return process.env.PYTHON_EXECUTABLE ?? "python3.12";
}

function mlDir() {
  const candidates = [
    resolve(process.cwd(), "ml"),
    resolve(process.cwd(), "artifacts/api-server/ml"),
  ];
  return candidates.find((path) => {
    try {
      return require("node:fs").existsSync(join(path, "import_dataset.py"));
    } catch {
      return false;
    }
  }) ?? resolve(process.cwd(), "artifacts/api-server/ml");
}

async function updateJob(id: string, values: Partial<JobRow>) {
  await db.update(processingJobsTable).set(values).where(eq(processingJobsTable.id, id));
}

function runPython(args: string[], cwd: string): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolvePromise) => {
    const child = spawn(pythonBin(), args, { cwd, env: process.env });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("close", (code) => resolvePromise({ code: code ?? 1, stdout, stderr }));
    child.on("error", (error) => resolvePromise({ code: 1, stdout, stderr: String(error) }));
  });
}

async function materializeObject(objectPath: string, suffix: string): Promise<{ path: string; cleanup: () => void; bytes: Buffer; sha256: string }> {
  const file = await objectStorage.getObjectEntityFile(objectPath);
  const [bytes] = await file.download();
  const dir = mkdtempSync(join(tmpdir(), "visiontrace-"));
  const path = join(dir, `media${suffix}`);
  writeFileSync(path, bytes);
  return {
    path,
    bytes,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    cleanup: () => {
      try {
        rmSync(dir, { recursive: true, force: true });
      } catch {
        /* ignore */
      }
    },
  };
}

async function indexPendingEmbeddings() {
  const pending = await db
    .select()
    .from(reidEmbeddingsTable)
    .where(eq(reidEmbeddingsTable.indexStatus, "pending"))
    .limit(500);
  if (!pending.length) return;
  const result = await callMl<{ added: number }>(
    "/vectors/add",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        vectorIds: pending.map((row) => row.vectorId),
        vectors: pending.map((row) => row.vector),
        embeddingSpace: pending[0]?.embeddingSpace,
      }),
    },
    60_000,
  );
  if (!result.ok) {
    throw new Error(result.error);
  }
  await db
    .update(reidEmbeddingsTable)
    .set({ indexStatus: "indexed" })
    .where(
      inArray(
        reidEmbeddingsTable.vectorId,
        pending.map((row) => row.vectorId),
      ),
    );
}

async function processVideoIndex(job: JobRow) {
  if (!job.videoId) throw new Error("Video indexing job is missing a video id.");
  const [video] = await db.select().from(sourceVideosTable).where(eq(sourceVideosTable.id, job.videoId)).limit(1);
  if (!video) throw new Error("Source video was not found.");
  await db.update(sourceVideosTable).set({ status: "processing" }).where(eq(sourceVideosTable.id, video.id));
  await updateJob(job.id, { status: "running", stage: "validating", progress: 5, error: null });
  const runtime = await getVisionRuntime();
  if (runtime.status !== "ready") {
    throw new Error(runtime.limitations.join(" ") || "Models are not verified.");
  }
  const local = await materializeObject(video.objectPath, ".mp4");
  try {
    const probe = await callMl<{
      durationSeconds: number | null;
      fps: number | null;
      width: number | null;
      height: number | null;
    }>(
      "/probe-video",
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ path: local.path, sampleFps: SAMPLE_FPS }),
      },
      60_000,
    );
    if (!probe.ok) throw new Error(probe.error);
    await db
      .update(sourceVideosTable)
      .set({
        durationSeconds: probe.data.durationSeconds,
        fps: probe.data.fps,
        width: probe.data.width,
        height: probe.data.height,
        sampleFps: SAMPLE_FPS,
        sha256: local.sha256,
        sizeBytes: local.bytes.length,
        contentType: "video/mp4",
      })
      .where(eq(sourceVideosTable.id, video.id));
    let startFrame = 0;
    let people = 0;
    let frames = 0;
    let exhausted = false;
    const space = "osnet_x1_0_imagenet_l2";
    while (!exhausted) {
      await updateJob(job.id, {
        stage: frames === 0 ? "extracting_frames" : "detecting_people",
        progress: Math.min(90, 10 + frames),
      });
      const chunk = await callMl<{
        frames: Array<{
          frameNumber: number;
          timestampSeconds: number;
          thumbnailJpegBase64: string;
          detections: Array<{
            bbox: number[];
            detectorScore: number;
            trackletId: string | null;
            embedding: number[];
          }>;
        }>;
        lastFrame: number;
        exhausted: boolean;
        embeddingSpace: string;
      }>(
        "/process-video-chunk",
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            path: local.path,
            sampleFps: SAMPLE_FPS,
            startFrame,
            maxFrames: 8,
            track: true,
          }),
        },
        180_000,
      );
      if (!chunk.ok) throw new Error(chunk.error);
      if (chunk.data.embeddingSpace && chunk.data.embeddingSpace !== space) {
        throw new Error("Embedding space from the runtime does not match the index.");
      }
      for (const frame of chunk.data.frames) {
        const [insertedFrame] = await db
          .insert(indexedFramesTable)
          .values({
            videoId: video.id,
            frameNumber: frame.frameNumber,
            timestampSeconds: frame.timestampSeconds,
            thumbnailPath: null,
          })
          .onConflictDoNothing()
          .returning();
        const frameRow = insertedFrame
          ? insertedFrame
          : (
              await db
                .select()
                .from(indexedFramesTable)
                .where(eq(indexedFramesTable.videoId, video.id))
            ).find((row) => row.frameNumber === frame.frameNumber);
        if (!frameRow) continue;
        frames += 1;
        let detectionIndex = 0;
        for (const detection of frame.detections) {
          const [insertedDetection] = await db
            .insert(personDetectionsTable)
            .values({
              frameId: frameRow.id,
              detectionIndex,
              bbox: detection.bbox,
              detectorScore: detection.detectorScore,
              trackletId: detection.trackletId ? `${video.id}:${detection.trackletId}` : null,
            })
            .onConflictDoNothing()
            .returning();
          detectionIndex += 1;
          if (!insertedDetection) continue;
          people += 1;
          await db
            .insert(reidEmbeddingsTable)
            .values({
              detectionId: insertedDetection.id,
              dimension: detection.embedding.length,
              vector: detection.embedding,
              modelVersion: "osnet_x1_0",
              embeddingSpace: space,
              indexStatus: "pending",
            })
            .onConflictDoNothing();
        }
      }
      await updateJob(job.id, { stage: "generating_embeddings", progress: Math.min(92, 20 + frames) });
      await indexPendingEmbeddings();
      exhausted = chunk.data.exhausted;
      startFrame = chunk.data.lastFrame + 1;
    }
    await updateJob(job.id, { stage: "indexing", progress: 96 });
    await db
      .update(sourceVideosTable)
      .set({ status: "indexed", frameCount: frames, personCount: people })
      .where(eq(sourceVideosTable.id, video.id));
    await updateJob(job.id, { status: "completed", stage: "completed", progress: 100, error: null });
  } finally {
    local.cleanup();
  }
}

async function processDatasetImport(job: JobRow) {
  if (!job.datasetId) throw new Error("Dataset import job is missing a dataset id.");
  const root = getDatasetRoot(job.datasetId);
  if (!root) throw new Error("No authorized local dataset root is configured.");
  const source = job.sourceName ?? "";
  const parts = source.split(" · ");
  const limitMatch = /limit (\d+)/.exec(source);
  const resultJson = {
    split: parts[1],
    purpose: parts[2]?.split(" ")[0],
    maxItems: limitMatch ? Number(limitMatch[1]) : undefined,
  };
  await updateJob(job.id, { status: "running", stage: "validating", progress: 10 });
  const output = join("data/dataset-imports", `${job.datasetId}-${job.id}.json`);
  const args = [
    "import_dataset.py",
    "--dataset",
    job.datasetId,
    "--root",
    root,
    "--purpose",
    resultJson?.purpose ?? "benchmark",
    "--output",
    resolve("data/dataset-imports"),
  ];
  if (resultJson?.split) args.push("--split", resultJson.split);
  if (resultJson?.maxItems) args.push("--max-items", String(resultJson.maxItems));
  const ran = await runPython(args, mlDir());
  if (ran.code !== 0) {
    throw new Error(ran.stderr || ran.stdout || "Dataset parser failed.");
  }
  const parsed = JSON.parse(ran.stdout) as {
    success?: boolean;
    output?: string;
    discoveredFiles?: number;
    validSamples?: number;
    identities?: number;
    cameras?: number;
    skipped?: number;
    errors?: string[];
    splits?: Record<string, number>;
  };
  const manifestPath = parsed.output ?? output;
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
    samples: Array<{
      id: string;
      datasetId: string;
      split: string;
      identityId: string;
      cameraId: string | null;
      path: string;
      originalId: string;
    }>;
    discoveredFiles: number;
    validSamples: number;
    identities: number;
    cameras: number;
    skipped: number;
    errors: string[];
  };
  const [record] = await db
    .insert(datasetImportsTable)
    .values({
      ownerId: job.ownerId,
      datasetId: job.datasetId,
      split: resultJson?.split ?? "all",
      purpose: resultJson?.purpose ?? "benchmark",
      itemCount: manifest.validSamples,
      status: manifest.errors?.length && !manifest.validSamples ? "failed" : "completed",
      rootPath: root,
      accessNote: datasetAccessNote(job.datasetId),
      discoveredFiles: manifest.discoveredFiles,
      validSamples: manifest.validSamples,
      identities: manifest.identities,
      cameras: manifest.cameras,
      skipped: manifest.skipped,
      validationErrors: manifest.errors ?? [],
      importedAt: new Date(),
    })
    .returning();
  for (const sample of manifest.samples) {
    await db
      .insert(datasetSamplesTable)
      .values({
        id: sample.id,
        importId: record.id,
        datasetId: sample.datasetId,
        split: sample.split,
        identityId: sample.identityId,
        cameraId: sample.cameraId,
        filePath: sample.path,
        originalId: sample.originalId,
      })
      .onConflictDoNothing();
  }
  await updateJob(job.id, {
    status: "completed",
    stage: "completed",
    progress: 100,
    error: manifest.errors?.length ? manifest.errors.join("; ") : null,
  });
}

function aggregateHits(
  hits: Array<{
    vectorId: number;
    similarity: number;
    videoId: string;
    sourceName: string;
    cameraId: string | null;
    frameNumber: number;
    timestampSeconds: number;
    thumbnailPath: string | null;
    objectPath: string;
    trackletId: string | null;
  }>,
) {
  const grouped = new Map<string, typeof hits>();
  for (const hit of hits) {
    const key = `${hit.videoId}:${hit.trackletId ?? hit.frameNumber}`;
    grouped.set(key, [...(grouped.get(key) ?? []), hit]);
  }
  const collapsed = [];
  for (const group of grouped.values()) {
    group.sort((a, b) => b.similarity - a.similarity);
    const best = group[0];
    const adjacent = group.filter((item) => Math.abs(item.timestampSeconds - best.timestampSeconds) <= 1.5);
    const score =
      adjacent.reduce((sum, item) => sum + item.similarity, 0) / Math.max(adjacent.length, 1);
    collapsed.push({
      id: crypto.randomUUID(),
      videoId: best.videoId,
      sourceName: best.sourceName,
      cameraId: best.cameraId,
      frameNumber: best.frameNumber,
      timestampSeconds: best.timestampSeconds,
      similarity: score,
      thumbnailPath: best.thumbnailPath,
      videoPath: `/api/storage/objects/${best.objectPath.replace(/^\/objects\//, "")}`,
    });
  }
  return collapsed.sort((a, b) => b.similarity - a.similarity);
}

async function processVideoSearch(job: JobRow) {
  const runtime = await getVisionRuntime();
  if (runtime.status !== "ready") throw new Error(runtime.limitations.join(" "));
  const payload = {
    objectPath: job.sourceName ?? "",
    filters: { topK: 10 },
  };
  if (!payload.objectPath) throw new Error("Video search job is missing the query object.");
  await updateJob(job.id, { status: "running", stage: "extracting_frames", progress: 15 });
  const local = await materializeObject(payload.objectPath, ".mp4");
  try {
    const chunk = await callMl<{
      frames: Array<{ detections: Array<{ embedding: number[] }> }>;
    }>(
      "/process-video-chunk",
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ path: local.path, sampleFps: SAMPLE_FPS, startFrame: 0, maxFrames: 24, track: true }),
      },
      180_000,
    );
    if (!chunk.ok) throw new Error(chunk.error);
    const embeddings = chunk.data.frames.flatMap((frame) => frame.detections.map((item) => item.embedding));
    if (!embeddings.length) throw new Error("No person was detected in the query video.");
    const matches: { vectorId: number; similarity: number }[] = [];
    for (const vector of embeddings.slice(0, 32)) {
      const searched = await callMl<{ matches: { vectorId: number; similarity: number }[] }>(
        "/vectors/search",
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ vector, topK: Math.min(100, (payload.filters?.topK ?? 10) * 10) }),
        },
        30_000,
      );
      if (searched.ok) matches.push(...searched.data.matches);
    }
    const ids = [...new Set(matches.map((item) => item.vectorId))];
    const rows = ids.length
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
            trackletId: personDetectionsTable.trackletId,
            ownerId: sourceVideosTable.ownerId,
          })
          .from(reidEmbeddingsTable)
          .innerJoin(personDetectionsTable, eq(reidEmbeddingsTable.detectionId, personDetectionsTable.id))
          .innerJoin(indexedFramesTable, eq(personDetectionsTable.frameId, indexedFramesTable.id))
          .innerJoin(sourceVideosTable, eq(indexedFramesTable.videoId, sourceVideosTable.id))
          .where(inArray(reidEmbeddingsTable.vectorId, ids))
      : [];
    const scoreById = new Map(matches.map((item) => [item.vectorId, item.similarity]));
    const authorized = rows
      .filter((row) => row.ownerId === job.ownerId)
      .map((row) => ({
        vectorId: row.vectorId,
        similarity: scoreById.get(row.vectorId) ?? 0,
        videoId: row.videoId,
        sourceName: row.sourceName,
        cameraId: row.cameraId,
        frameNumber: row.frameNumber,
        timestampSeconds: row.timestampSeconds,
        thumbnailPath: row.thumbnailPath,
        objectPath: row.objectPath,
        trackletId: row.trackletId,
      }));
    const results = aggregateHits(authorized).slice(0, payload.filters?.topK ?? 10);
    await db.insert(completedSearchesTable).values({
      ownerId: job.ownerId,
      mode: "video",
      queryObjectPath: payload.objectPath,
      resultCount: results.length,
    });
    await updateJob(job.id, {
      status: "completed",
      stage: "completed",
      progress: 100,
      result: {
        queryMode: "video",
        results,
        resultCount: results.length,
        rankingNote:
          "Scores are raw cosine similarities aggregated across nearby frames, not identity probabilities. Sampling rate and adjacent-frame collapse affect which timestamps appear.",
      },
    });
  } finally {
    local.cleanup();
  }
}

async function claimJob(): Promise<JobRow | null> {
  const [queued] = await db
    .select()
    .from(processingJobsTable)
    .where(eq(processingJobsTable.status, "queued"))
    .orderBy(processingJobsTable.createdAt)
    .limit(1);
  if (!queued) return null;
  const claimed = await db
    .update(processingJobsTable)
    .set({ status: "running", stage: queued.stage === "queued" ? "running" : queued.stage })
    .where(eq(processingJobsTable.id, queued.id))
    .returning();
  return claimed[0] ?? null;
}

async function tick() {
  const job = await claimJob();
  if (!job) return;
  try {
    if (job.kind === "video_index") await processVideoIndex(job);
    else if (job.kind === "dataset_import") await processDatasetImport(job);
    else if (job.kind === "video_search") await processVideoSearch(job);
    else await updateJob(job.id, { status: "failed", stage: "failed", error: `Unsupported job kind ${job.kind}.` });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Processing failed.";
    logger.error({ err: error, jobId: job.id }, "VisionTrace job failed");
    await updateJob(job.id, { status: "failed", stage: "failed", error: message, progress: job.progress });
    if (job.videoId) {
      await db.update(sourceVideosTable).set({ status: "failed" }).where(eq(sourceVideosTable.id, job.videoId));
    }
  }
}

export function startJobWorker() {
  const timer = setInterval(() => {
    tick().catch((error) => logger.error({ err: error }, "Job worker tick failed"));
  }, 2000);
  timer.unref?.();
  return timer;
}
