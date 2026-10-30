import { Readable } from "node:stream";
import { eq, and } from "drizzle-orm";
import {
  RequestUploadUrlBody,
  RequestUploadUrlResponse,
} from "@workspace/api-zod";
import { db, privateUploadsTable } from "@workspace/db";
import { Router, type IRouter, type Request, type Response } from "express";
import { authenticatedUserId } from "../lib/clerkAuth";
import { ObjectPermission } from "../lib/objectAcl";
import { ObjectStorageService } from "../lib/objectStorage";

const router: IRouter = Router();
const objectStorage = new ObjectStorageService();
const MAX_IMAGE_BYTES = 32 * 1024 * 1024;
const MAX_VIDEO_BYTES = 512 * 1024 * 1024;
const IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const VIDEO_TYPES = new Set(["video/mp4", "video/quicktime", "video/webm"]);
const uploadPathPattern = /^\/objects\/(uploads|thumbs|crops)\/[0-9a-f-]{36}$/i;

function sendObject(response: Response, objectResponse: globalThis.Response): void {
  response.status(objectResponse.status);
  objectResponse.headers.forEach((value, key) => response.setHeader(key, value));
  if (objectResponse.body) {
    Readable.fromWeb(
      objectResponse.body as ReadableStream<Uint8Array>,
    ).pipe(response);
  } else {
    response.end();
  }
}

function matchesContent(bytes: Buffer, contentType: string): boolean {
  if (contentType === "image/jpeg") {
    return bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
  }
  if (contentType === "image/png") {
    return bytes.subarray(0, 8).equals(
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    );
  }
  if (contentType === "image/webp") {
    return (
      bytes.toString("ascii", 0, 4) === "RIFF" &&
      bytes.toString("ascii", 8, 12) === "WEBP"
    );
  }
  if (contentType === "video/mp4" || contentType === "video/quicktime") {
    return bytes.toString("ascii", 4, 8) === "ftyp";
  }
  if (contentType === "video/webm") {
    return bytes.subarray(0, 4).equals(Buffer.from([0x1a, 0x45, 0xdf, 0xa3]));
  }
  return false;
}

export async function getOwnedUpload(
  ownerId: string,
  objectPath: string,
  kind: "image" | "video",
): Promise<
  | { ok: true; file: Awaited<ReturnType<ObjectStorageService["getObjectEntityFile"]>> }
  | { ok: false; status: number; error: string }
> {
  if (!uploadPathPattern.test(objectPath)) {
    return { ok: false, status: 400, error: "Invalid private upload path." };
  }
  const [upload] = await db
    .select()
    .from(privateUploadsTable)
    .where(
      and(
        eq(privateUploadsTable.ownerId, ownerId),
        eq(privateUploadsTable.objectPath, objectPath),
      ),
    )
    .limit(1);
  if (!upload) {
    return { ok: false, status: 404, error: "Uploaded file not found." };
  }
  const acceptedTypes = kind === "image" ? IMAGE_TYPES : VIDEO_TYPES;
  const maxBytes = kind === "image" ? MAX_IMAGE_BYTES : MAX_VIDEO_BYTES;
  if (!acceptedTypes.has(upload.contentType)) {
    return { ok: false, status: 415, error: `Unsupported ${kind} content type.` };
  }
  if (upload.sizeBytes > maxBytes) {
    return { ok: false, status: 413, error: `${kind} exceeds the configured size limit.` };
  }

  try {
    const file = await objectStorage.getObjectEntityFile(objectPath);
    const [metadata] = await file.getMetadata();
    const actualSize = Number(metadata.size ?? 0);
    const actualType = String(metadata.contentType ?? "");
    if (!actualSize || actualSize > maxBytes) {
      return { ok: false, status: 413, error: `${kind} exceeds the configured size limit.` };
    }
    const [header] = await file.download({ start: 0, end: 31 });
    if (!acceptedTypes.has(actualType) || !matchesContent(header, actualType)) {
      return { ok: false, status: 415, error: "File content does not match an allowed media format." };
    }
    await objectStorage.trySetObjectEntityAclPolicy(objectPath, {
      owner: ownerId,
      visibility: "private",
    });
    if (!upload.verified) {
      await db
        .update(privateUploadsTable)
        .set({ verified: true })
        .where(eq(privateUploadsTable.id, upload.id));
    }
    return { ok: true, file };
  } catch (error) {
    return {
      ok: false,
      status: 404,
      error:
        error instanceof Error ? error.message : "Uploaded object is not available yet.",
    };
  }
}

router.post(
  "/storage/uploads/request-url",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = authenticatedUserId(req, res);
    if (!ownerId) return;
    const parsed = RequestUploadUrlBody.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Missing or invalid upload metadata." });
      return;
    }
    const { name, size, contentType } = parsed.data;
    const isImage = IMAGE_TYPES.has(contentType);
    const isVideo = VIDEO_TYPES.has(contentType);
    if (!isImage && !isVideo) {
      res.status(415).json({ error: "Use a JPEG, PNG, WebP, MP4, MOV, or WebM file." });
      return;
    }
    const maxBytes = isImage ? MAX_IMAGE_BYTES : MAX_VIDEO_BYTES;
    if (size > maxBytes) {
      res.status(413).json({
        error: isImage
          ? "Images must be 32 MB or smaller."
          : "Videos must be 512 MB or smaller.",
      });
      return;
    }

    try {
      const uploadURL = await objectStorage.getObjectEntityUploadURL();
      const objectPath = objectStorage.normalizeObjectEntityPath(uploadURL);
      if (!uploadPathPattern.test(objectPath)) {
        req.log.error({ objectPath }, "Storage returned an unexpected private path");
        res.status(500).json({ error: "Could not create a safe upload path." });
        return;
      }
      await db.insert(privateUploadsTable).values({
        ownerId,
        objectPath,
        name,
        contentType,
        sizeBytes: size,
      });
      res.json(
        RequestUploadUrlResponse.parse({
          uploadURL,
          objectPath,
          metadata: { name, size, contentType },
        }),
      );
    } catch (error) {
      req.log.error({ err: error }, "Failed to create private upload URL");
      res.status(500).json({ error: "Failed to prepare private upload." });
    }
  },
);

router.put(
  "/storage/uploads/direct/objects/:kind/:id",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = authenticatedUserId(req, res);
    if (!ownerId) return;
    const objectPath = `/objects/${req.params.kind}/${req.params.id}`;
    if (!uploadPathPattern.test(objectPath)) {
      res.status(400).json({ error: "Invalid upload path." });
      return;
    }
    const chunks: Buffer[] = [];
    for await (const chunk of req) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    const bytes = Buffer.concat(chunks);
    if (!bytes.length) {
      res.status(400).json({ error: "Empty upload." });
      return;
    }
    const [upload] = await db
      .select()
      .from(privateUploadsTable)
      .where(
        and(
          eq(privateUploadsTable.ownerId, ownerId),
          eq(privateUploadsTable.objectPath, objectPath),
        ),
      )
      .limit(1);
    if (!upload) {
      res.status(404).json({ error: "Upload session was not found." });
      return;
    }
    if (!matchesContent(bytes.subarray(0, 32), upload.contentType)) {
      res.status(415).json({ error: "File content does not match an allowed media format." });
      return;
    }
    await objectStorage.writeLocalObject(objectPath, bytes, upload.contentType, upload.name);
    res.status(200).json({ ok: true, objectPath });
  },
);

router.get(
  "/storage/objects/*path",
  async (req: Request, res: Response): Promise<void> => {
    const ownerId = authenticatedUserId(req, res);
    if (!ownerId) return;
    const raw = req.params.path;
    const pathPart = Array.isArray(raw) ? raw.join("/") : raw;
    const objectPath = `/objects/${pathPart}`;
    if (!uploadPathPattern.test(objectPath)) {
      res.status(404).json({ error: "Object not found." });
      return;
    }
    const [upload] = await db
      .select()
      .from(privateUploadsTable)
      .where(
        and(
          eq(privateUploadsTable.ownerId, ownerId),
          eq(privateUploadsTable.objectPath, objectPath),
        ),
      )
      .limit(1);
    if (!upload || !upload.verified) {
      res.status(404).json({ error: "Object not found." });
      return;
    }
    try {
      const file = await objectStorage.getObjectEntityFile(objectPath);
      const canRead = await objectStorage.canAccessObjectEntity({
        userId: ownerId,
        objectFile: file,
        requestedPermission: ObjectPermission.READ,
      });
      if (!canRead) {
        res.status(403).json({ error: "You do not have access to this object." });
        return;
      }
      sendObject(res, await objectStorage.downloadObject(file, 120));
    } catch (error) {
      req.log.warn({ err: error }, "Private object is not available");
      res.status(404).json({ error: "Object not found." });
    }
  },
);

export default router;
