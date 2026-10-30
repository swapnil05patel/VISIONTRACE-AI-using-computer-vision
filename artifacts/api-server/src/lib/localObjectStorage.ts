import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { Readable } from "node:stream";
import { randomUUID } from "node:crypto";

export type LocalObjectMeta = {
  size: number;
  contentType: string;
  name: string;
  metadata?: Record<string, string>;
};

export class LocalStoredFile {
  readonly name: string;

  constructor(private readonly absolutePath: string, private readonly objectPath: string) {
    this.name = this.objectPath;
  }

  private metaPath() {
    return `${this.absolutePath}.meta.json`;
  }

  async exists(): Promise<[boolean]> {
    return [existsSync(this.absolutePath)];
  }

  async getMetadata(): Promise<[LocalObjectMeta]> {
    const stat = statSync(this.absolutePath);
    let extra: Partial<LocalObjectMeta> = {};
    if (existsSync(this.metaPath())) {
      extra = JSON.parse(readFileSync(this.metaPath(), "utf8")) as LocalObjectMeta;
    }
    return [
      {
        size: stat.size,
        contentType: extra.contentType || "application/octet-stream",
        name: extra.name || this.objectPath.split("/").pop() || "object",
        metadata: extra.metadata,
      },
    ];
  }

  async setMetadata(update: { metadata?: Record<string, string>; contentType?: string }): Promise<void> {
    const [current] = await this.getMetadata();
    writeFileSync(
      this.metaPath(),
      JSON.stringify({
        ...current,
        contentType: update.contentType || current.contentType,
        metadata: { ...(current.metadata ?? {}), ...(update.metadata ?? {}) },
      }),
    );
  }

  async download(range?: { start?: number; end?: number }): Promise<[Buffer]> {
    const bytes = readFileSync(this.absolutePath);
    if (range?.start != null || range?.end != null) {
      const start = range.start ?? 0;
      const end = (range.end ?? bytes.length - 1) + 1;
      return [bytes.subarray(start, end)];
    }
    return [bytes];
  }

  createReadStream() {
    return createReadStream(this.absolutePath);
  }

  save(bytes: Buffer, contentType: string, name: string) {
    mkdirSync(dirname(this.absolutePath), { recursive: true });
    writeFileSync(this.absolutePath, bytes);
    writeFileSync(
      this.metaPath(),
      JSON.stringify({ size: bytes.length, contentType, name, metadata: {} }),
    );
  }
}

export function usesLocalObjectStorage(): boolean {
  if (process.env.VISIONTRACE_OBJECT_BACKEND === "gcs") return false;
  if (process.env.VISIONTRACE_OBJECT_BACKEND === "fs") return true;
  const dir = process.env.PRIVATE_OBJECT_DIR || resolve("data/objects");
  return dir.startsWith(".") || dir.includes("data/objects") || existsSync(dir);
}

export class LocalObjectStorage {
  constructor(private readonly root: string) {
    mkdirSync(this.root, { recursive: true });
  }

  private abs(objectPath: string): string {
    const relative = objectPath.replace(/^\/objects\//, "");
    const resolved = resolve(join(this.root, relative));
    if (!resolved.startsWith(resolve(this.root))) {
      throw new Error("Invalid object path.");
    }
    return resolved;
  }

  createUpload(): { uploadURL: string; objectPath: string } {
    const id = randomUUID();
    const objectPath = `/objects/uploads/${id}`;
    return {
      objectPath,
      uploadURL: `/api/storage/uploads/direct${objectPath}`,
    };
  }

  file(objectPath: string): LocalStoredFile {
    return new LocalStoredFile(this.abs(objectPath), objectPath);
  }
}

export function downloadLocalFile(file: LocalStoredFile, cacheTtlSec: number): globalThis.Response {
  const nodeStream = file.createReadStream();
  const webStream = Readable.toWeb(nodeStream as unknown as Readable) as ReadableStream;
  return new Response(webStream, {
    headers: {
      "Content-Type": "application/octet-stream",
      "Cache-Control": `private, max-age=${cacheTtlSec}`,
    },
  });
}
