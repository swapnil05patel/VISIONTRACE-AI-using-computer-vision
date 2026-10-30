import { existsSync, statSync } from "node:fs";
import { eq, sql } from "drizzle-orm";
import { db, datasetImportsTable, datasetSamplesTable } from "@workspace/db";
import { ListDatasetsResponse } from "@workspace/api-zod";

const DATASETS = [
  {
    id: "mars",
    name: "MARS",
    rootVariable: "VISIONTRACE_MARS_ROOT",
    accessNote:
      "MARS is a video-based person Re-ID dataset. No automatic download. Configure an authorized local copy (bbox_train/ and bbox_test/) and confirm provider terms before import. Benchmark samples stay separate from source-video indexing.",
  },
  {
    id: "msmt17",
    name: "MSMT17",
    rootVariable: "VISIONTRACE_MSMT17_ROOT",
    accessNote:
      "MSMT17 requires dataset-provider agreement. No automatic download. Configure list_train.txt or train/test identity folders. Benchmark data is kept separate from operational footage.",
  },
  {
    id: "market1501",
    name: "Market-1501",
    rootVariable: "VISIONTRACE_MARKET1501_ROOT",
    accessNote:
      "Market-1501 is a research Re-ID dataset. No automatic download. Configure bounding_box_train, bounding_box_test, and query. Confirm Tsinghua/provider terms before use.",
  },
] as const;

function isConfigured(rootVariable: string): boolean {
  const root = process.env[rootVariable];
  if (!root) return false;
  try {
    return statSync(root).isDirectory();
  } catch {
    return false;
  }
}

export async function listDatasetAdapters(ownerId: string) {
  const imports = await db
    .select({
      datasetId: datasetImportsTable.datasetId,
      itemCount: datasetImportsTable.itemCount,
      status: datasetImportsTable.status,
    })
    .from(datasetImportsTable)
    .where(eq(datasetImportsTable.ownerId, ownerId));
  const counts = new Map<string, number>();
  for (const item of imports) {
    counts.set(item.datasetId, (counts.get(item.datasetId) ?? 0) + item.itemCount);
  }
  const sampleCounts = await db
    .select({
      datasetId: datasetSamplesTable.datasetId,
      value: sql<number>`count(*)`,
    })
    .from(datasetSamplesTable)
    .groupBy(datasetSamplesTable.datasetId);
  for (const row of sampleCounts) {
    counts.set(row.datasetId, Number(row.value));
  }

  return ListDatasetsResponse.parse(
    DATASETS.map((dataset) => {
      const configured = isConfigured(dataset.rootVariable);
      return {
        id: dataset.id,
        name: dataset.name,
        adapterStatus: configured ? "supported" : "unavailable",
        configured,
        importedItems: counts.get(dataset.id) ?? 0,
        accessNote: dataset.accessNote,
      };
    }),
  );
}

export function getDatasetRoot(datasetId: string): string | null {
  const dataset = DATASETS.find((item) => item.id === datasetId);
  if (!dataset) return null;
  const root = process.env[dataset.rootVariable];
  if (!root || !existsSync(root)) return null;
  try {
    return statSync(root).isDirectory() ? root : null;
  } catch {
    return null;
  }
}

export function datasetAccessNote(datasetId: string): string {
  return DATASETS.find((item) => item.id === datasetId)?.accessNote ?? "Confirm dataset terms before import.";
}
