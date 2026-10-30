import {
  index,
  jsonb,
  pgTable,
  text,
  timestamp,
  uuid,
  integer,
} from "drizzle-orm/pg-core";
import { sourceVideosTable } from "./vision-videos";

export const processingJobsTable = pgTable(
  "vision_processing_jobs",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    ownerId: text("owner_id").notNull(),
    kind: text("kind").notNull(),
    status: text("status").notNull().default("queued"),
    progress: integer("progress").notNull().default(0),
    stage: text("stage").notNull().default("queued"),
    sourceName: text("source_name"),
    error: text("error"),
    videoId: uuid("video_id").references(() => sourceVideosTable.id, {
      onDelete: "set null",
    }),
    datasetId: text("dataset_id"),
    result: jsonb("result"),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
  },
  (table) => [
    index("vision_processing_jobs_owner_created_idx").on(
      table.ownerId,
      table.createdAt,
    ),
    index("vision_processing_jobs_owner_status_idx").on(
      table.ownerId,
      table.status,
    ),
  ],
);

export type ProcessingJobRecord = typeof processingJobsTable.$inferSelect;
