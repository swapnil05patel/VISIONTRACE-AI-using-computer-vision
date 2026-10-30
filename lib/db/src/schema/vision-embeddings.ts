import {
  index,
  pgTable,
  real,
  timestamp,
  uuid,
  integer,
  serial,
  text,
} from "drizzle-orm/pg-core";
import { personDetectionsTable } from "./vision-frames";

export const reidEmbeddingsTable = pgTable(
  "vision_reid_embeddings",
  {
    vectorId: serial("vector_id").primaryKey(),
    detectionId: uuid("detection_id")
      .notNull()
      .references(() => personDetectionsTable.id, { onDelete: "cascade" })
      .unique(),
    dimension: integer("dimension").notNull(),
    vector: real("vector").array().notNull(),
    modelVersion: text("model_version").notNull().default("osnet_x1_0"),
    embeddingSpace: text("embedding_space").notNull().default("osnet_x1_0_imagenet_l2"),
    indexStatus: text("index_status").notNull().default("pending"),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
  },
  (table) => [index("vision_reid_embeddings_detection_idx").on(table.detectionId)],
);

export type ReidEmbeddingRecord = typeof reidEmbeddingsTable.$inferSelect;
