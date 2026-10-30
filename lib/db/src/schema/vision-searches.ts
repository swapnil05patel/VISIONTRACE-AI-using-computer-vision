import {
  index,
  pgTable,
  text,
  timestamp,
  uuid,
  integer,
  jsonb,
  uniqueIndex,
} from "drizzle-orm/pg-core";

export const completedSearchesTable = pgTable(
  "vision_completed_searches",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    ownerId: text("owner_id").notNull(),
    mode: text("mode").notNull(),
    queryObjectPath: text("query_object_path").notNull(),
    resultCount: integer("result_count").notNull().default(0),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
  },
  (table) => [
    index("vision_completed_searches_owner_created_idx").on(
      table.ownerId,
      table.createdAt,
    ),
  ],
);

export const datasetImportsTable = pgTable(
  "vision_dataset_imports",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    ownerId: text("owner_id").notNull(),
    datasetId: text("dataset_id").notNull(),
    split: text("split").notNull(),
    purpose: text("purpose").notNull(),
    itemCount: integer("item_count").notNull().default(0),
    status: text("status").notNull().default("queued"),
    version: text("version"),
    rootPath: text("root_path"),
    accessNote: text("access_note"),
    discoveredFiles: integer("discovered_files").notNull().default(0),
    validSamples: integer("valid_samples").notNull().default(0),
    identities: integer("identities").notNull().default(0),
    cameras: integer("cameras").notNull().default(0),
    skipped: integer("skipped").notNull().default(0),
    validationErrors: jsonb("validation_errors").$type<string[]>().notNull().default([]),
    importedAt: timestamp("imported_at", { withTimezone: true }),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
  },
  (table) => [
    index("vision_dataset_imports_owner_dataset_idx").on(
      table.ownerId,
      table.datasetId,
    ),
  ],
);

export const datasetSamplesTable = pgTable(
  "vision_dataset_samples",
  {
    id: text("id").primaryKey(),
    importId: uuid("import_id")
      .notNull()
      .references(() => datasetImportsTable.id, { onDelete: "cascade" }),
    datasetId: text("dataset_id").notNull(),
    split: text("split").notNull(),
    identityId: text("identity_id").notNull(),
    cameraId: text("camera_id"),
    filePath: text("file_path").notNull(),
    originalId: text("original_id").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
  },
  (table) => [
    uniqueIndex("vision_dataset_samples_dataset_original_idx").on(
      table.datasetId,
      table.originalId,
    ),
    index("vision_dataset_samples_identity_idx").on(table.datasetId, table.identityId),
  ],
);

export type CompletedSearchRecord = typeof completedSearchesTable.$inferSelect;
export type DatasetImportRecord = typeof datasetImportsTable.$inferSelect;
export type DatasetSampleRecord = typeof datasetSamplesTable.$inferSelect;
