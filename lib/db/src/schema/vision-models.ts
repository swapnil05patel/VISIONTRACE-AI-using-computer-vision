import { index, jsonb, pgTable, text, timestamp, uuid, integer, boolean } from "drizzle-orm/pg-core";

export const modelArtifactsTable = pgTable(
  "vision_model_artifacts",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    name: text("name").notNull(),
    type: text("type").notNull(),
    architecture: text("architecture").notNull(),
    version: text("version").notNull(),
    artifactPath: text("artifact_path").notNull(),
    checksum: text("checksum").notNull(),
    embeddingDimension: integer("embedding_dimension"),
    preprocessId: text("preprocess_id"),
    embeddingSpace: text("embedding_space"),
    datasetRef: text("dataset_ref"),
    trainingRunId: uuid("training_run_id"),
    trainingConfig: jsonb("training_config"),
    evaluationMetrics: jsonb("evaluation_metrics"),
    status: text("status").notNull().default("registered"),
    verified: boolean("verified").notNull().default(false),
    active: boolean("active").notNull().default(false),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
  },
  (table) => [
    index("vision_model_artifacts_type_status_idx").on(table.type, table.status),
  ],
);

export const trainingRunsTable = pgTable(
  "vision_training_runs",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    ownerId: text("owner_id").notNull(),
    datasetId: text("dataset_id").notNull(),
    status: text("status").notNull().default("queued"),
    config: jsonb("config").notNull(),
    metrics: jsonb("metrics"),
    checkpointPath: text("checkpoint_path"),
    checkpointSha256: text("checkpoint_sha256"),
    error: text("error"),
    startedAt: timestamp("started_at", { withTimezone: true }),
    completedAt: timestamp("completed_at", { withTimezone: true }),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [index("vision_training_runs_owner_idx").on(table.ownerId, table.createdAt)],
);

export const modelActivationHistoryTable = pgTable("vision_model_activation_history", {
  id: uuid("id").defaultRandom().primaryKey(),
  modelId: uuid("model_id")
    .notNull()
    .references(() => modelArtifactsTable.id, { onDelete: "cascade" }),
  action: text("action").notNull(),
  actorId: text("actor_id").notNull(),
  previousModelId: uuid("previous_model_id"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});
