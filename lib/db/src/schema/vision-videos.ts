import {
  index,
  pgTable,
  real,
  text,
  timestamp,
  uniqueIndex,
  uuid,
  integer,
} from "drizzle-orm/pg-core";

export const sourceVideosTable = pgTable(
  "vision_source_videos",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    ownerId: text("owner_id").notNull(),
    name: text("name").notNull(),
    objectPath: text("object_path").notNull(),
    cameraId: text("camera_id"),
    recordedAt: timestamp("recorded_at", { withTimezone: true }),
    durationSeconds: real("duration_seconds"),
    status: text("status").notNull().default("queued"),
    frameCount: integer("frame_count").notNull().default(0),
    personCount: integer("person_count").notNull().default(0),
    sha256: text("sha256"),
    sizeBytes: integer("size_bytes"),
    contentType: text("content_type"),
    fps: real("fps"),
    width: integer("width"),
    height: integer("height"),
    sampleFps: real("sample_fps"),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true })
      .notNull()
      .defaultNow()
      .$onUpdate(() => new Date()),
  },
  (table) => [
    uniqueIndex("vision_source_videos_owner_object_path_idx").on(
      table.ownerId,
      table.objectPath,
    ),
    index("vision_source_videos_owner_created_idx").on(
      table.ownerId,
      table.createdAt,
    ),
    index("vision_source_videos_owner_camera_idx").on(
      table.ownerId,
      table.cameraId,
    ),
  ],
);

export type SourceVideoRecord = typeof sourceVideosTable.$inferSelect;
export type NewSourceVideoRecord = typeof sourceVideosTable.$inferInsert;
