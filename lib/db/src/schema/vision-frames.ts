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
import { sourceVideosTable } from "./vision-videos";

export const indexedFramesTable = pgTable(
  "vision_indexed_frames",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    videoId: uuid("video_id")
      .notNull()
      .references(() => sourceVideosTable.id, { onDelete: "cascade" }),
    frameNumber: integer("frame_number").notNull(),
    timestampSeconds: real("timestamp_seconds").notNull(),
    thumbnailPath: text("thumbnail_path"),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
  },
  (table) => [
    uniqueIndex("vision_indexed_frames_video_number_idx").on(
      table.videoId,
      table.frameNumber,
    ),
    index("vision_indexed_frames_video_timestamp_idx").on(
      table.videoId,
      table.timestampSeconds,
    ),
  ],
);

export const personDetectionsTable = pgTable(
  "vision_person_detections",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    frameId: uuid("frame_id")
      .notNull()
      .references(() => indexedFramesTable.id, { onDelete: "cascade" }),
    detectionIndex: integer("detection_index").notNull(),
    bbox: real("bbox").array().notNull(),
    detectorScore: real("detector_score").notNull(),
    cropPath: text("crop_path"),
    trackletId: text("tracklet_id"),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
  },
  (table) => [
    uniqueIndex("vision_person_detections_frame_index_idx").on(
      table.frameId,
      table.detectionIndex,
    ),
    index("vision_person_detections_tracklet_idx").on(table.trackletId),
  ],
);

export type IndexedFrameRecord = typeof indexedFramesTable.$inferSelect;
export type PersonDetectionRecord = typeof personDetectionsTable.$inferSelect;
