// scripts/demo_import_videos.js
// Demo script to import local video files into VisionTrace instance.
// It requests a private upload URL, uploads the video, then registers it.
// Run via: npm run demo:import

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Use node-fetch if available, otherwise fallback to built‑in https.
let fetch;
try {
  fetch = require('node-fetch');
} catch (e) {
  // Node 18+ has global fetch.
  fetch = global.fetch;
}

// Configuration – adjust as needed.
const API_BASE = process.env.VISIONTRACE_API_BASE || 'http://localhost:3000';
const DEMO_VIDEO_DIR = process.env.DEMO_VIDEO_DIR || '/home/swapnil/BUG PROOF/Captain Up/VisionAI/VisionTrace-AI-Lab/data/aicity_raw/MTMC_Tracking_2025/train/Warehouse_001/videos';

// Helper to determine MIME type for .mp4 files.
function mimeForFile(fileName) {
  const ext = path.extname(fileName).toLowerCase();
  if (ext === '.mp4') return 'video/mp4';
  if (ext === '.mov') return 'video/quicktime';
  if (ext === '.webm') return 'video/webm';
  return 'application/octet-stream';
}

async function requestUploadUrl(name, size, contentType) {
  const resp = await fetch(`${API_BASE}/api/storage/uploads/request-url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, size, contentType }),
  });
  if (!resp.ok) throw new Error(`Upload URL request failed: ${resp.status}`);
  const data = await resp.json();
  return data; // { uploadURL, objectPath, metadata }
}

async function uploadFile(uploadURL, filePath, contentType) {
  const fileStream = fs.createReadStream(filePath);
  const stats = fs.statSync(filePath);
  const resp = await fetch(uploadURL, {
    method: 'PUT',
    headers: { 'Content-Type': contentType, 'Content-Length': stats.size },
    body: fileStream,
  });
  if (!resp.ok) throw new Error(`PUT upload failed: ${resp.status}`);
  const result = await resp.json();
  return result.objectPath;
}

async function registerVideo(name, objectPath) {
  const resp = await fetch(`${API_BASE}/visiontrace/videos`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, objectPath }),
  });
  if (!resp.ok) throw new Error(`Register video failed: ${resp.status}`);
  const data = await resp.json();
  return data;
}

async function importVideos() {
  const files = fs.readdirSync(DEMO_VIDEO_DIR).filter(f => f.endsWith('.mp4'));
  console.log(`Found ${files.length} video(s) to import.`);
  for (const file of files) {
    try {
      const fullPath = path.join(DEMO_VIDEO_DIR, file);
      const stats = fs.statSync(fullPath);
      const mime = mimeForFile(file);
      console.log(`Processing ${file} (${(stats.size / 1e6).toFixed(2)} MB)`);

      const uploadInfo = await requestUploadUrl(file, stats.size, mime);
      await uploadFile(uploadInfo.uploadURL, fullPath, mime);
      const registration = await registerVideo(file, uploadInfo.objectPath);
      console.log(`✅ Registered: ${registration.video.id}`);
    } catch (err) {
      console.error(`❌ Failed for ${file}:`, err.message);
    }
  }
}

importVideos().catch(err => console.error('Fatal error:', err));
