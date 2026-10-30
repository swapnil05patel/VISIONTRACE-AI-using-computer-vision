// scripts/demo_import_videos.cjs
// CommonJS version of demo script to import local video files into VisionTrace.
// Run via: pnpm run demo:import

const fs = require('fs');
const path = require('path');

const API_BASE = new URL(
  process.env.VISIONTRACE_API_BASE || 'http://127.0.0.1:8000'
).origin;

const TRAIN_DIR = path.resolve(
  process.env.DEMO_VIDEO_DIR ||
  path.join(process.cwd(), 'data/aicity_raw/MTMC_Tracking_2025/train')
);

const USER_ID = process.env.VISIONTRACE_DEV_USER_ID || 'dev-user';
const MANIFEST = path.join(process.cwd(), '.visiontrace-import-manifest.json');

const EXTENSIONS = new Set(['.mp4', '.mov', '.webm']);
const HEADERS = {
  'Content-Type': 'application/json',
  'x-visiontrace-dev-user': USER_ID,
};

const EXISTING_UPLOADS = {
  'Warehouse_001/videos/Camera_0000.mp4':
    '/objects/uploads/a3f4a43e-a5fc-4ed2-b781-a1256956a30b',
};

function mimeForFile(file) {
  switch (path.extname(file).toLowerCase()) {
    case '.mp4': return 'video/mp4';
    case '.mov': return 'video/quicktime';
    case '.webb': return 'video/webm';
    default: throw new Error(`Unsupported video type: ${file}`);
  }
}

function discoverVideos(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return discoverVideos(full);
    if (entry.isFile() && EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      return [full];
    }
    return [];
  }).sort();
}

function loadManifest() {
  if (!fs.existsSync(MANIFEST)) return {};
  return JSON.parse(fs.readFileSync(MANIFEST, 'utf8'));
}

function saveManifest(manifest) {
  const temp = `${MANIFEST}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(manifest, null, 2));
  fs.renameSync(temp, MANIFEST);
}

async function api(pathname, options = {}) {
  return fetch(`${API_BASE}${pathname}`, {
    ...options,
    headers: { ...HEADERS, ...(options.headers || {}) },
    signal: AbortSignal.timeout(30000),
  });
}

async function requestUploadUrl(name, size, contentType) {
  const response = await api('/api/storage/uploads/request-url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, size, contentType }),
  });

  if (!response.ok) {
    throw new Error(`Upload URL request: HTTP ${response.status} ${await response.text()}`);
  }

  return response.json();
}

async function uploadFile(uploadURL, filePath, contentType) {
  const absoluteURL = new URL(uploadURL, API_BASE).toString();
  const stats = fs.statSync(filePath);

  const response = await fetch(absoluteURL, {
    method: 'PUT',
    headers: {
      'Content-Type': contentType,
      'Content-Length': String(stats.size),
    },
    body: fs.createReadStream(filePath),
    duplex: 'half',
    signal: AbortSignal.timeout(60 * 60 * 1000),
  });

  if (!response.ok) {
    throw new Error(`Upload: HTTP ${response.status} ${await response.text()}`);
  }

  const result = await response.json();
  if (!result.objectPath) throw new Error('Upload response has no objectPath.');
  return result.objectPath;
}

async function registerVideo(name, objectPath) {
  const response = await api('/api/visiontrace/videos', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, objectPath }),
  });

  if (response.status === 409) {
    return { alreadyRegistered: true };
  }

  if (response.status !== 202 && !response.ok) {
    throw new Error(`Registration: HTTP ${response.status} ${await response.text()}`);
  }

  return response.json();
}

async function main() {
  if (!fs.existsSync(TRAIN_DIR)) {
    throw new Error(`Training directory not found: ${TRAIN_DIR}`);
  }

  const runtimeResponse = await fetch(`${API_BASE}/api/visiontrace/runtime`, {
    signal: AbortSignal.timeout(5000),
  });

  if (!runtimeResponse.ok) {
    throw new Error(`Runtime endpoint returned HTTP ${runtimeResponse.status}`);
  }

  const runtime = await runtimeResponse.json();
  if (runtime.status !== 'ready') {
    throw new Error(`ML runtime is "${runtime.status}". Start/fix the ML sidecar before importing.`);
  }

  const files = discoverVideos(TRAIN_DIR);
  const manifest = loadManifest();

  console.log(`Training directory: ${TRAIN_DIR}`);
  console.log(`API: ${API_BASE}`);
  console.log(`Videos discovered: ${files.length}`);
  console.log(`Manifest: ${MANIFEST}`);

  let registered = 0;
  let skipped = 0;
  let failed = 0;

  for (const filePath of files) {
    const relativePath = path.relative(TRAIN_DIR, filePath).split(path.sep).join('/');
    const name = path.basename(filePath);
    const stats = fs.statSync(filePath);

    let entry = manifest[relativePath];

    if (!entry && EXISTING_UPLOADS[relativePath]) {
      entry = { objectPath: EXISTING_UPLOADS[relativePath], state: 'uploaded' };
      manifest[relativePath] = entry;
      saveManifest(manifest);
    }

    if (entry?.state === 'registered') {
      skipped++;
      continue;
    }

    try {
      console.log(`\n[${registered + skipped + failed + 1}/${files.length}] ${relativePath}`);

      if (!entry?.objectPath) {
        const contentType = mimeForFile(filePath);
        console.log(`Uploading ${(stats.size / 1e6).toFixed(1)} MB...`);

        const uploadInfo = await requestUploadUrl(name, stats.size, contentType);
        const objectPath = await uploadFile(uploadInfo.uploadURL, filePath, contentType);

        entry = { objectPath, state: 'uploaded' };
        manifest[relativePath] = entry;
        saveManifest(manifest);
      } else {
        console.log('Reusing saved upload; not uploading the file again.');
      }

      const result = await registerVideo(name, entry.objectPath);
      entry.state = 'registered';
      entry.registeredAt = new Date().toISOString();
      entry.videoId = result.video?.id || null;
      saveManifest(manifest);

      registered++;
      console.log(
        result.alreadyRegistered
          ? 'Already registered; marked complete.'
          : `Registered and queued${entry.videoId ? `: ${entry.videoId}` : '.'}`
      );
    } catch (error) {
      failed++;
      console.error(`FAILED ${relativePath}: ${error.message}`);
      saveManifest(manifest);
    }
  }

  console.log('\nImport pass finished.');
  console.log(`Registered/queued this pass: ${registered}`);
  console.log(`Already complete: ${skipped}`);
  console.log(`Failed: ${failed}`);
  console.log(`Manifest saved at: ${MANIFEST}`);

  if (failed) process.exitCode = 1;
}

main().catch(error => {
  console.error(`Importer stopped: ${error.message}`);
  process.exitCode = 1;
});
