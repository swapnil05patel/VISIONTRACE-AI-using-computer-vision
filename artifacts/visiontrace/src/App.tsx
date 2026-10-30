import { type ChangeEvent, type DragEvent, type FormEvent, type ReactNode, useEffect, useRef, useState } from 'react';
import { QueryClient, QueryClientProvider, useQueryClient } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import {
  getGetVisionTraceOverviewQueryKey,
  getListDatasetsQueryKey,
  getListVideosQueryKey,
  useGetVisionTraceOverview,
  useGetVisionRuntime,
  useHealthCheck,
  useListDatasets,
  useListVideos,
  useListCameras,
  useGetVideo,
  useListVideoFrames,
  useRequestUploadUrl,
  useRegisterVideo,
  useDetectPeopleInImage,
  useSearchImage,
  useSearchVideo,
  useGetProcessingJob,
  useStartDatasetImport,
} from '@workspace/api-client-react';
import type { DatasetAdapter, PersonDetection, ProcessingJob, SearchHit, SearchResponse } from '@workspace/api-client-react';
import { Link, Route, Router as WouterRouter, Switch, useLocation } from 'wouter';
import {
  Activity, AlertTriangle, Archive, ArrowDownToLine, ChevronRight,
  Clock3, Cpu, Database, FileVideo2, Fingerprint, FolderOpen, ImagePlus, LoaderCircle,
  Search, ShieldCheck, SlidersHorizontal, Upload, Users, Video, ScanSearch, BarChart2, CheckCircle2, Circle, Layers,
} from 'lucide-react';
import NotFound from '@/pages/not-found';
import './index.css';

const queryClient = new QueryClient();
type QueryFilters = { cameraId?: string; from?: string; to?: string; topK: number };

function failureMessage(error: unknown) {
  if (error && typeof error === 'object' && 'message' in error) return String(error.message);
  return 'The request could not be completed. Check service availability and try again.';
}
function formatDate(value?: string | null) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
}
function duration(value?: number | null) {
  if (value == null) return '—';
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const seconds = Math.floor(value % 60);
  return hours ? `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}` : `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}
function statusClass(status: string) {
  return status === 'failed' || status === 'unavailable' ? 'failed' : ['queued', 'processing', 'running', 'partial'].includes(status) ? 'processing' : '';
}
function StatusBadge({ status }: { status: string }) {
  return <span className={`badge ${statusClass(status)}`} data-testid={`status-${status}`}><i className={`status-dot ${statusClass(status)}`} />{status.replaceAll('_', ' ')}</span>;
}
function SkeletonBlock() {
  return <div className="panel panel-body" aria-label="Loading"><div className="skeleton skeleton-line" /><div className="skeleton skeleton-line" /><div className="skeleton skeleton-line" /></div>;
}
function Shell({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  const runtime = useGetVisionRuntime({ query: { refetchInterval: 30000 } });
  const health = useHealthCheck({ query: { refetchInterval: 30000 } });
  const runtimeStatus = runtime.data?.status ?? (health.isError ? 'unavailable' : 'partial');
  const links = [
    { href: '/', title: 'Search workspace', icon: Search },
    { href: '/gallery', title: 'Detected people', icon: Users },
    { href: '/pipeline', title: 'Processing pipeline', icon: Cpu },
    { href: '/library', title: 'Video library', icon: FolderOpen },
    { href: '/jobs', title: 'Processing jobs', icon: Activity },
    { href: '/datasets', title: 'Dataset adapters', icon: Database },
    { href: '/models', title: 'Model registry', icon: ShieldCheck },
  ];
  return <div className="app-shell">
    <aside className="sidebar">
      <Link href="/" className="brand"><span className="brand-mark"><ScanSearch size={18} /></span><span className="brand-name">Vision<span>Trace</span></span></Link>
      <div className="nav-label">Investigation</div>
      <nav className="nav-list">{links.map(({ href, title, icon: Icon }) => <Link key={href} href={href} className={`nav-item ${location === href ? 'active' : ''}`} data-testid={`link-${href === '/' ? 'search' : href.slice(1)}`}><Icon /><span>{title}</span></Link>)}</nav>
      <div className="side-bottom">
        <div className="case-card"><ShieldCheck size={16} color="#9fc7b9" /><strong>Authorized footage only</strong><p>Use within your documented scope of access. Similarity is a navigation aid, not identity evidence.</p></div>
        <div className="side-foot">VISIONTRACE · AUDITABLE SEARCH</div>
      </div>
    </aside>
    <main className="main-area">
      <header className="topbar"><div className="crumb">Workspace <ChevronRight size={12} style={{ verticalAlign: 'middle', margin: '0 5px' }} /> {links.find(item => item.href === location)?.title ?? 'Investigation'}</div><div className="top-right"><div className="runtime-pill"><i className={`status-dot ${statusClass(runtimeStatus)}`} />Runtime {runtimeStatus}<span className="mono">/</span>{runtime.data?.device ?? 'checking'}</div><div className="top-user">VT</div></div></header>
      {children}
    </main>
  </div>;
}
function AppFrame() {
  return <Shell><Switch>
    <Route path="/" component={WorkspacePage} />
    <Route path="/gallery" component={GalleryPage} />
    <Route path="/pipeline" component={PipelineStatusPage} />
    <Route path="/library" component={LibraryPage} />
    <Route path="/jobs" component={JobsPage} />
    <Route path="/datasets" component={DatasetsPage} />
    <Route path="/models" component={ModelsPage} />
    <Route component={NotFound} />
  </Switch></Shell>;
}
function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}
function App() {
  return <QueryClientProvider client={queryClient}><TooltipProvider><WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}><RoutedErrorBoundary><AppFrame /></RoutedErrorBoundary></WouterRouter><Toaster /></TooltipProvider></QueryClientProvider>;
}

function WorkspacePage() {
  const queryClient = useQueryClient();
  const overview = useGetVisionTraceOverview({ query: { refetchInterval: 30000 } });
  const cameras = useListCameras();
  const uploadMutation = useRequestUploadUrl();
  const detectMutation = useDetectPeopleInImage();
  const imageSearchMutation = useSearchImage();
  const videoSearchMutation = useSearchVideo();
  const pollJob = useGetProcessingJob();
  const [mode, setMode] = useState<'image' | 'video'>('image');
  const [file, setFile] = useState<File | null>(null);
  const [objectPath, setObjectPath] = useState('');
  const [preview, setPreview] = useState('');
  const [detections, setDetections] = useState<PersonDetection[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [job, setJob] = useState<ProcessingJob | null>(null);
  const [cameraId, setCameraId] = useState('');
  const [from, setFrom] = useState('');
  const [to, setTo] = useState('');
  const [topK, setTopK] = useState('20');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!job || ['completed', 'failed'].includes(job.status)) return;
    const timer = window.setTimeout(() => pollJob.mutate({ data: { jobId: job.id } }, {
      onSuccess: next => { setJob(next); if (next.result) setResults(next.result); },
      onError: err => setError(failureMessage(err)),
    }), 2400);
    return () => window.clearTimeout(timer);
  }, [job, pollJob.mutate]);
  const chooseFile = (event: ChangeEvent<HTMLInputElement>) => {
    const next = event.target.files?.[0] ?? null;
    if (!next) return;
    setFile(next); setObjectPath(''); setDetections([]); setSelected(null); setResults(null); setJob(null); setError('');
    setPreview(URL.createObjectURL(next));
  };
  const filters = (): QueryFilters => ({ ...(cameraId ? { cameraId } : {}), ...(from ? { from } : {}), ...(to ? { to } : {}), topK: Math.max(1, Math.min(100, Number(topK) || 20)) });
  const upload = async () => {
    if (!file) return '';
    const signed = await uploadMutation.mutateAsync({ data: { name: file.name, size: file.size, contentType: file.type || 'application/octet-stream' } });
    const put = await fetch(signed.uploadURL, { method: 'PUT', credentials: 'include', headers: { 'Content-Type': file.type || 'application/octet-stream', ...(import.meta.env.VITE_VISIONTRACE_DEV_USER_ID ? { 'x-visiontrace-dev-user': import.meta.env.VITE_VISIONTRACE_DEV_USER_ID } : {}) }, body: file });
    if (!put.ok) throw new Error(`Upload failed with status ${put.status}.`);
    setObjectPath(signed.objectPath);
    return signed.objectPath;
  };
  const runSearch = async () => {
    if (!file) return;
    setBusy(true); setError(''); setResults(null);
    try {
      const path = objectPath || await upload();
      if (mode === 'image') {
        let people = detections;
        if (!people.length) {
          const detected = await detectMutation.mutateAsync({ data: { objectPath: path } });
          people = detected.detections; setDetections(people);
        }
        if (selected == null && people.length) setSelected(people[0].detectionIndex);
        const selection = selected ?? people[0]?.detectionIndex;
        if (selection == null) throw new Error('No person detections were returned for this image.');
        const response = await imageSearchMutation.mutateAsync({ data: { objectPath: path, detectionIndex: selection, filters: filters() } });
        setResults(response);
      } else {
        const nextJob = await videoSearchMutation.mutateAsync({ data: { objectPath: path, filters: filters() } });
        setJob(nextJob);
        if (nextJob.result) setResults(nextJob.result);
      }
    } catch (err) { setError(failureMessage(err)); }
    finally { setBusy(false); queryClient.invalidateQueries({ queryKey: getGetVisionTraceOverviewQueryKey() }); }
  };
  const active = busy || (job != null && !['completed', 'failed'].includes(job.status));
  const onDrop = (event: DragEvent<HTMLLabelElement>) => { event.preventDefault(); const dropped = event.dataTransfer.files[0]; if (dropped) { const input = document.getElementById('query-file') as HTMLInputElement; const transfer = new DataTransfer(); transfer.items.add(dropped); input.files = transfer.files; input.dispatchEvent(new Event('change', { bubbles: true })); } };
  return <div className="page">
    <div className="page-heading"><div><div className="eyebrow">Visual investigation · authorized sources</div><h1>Find appearances.<br />Follow the evidence.</h1><p className="subtitle">Search indexed recordings with a person query. Every result links back to its source and frame.</p></div><div className="runtime-pill"><i className={`status-dot ${statusClass(overview.data?.runtime.status ?? 'partial')}`} />{overview.data?.runtime.status ?? 'Runtime status loading'}</div></div>
    <div className="stats-strip">
      {[['Indexed videos', overview.data?.indexedVideos], ['Indexed frames', overview.data?.indexedFrames], ['Indexed people', overview.data?.indexedPeople], ['Active jobs', overview.data?.activeJobs]].map(([label, value]) => <div className="stat-box" key={String(label)}><div className="stat-label">{label}</div><div className="stat-value mono">{overview.isLoading ? '—' : value ?? '—'}</div><div className="stat-foot">Live index status</div></div>)}
    </div>
    {overview.data?.runtime.limitations?.length ? <div className="notice" style={{ marginBottom: 16 }}><strong>Runtime limitations</strong><br />{overview.data.runtime.limitations.join(' · ')}</div> : null}
    {overview.isError && <div className="toast-error" role="alert">Workspace status unavailable. {failureMessage(overview.error)}</div>}
    <div className="workspace-grid">
      <section className="panel">
        <div className="panel-head"><h2>Query footage</h2><span className="mono" style={{ fontSize: 10, color: '#7b8682' }}>01 / SOURCE</span></div>
        <div className="panel-body">
          <div className="query-tabs"><button className={`query-tab ${mode === 'image' ? 'active' : ''}`} onClick={() => { setMode('image'); setFile(null); setResults(null); setPreview(''); setObjectPath(''); }} data-testid="button-image-mode"><ImagePlus size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />Image</button><button className={`query-tab ${mode === 'video' ? 'active' : ''}`} onClick={() => { setMode('video'); setFile(null); setResults(null); setPreview(''); setObjectPath(''); }} data-testid="button-video-mode"><Video size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />Video</button></div>
          {preview ? <div className="query-preview">{mode === 'image' ? <img src={preview} alt="Uploaded query" className="query-preview" /> : <video src={preview} controls />}</div> : null}
          <input id="query-file" type="file" accept={mode === 'image' ? 'image/*' : 'video/*'} onChange={chooseFile} hidden data-testid="input-query-file" />
          {!preview && <label htmlFor="query-file" className="drop-zone" onDragOver={event => event.preventDefault()} onDrop={onDrop} data-testid="drop-query-file"><div><span className="drop-icon"><Upload size={17} /></span><strong>Drop {mode === 'image' ? 'an image' : 'a video'} to begin</strong><small>or browse your authorized evidence files</small></div></label>}
          {file && <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 10, marginTop: 8 }}><span>{file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB</span><button className="text-action" onClick={() => { setFile(null); setPreview(''); setObjectPath(''); }}>Remove</button></div>}
          {mode === 'image' && detections.length > 0 && <><div className="divider" /><div className="section-kicker">Select a detected person</div><div className="detect-list">{detections.map(detection => <button key={detection.id} className={`detect-option ${selected === detection.detectionIndex ? 'selected' : ''}`} onClick={() => setSelected(detection.detectionIndex)} data-testid={`detection-${detection.detectionIndex}`}><div className="detect-swatch">{detection.cropPath ? <img src={detection.cropPath} alt={`Detector crop ${detection.detectionIndex + 1}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : <ScanSearch size={18} />}</div><small>Detection {detection.detectionIndex + 1} · {(detection.detectorScore * 100).toFixed(0)}% detector score</small></button>)}</div></>}
          <div className="divider" /><div className="section-kicker"><SlidersHorizontal size={12} style={{ verticalAlign: 'middle', marginRight: 5 }} />Search scope</div>
          <div className="field"><label htmlFor="camera-filter">Camera source</label><select id="camera-filter" className="control" value={cameraId} onChange={e => setCameraId(e.target.value)} data-testid="select-camera"><option value="">All indexed cameras</option>{(cameras.data ?? []).map(camera => <option value={camera.id} key={camera.id}>{camera.id} · {camera.videoCount} videos</option>)}</select></div>
          <div className="form-row"><div className="field"><label htmlFor="date-from">From date</label><input id="date-from" className="control" type="date" value={from} onChange={e => setFrom(e.target.value)} data-testid="input-date-from" /></div><div className="field"><label htmlFor="date-to">To date</label><input id="date-to" className="control" type="date" value={to} onChange={e => setTo(e.target.value)} data-testid="input-date-to" /></div></div>
          <div className="field"><label htmlFor="top-k">Maximum results</label><select id="top-k" className="control" value={topK} onChange={e => setTopK(e.target.value)} data-testid="select-top-k">{[10, 20, 50, 100].map(count => <option key={count} value={count}>{count} frames</option>)}</select></div>
          {error && <div className="toast-error" role="alert">{error}</div>}
          <button className="btn btn-primary btn-block" onClick={runSearch} disabled={!file || active || overview.data?.runtime.status === 'unavailable'} data-testid="button-run-search">{active ? <LoaderCircle size={15} className="spin" /> : <Search size={15} />}{active ? (job ? `Searching · ${job.progress}%` : 'Preparing query…') : mode === 'image' ? 'Find similar appearances' : 'Search from video'}</button>
          <div className="notice" style={{ marginTop: 12 }}>Visual similarity is a ranking signal for navigating footage. It does not verify identity. Confirm each appearance against its source recording.</div>
        </div>
      </section>
      <section className="panel">
        <div className="result-toolbar"><div><strong>Matching frames</strong><div style={{ fontSize: 10, color: '#7d8985', marginTop: 4 }}>{results ? `${results.resultCount} returned · ${results.queryMode} query` : 'Results appear here after a search'}</div></div><span>{results ? results.queryMode.toUpperCase() : 'NO ACTIVE QUERY'}</span></div>
        {results?.rankingNote && <div className="source-note" style={{ borderTop: 0, borderBottom: '1px solid hsl(var(--border))' }}>{results.rankingNote}</div>}
        {active && !results ? <div className="empty-state"><div><div className="empty-symbol"><LoaderCircle size={20} className="spin" /></div><h3>Working through indexed footage</h3><p>{job?.stage ?? 'Preparing the query and searching the available index.'}</p>{job && <div className="progress-track" style={{ marginTop: 15 }}><div className="progress-fill" style={{ transform: `scaleX(${Math.max(0, Math.min(100, job.progress)) / 100})` }} /></div>}</div></div>
          : results?.results.length ? <div className="results-grid">{results.results.map(hit => <HitCard key={hit.id} hit={hit} />)}</div>
          : <div className="empty-state"><div><div className="empty-symbol"><Fingerprint size={21} /></div><h3>{results ? 'No matching frames returned' : 'Start with a query'}</h3><p>{results ? 'The search completed without matching appearances. No results have been added or inferred.' : 'Upload a still or clip, choose search scope, then inspect source-linked frames.'}</p></div></div>}
        {job?.status === 'failed' && <div className="toast-error" style={{ margin: '0 14px 14px' }}>Search job failed: {job.error || 'No error detail was returned.'}</div>}
      </section>
    </div>
  </div>;
}
function HitCard({ hit }: { hit: SearchHit }) {
  const [open, setOpen] = useState(false);
  const videoDetails = useGetVideo();
  const frames = useListVideoFrames();
  const playerRef = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (open) {
      videoDetails.mutate({ data: { videoId: hit.videoId } });
      frames.mutate({ data: { videoId: hit.videoId, limit: 40 } });
    }
  }, [open, hit.videoId]);
  return <>
    <article className="hit-card" onClick={() => setOpen(true)} data-testid={`card-hit-${hit.id}`} role="button" tabIndex={0} onKeyDown={event => { if (event.key === 'Enter') setOpen(true); }}>
      <div className="hit-image">{hit.thumbnailPath ? <img src={hit.thumbnailPath} alt={`Frame ${hit.frameNumber} from ${hit.sourceName}`} /> : <div className="empty-symbol"><FileVideo2 size={21} /></div>}<span className="frame-label">FRAME {hit.frameNumber}</span><span className="sim-chip">Visual similarity · {hit.similarity.toFixed(3)}</span></div>
      <div className="hit-meta"><div className="hit-title">{hit.sourceName}</div><div className="hit-sub"><span>{hit.cameraId ?? 'Camera unavailable'}</span><span>{duration(hit.timestampSeconds)}</span></div></div>
    </article>
    {open && <div role="dialog" aria-modal="true" onClick={() => setOpen(false)} style={{ position: 'fixed', inset: 0, zIndex: 50, background: '#172b2dcc', display: 'grid', placeItems: 'center', padding: 18 }}><div className="panel" onClick={e => e.stopPropagation()} style={{ width: 'min(760px,100%)', maxHeight: '90vh', overflow: 'auto', padding: 18 }}><div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}><div><div className="eyebrow">Source evidence</div><strong>{videoDetails.data?.name ?? hit.sourceName}</strong></div><button className="btn" onClick={() => setOpen(false)}>Close</button></div><video ref={playerRef} src={hit.videoPath} controls preload="metadata" style={{ width: '100%', maxHeight: '55vh', background: '#1c2828' }} onLoadedMetadata={e => { e.currentTarget.currentTime = hit.timestampSeconds; }} /><div className="source-note" style={{ paddingLeft: 0 }}>Camera {videoDetails.data?.cameraId ?? hit.cameraId ?? 'not supplied'} · Frame {hit.frameNumber} · {duration(hit.timestampSeconds)} · visual similarity {hit.similarity.toFixed(3)}</div><div className="section-kicker" style={{ marginTop: 12 }}>Indexed frames · select to seek source</div>{frames.isPending ? <div className="skeleton skeleton-line" /> : frames.data?.length ? <div style={{ display: 'flex', gap: 8, overflowX: 'auto', paddingBottom: 5 }}>{frames.data.map(frame => <button key={frame.id} className="detect-option" style={{ minWidth: 100 }} onClick={() => { if (playerRef.current) playerRef.current.currentTime = frame.timestampSeconds; }}><div className="detect-swatch" style={{ height: 55 }}>{frame.thumbnailPath ? <img src={frame.thumbnailPath} alt={`Frame ${frame.frameNumber}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : <FileVideo2 size={17} />}</div><small>Frame {frame.frameNumber} · {duration(frame.timestampSeconds)}</small></button>)}</div> : <p className="subtitle">{frames.isError ? 'Indexed frames could not be loaded.' : 'No indexed frame records were returned for this source.'}</p>}</div></div>}
  </>;
}

function LibraryPage() {
  const queryClient = useQueryClient();
  const videos = useListVideos({ query: { refetchInterval: 12000 } });
  const uploadMutation = useRequestUploadUrl();
  const register = useRegisterVideo();
  const [file, setFile] = useState<File | null>(null);
  const [cameraId, setCameraId] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const submit = async (event: FormEvent) => {
    event.preventDefault(); if (!file) return;
    setError(''); setNotice('');
    try {
      const signed = await uploadMutation.mutateAsync({ data: { name: file.name, size: file.size, contentType: file.type || 'application/octet-stream' } });
      const response = await fetch(signed.uploadURL, { method: 'PUT', credentials: 'include', headers: { 'Content-Type': file.type || 'application/octet-stream', ...(import.meta.env.VITE_VISIONTRACE_DEV_USER_ID ? { 'x-visiontrace-dev-user': import.meta.env.VITE_VISIONTRACE_DEV_USER_ID } : {}) }, body: file });
      if (!response.ok) throw new Error(`Upload failed with status ${response.status}.`);
      const registered = await register.mutateAsync({ data: { name: file.name, objectPath: signed.objectPath, cameraId: cameraId || null } });
      setNotice(`Source registered. Indexing job ${registered.job.id} is ${registered.job.status}.`);
      setFile(null); setCameraId('');
      await Promise.all([queryClient.invalidateQueries({ queryKey: getListVideosQueryKey() }), queryClient.invalidateQueries({ queryKey: getGetVisionTraceOverviewQueryKey() })]);
    } catch (err) { setError(failureMessage(err)); }
  };
  return <div className="page"><div className="page-heading"><div><div className="eyebrow">Source archive</div><h1>Video library</h1><p className="subtitle">Registered recordings, indexing state, and source metadata.</p></div><span className="runtime-pill"><Archive size={14} />{videos.data?.length ?? '—'} source videos</span></div>
    <form className="panel panel-body" onSubmit={submit} style={{ marginBottom: 16 }}><div style={{ display: 'flex', gap: 15, alignItems: 'center', flexWrap: 'wrap' }}><div style={{ flex: '1 1 200px' }}><div className="section-kicker">Register authorized source</div><input className="control" type="file" accept="video/*" onChange={e => setFile(e.target.files?.[0] ?? null)} data-testid="input-library-video" /></div><div style={{ flex: '0 1 210px' }}><div className="section-kicker">Camera ID · optional</div><input className="control" value={cameraId} onChange={e => setCameraId(e.target.value)} placeholder="e.g. north-entry" data-testid="input-library-camera" /></div><button className="btn btn-primary" type="submit" disabled={!file || register.isPending || uploadMutation.isPending} data-testid="button-register-video">{register.isPending || uploadMutation.isPending ? <LoaderCircle size={14} /> : <Upload size={14} />}Upload & index</button></div>{error && <div className="toast-error">{error}</div>}{notice && <div className="notice" style={{ marginTop: 12 }}>{notice}</div>}<div className="source-note" style={{ paddingLeft: 0, paddingBottom: 0 }}>Only add footage you are authorized to process. Uploading stores the source and queues indexing.</div></form>
    <section className="panel"><div className="panel-head"><h2>Registered source recordings</h2><button className="btn" onClick={() => videos.refetch()} data-testid="button-refresh-library"><Activity size={13} />Refresh</button></div>
      {videos.isLoading ? <SkeletonBlock /> : videos.isError ? <div className="empty-state"><div><div className="empty-symbol"><AlertTriangle size={20} /></div><h3>Library could not load</h3><p>{failureMessage(videos.error)}</p><button className="btn" style={{ marginTop: 12 }} onClick={() => videos.refetch()}>Retry</button></div></div> : !videos.data?.length ? <div className="empty-state"><div><div className="empty-symbol"><FolderOpen size={20} /></div><h3>No source videos registered</h3><p>Add an authorized source recording above. Indexing status will be shown here.</p></div></div>
      : <div className="table-wrap"><table><thead><tr><th>Source video</th><th>Camera</th><th>Duration</th><th>Index status</th><th>Frames</th><th>People</th><th>Added</th></tr></thead><tbody>{videos.data.map(video => <tr key={video.id} data-testid={`row-video-${video.id}`}><td><div className="video-name"><span className="video-icon"><FileVideo2 size={15} /></span>{video.name}</div></td><td className="mono">{video.cameraId ?? '—'}</td><td className="mono">{duration(video.durationSeconds)}</td><td><StatusBadge status={video.status} /></td><td className="mono">{video.frameCount}</td><td className="mono">{video.personCount}</td><td>{formatDate(video.createdAt)}</td></tr>)}</tbody></table></div>}
    </section>
  </div>;
}

function JobsPage() {
  const overview = useGetVisionTraceOverview({ query: { refetchInterval: 5000 } });
  const jobs = overview.data?.recentJobs ?? [];
  return <div className="page"><div className="page-heading"><div><div className="eyebrow">Persistent processing record</div><h1>Processing jobs</h1><p className="subtitle">Indexing, dataset import, and video-search work reported by the service.</p></div><button className="btn" onClick={() => overview.refetch()} data-testid="button-refresh-jobs"><Activity size={14} />Refresh status</button></div>
    <div className="stats-strip"><div className="stat-box"><div className="stat-label">Active</div><div className="stat-value mono">{overview.data?.activeJobs ?? '—'}</div><div className="stat-foot">Service reported</div></div><div className="stat-box"><div className="stat-label">Recent jobs</div><div className="stat-value mono">{jobs.length || (overview.isLoading ? '—' : '0')}</div><div className="stat-foot">Returned by overview</div></div><div className="stat-box"><div className="stat-label">Updated</div><div className="stat-value" style={{ fontSize: 16 }}>{overview.data ? formatDate(overview.data.runtime.verifiedAt) : '—'}</div><div className="stat-foot">Runtime verification</div></div><div className="stat-box"><div className="stat-label">Runtime</div><div className="stat-value" style={{ fontSize: 16, textTransform: 'capitalize' }}>{overview.data?.runtime.status ?? '—'}</div><div className="stat-foot">Inference capability</div></div></div>
    <section className="panel"><div className="panel-head"><h2>Recent processing history</h2><span className="mono" style={{ fontSize: 10, color: '#7b8682' }}>AUTO-REFRESH · 5 SEC</span></div>
      {overview.isLoading ? <SkeletonBlock /> : overview.isError ? <div className="empty-state"><div><div className="empty-symbol"><AlertTriangle size={20} /></div><h3>Job status unavailable</h3><p>{failureMessage(overview.error)}</p><button className="btn" style={{ marginTop: 12 }} onClick={() => overview.refetch()}>Retry</button></div></div> : !jobs.length ? <div className="empty-state"><div><div className="empty-symbol"><Clock3 size={20} /></div><h3>No recent jobs returned</h3><p>When the service reports processing activity, job progress and failures appear here.</p></div></div> : <div className="job-list">{jobs.map(job => <JobRow job={job} key={job.id} />)}</div>}
      <div className="source-note">Job records and progress are sourced from the VisionTrace overview response. Errors are shown exactly as returned by the service.</div>
    </section>
  </div>;
}
function JobRow({ job }: { job: ProcessingJob }) {
  const Icon = job.kind === 'video_index' ? FileVideo2 : job.kind === 'dataset_import' ? Database : Search;
  return <div className="job-row" data-testid={`row-job-${job.id}`}><span className="job-icon"><Icon size={15} /></span><div className="job-main"><strong>{job.sourceName || job.kind.replaceAll('_', ' ')}</strong><small className="mono">{job.id} · {formatDate(job.createdAt)}</small>{job.error && <div className="error-note">{job.error}</div>}</div><div className="job-stage">{job.stage || job.status}</div><div className="progress-track"><div className="progress-fill" style={{ transform: `scaleX(${Math.max(0, Math.min(100, job.progress)) / 100})` }} /></div><StatusBadge status={job.status} /></div>;
}

function DatasetsPage() {
  const queryClient = useQueryClient();
  const datasets = useListDatasets({ query: { refetchInterval: 20000 } });
  const startImport = useStartDatasetImport();
  const [datasetId, setDatasetId] = useState('');
  const [split, setSplit] = useState('train');
  const [purpose, setPurpose] = useState('benchmark');
  const [maxItems, setMaxItems] = useState('100');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const importDataset = async (dataset: DatasetAdapter) => {
    if (!dataset.configured || dataset.adapterStatus !== 'supported') return;
    const limit = Math.max(1, Math.min(10000, Number(maxItems) || 100));
    setDatasetId(dataset.id); setError(''); setSuccess('');
    try {
      const job = await startImport.mutateAsync({ data: { datasetId: dataset.id, split: split as 'train' | 'query' | 'gallery' | 'test', purpose: purpose as 'benchmark' | 'index', maxItems: limit } });
      setSuccess(`Import job ${job.id} started · ${job.status}. Track it in Processing jobs.`);
      await Promise.all([queryClient.invalidateQueries({ queryKey: getListDatasetsQueryKey() }), queryClient.invalidateQueries({ queryKey: getGetVisionTraceOverviewQueryKey() })]);
    } catch (err) { setError(failureMessage(err)); }
    finally { setDatasetId(''); }
  };
  const known = datasets.data ?? [];
  return <div className="page"><div className="page-heading"><div><div className="eyebrow">Adapter access & bounded imports</div><h1>Dataset adapters</h1><p className="subtitle">Inspect service-reported access before importing a bounded dataset slice.</p></div><div className="runtime-pill"><Database size={14} />Configured access only</div></div>
    {error && <div className="toast-error" role="alert">{error}</div>}{success && <div className="notice" style={{ marginBottom: 14 }}>{success}</div>}
    {datasets.isLoading ? <div className="dataset-grid"><SkeletonBlock /><SkeletonBlock /><SkeletonBlock /></div> : datasets.isError ? <div className="panel empty-state"><div><div className="empty-symbol"><AlertTriangle size={20} /></div><h3>Adapter status unavailable</h3><p>{failureMessage(datasets.error)}</p><button className="btn" style={{ marginTop: 12 }} onClick={() => datasets.refetch()}>Retry</button></div></div> : !known.length ? <div className="panel empty-state"><div><div className="empty-symbol"><Database size={20} /></div><h3>No adapter records returned</h3><p>Dataset availability cannot be assumed. Wait for the service to report adapter access.</p></div></div> : <div className="dataset-grid">{known.map(dataset => <article className="panel dataset-card" key={dataset.id} data-testid={`card-dataset-${dataset.id}`}><div className="dataset-top"><span className="badge">{dataset.adapterStatus}</span><span className="dataset-count">{dataset.importedItems} imported</span></div><h2>{dataset.name}</h2><p>{dataset.accessNote}</p><div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#66746e', marginBottom: 15 }}><span>Adapter {dataset.id}</span><span>{dataset.configured ? 'Root configured' : 'Not configured'}</span></div><button className="btn btn-primary" style={{ width: '100%' }} disabled={!dataset.configured || dataset.adapterStatus !== 'supported' || !!datasetId || startImport.isPending} onClick={() => importDataset(dataset)} data-testid={`button-import-${dataset.id}`}>{datasetId === dataset.id ? <LoaderCircle size={13} /> : <ArrowDownToLine size={13} />}{dataset.configured && dataset.adapterStatus === 'supported' ? 'Start bounded import' : 'Import unavailable'}</button></article>)}</div>}
    <section className="panel" style={{ marginTop: 16 }}><div className="panel-head"><h2>Import parameters</h2><span className="mono" style={{ fontSize: 10, color: '#7b8682' }}>LIMITS ENFORCED</span></div><div className="panel-body"><div className="form-row" style={{ marginTop: 0 }}><div className="field"><label htmlFor="dataset-split">Dataset split</label><select className="control" id="dataset-split" value={split} onChange={e => setSplit(e.target.value)} data-testid="select-dataset-split">{['train', 'query', 'gallery', 'test'].map(item => <option value={item} key={item}>{item}</option>)}</select></div><div className="field"><label htmlFor="dataset-purpose">Purpose</label><select className="control" id="dataset-purpose" value={purpose} onChange={e => setPurpose(e.target.value)} data-testid="select-dataset-purpose"><option value="benchmark">Benchmark</option><option value="index">Index</option></select></div><div className="field"><label htmlFor="dataset-limit">Maximum items · 1–10,000</label><input className="control" id="dataset-limit" type="number" min="1" max="10000" value={maxItems} onChange={e => setMaxItems(e.target.value)} data-testid="input-dataset-limit" /></div></div><div className="notice">Imports require a service-reported configured adapter and supported status. A bounded import creates a persistent processing job; no dataset is assumed available unless the API confirms it.</div></div></section>
    <div className="source-note" style={{ paddingLeft: 0 }}>Benchmark datasets such as MARS, MSMT17, and Market-1501 are adapter integrations, not identity verification systems. Imported data remains subject to the adapter access note shown above.</div>
  </div>;
}

function ModelsPage() {
  const [models, setModels] = useState<Array<Record<string, unknown>>>([]);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const load = async () => {
    setError('');
    try {
      const response = await fetch('/api/visiontrace/models', { credentials: 'include', headers: import.meta.env.VITE_VISIONTRACE_DEV_USER_ID ? { 'x-visiontrace-dev-user': import.meta.env.VITE_VISIONTRACE_DEV_USER_ID } : {} });
      if (!response.ok) throw new Error(`Model registry returned ${response.status}.`);
      setModels(await response.json());
    } catch (err) {
      setError(failureMessage(err));
    }
  };
  useEffect(() => { void load(); }, []);
  const activate = async (modelId: string) => {
    setError(''); setNotice('');
    const response = await fetch('/api/visiontrace/models/activate', { method: 'POST', credentials: 'include', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ modelId }) });
    if (!response.ok) { setError(`Activation failed (${response.status}).`); return; }
    setNotice('Active model updated. Existing FAISS indexes remain valid only for the same embedding space.');
    await load();
  };
  const rollback = async () => {
    const response = await fetch('/api/visiontrace/models/rollback', { method: 'POST', credentials: 'include' });
    if (!response.ok) { setError(`Rollback failed (${response.status}).`); return; }
    setNotice('Rolled back to the previously active model.');
    await load();
  };
  return <div className="page"><div className="page-heading"><div><div className="eyebrow">Administrator workflow</div><h1>Model registry</h1><p className="subtitle">Verified artifacts only. Training never silently replaces the active production model.</p></div><button className="btn" onClick={() => rollback()}>Rollback</button></div>
    {error && <div className="toast-error">{error}</div>}{notice && <div className="notice">{notice}</div>}
    <section className="panel"><div className="panel-head"><h2>Registered artifacts</h2><button className="btn" onClick={() => load()}>Refresh</button></div>
      {!models.length ? <div className="empty-state"><div><h3>No registry rows yet</h3><p>Run model verification, then register artifacts with checksums outside git. Active detector and Re-ID hashes also live in data/active-models.json.</p></div></div>
        : <div className="table-wrap"><table><thead><tr><th>Name</th><th>Type</th><th>Status</th><th>Checksum</th><th></th></tr></thead><tbody>{models.map(model => <tr key={String(model.id)}><td>{String(model.name)}</td><td>{String(model.type)}</td><td><StatusBadge status={String(model.status)} /></td><td className="mono">{String(model.checksum).slice(0, 16)}</td><td><button className="btn" disabled={Boolean(model.active) || !model.verified} onClick={() => activate(String(model.id))}>Activate</button></td></tr>)}</tbody></table></div>}
    </section>
  </div>;
}

// ─── Detected People Gallery ──────────────────────────────────────────────────
type GalleryDetection = {
  id: string; detectionIndex: number; bbox: number[]; detectorScore: number;
  cropPath: string | null; trackletId: string | null;
  frameNumber: number; timestampSeconds: number; thumbnailPath: string | null;
  videoId: string; videoName: string; cameraId: string | null;
  videoPath: string; frameId: string;
};
type GalleryVideo = { id: string; name: string; cameraId: string | null; personCount: number; frameCount: number; durationSeconds: number | null; status: string };
type GalleryResponse = { detections: GalleryDetection[]; total: number; page: number; pageSize: number; pages: number };
type SearchByDetectionResult = { id: string; videoId: string; sourceName: string; cameraId: string | null; frameNumber: number; timestampSeconds: number; similarity: number; thumbnailPath: string | null; videoPath: string; bbox: number[]; detectorScore: number; detectionId: string; trackletId: string | null };

const devHeaders = () => import.meta.env.VITE_VISIONTRACE_DEV_USER_ID ? { 'x-visiontrace-dev-user': import.meta.env.VITE_VISIONTRACE_DEV_USER_ID } : {};

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { credentials: 'include', headers: { ...devHeaders(), ...(init?.headers ?? {}) }, ...init });
  const payload = await res.json() as { error?: string } & T;
  if (!res.ok) throw new Error((payload as { error?: string }).error || `Request failed (${res.status})`);
  return payload as T;
}

function GalleryPage() {
  const [videos, setVideos] = useState<GalleryVideo[]>([]);
  const [detections, setDetections] = useState<GalleryDetection[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [filterVideoId, setFilterVideoId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchTarget, setSearchTarget] = useState<GalleryDetection | null>(null);
  const [searchResults, setSearchResults] = useState<SearchByDetectionResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [rankingNote, setRankingNote] = useState('');
  const PAGE_SIZE = 48;

  const loadGallery = async (p: number, videoId: string) => {
    setLoading(true); setError('');
    try {
      const params = new URLSearchParams({ page: String(p), pageSize: String(PAGE_SIZE) });
      if (videoId) params.set('videoId', videoId);
      const data = await apiFetch<GalleryResponse>(`/api/visiontrace/gallery?${params}`);
      setDetections(data.detections); setTotal(data.total); setPage(data.page); setPages(data.pages);
    } catch (err) { setError(failureMessage(err)); } finally { setLoading(false); }
  };

  const loadVideos = async () => {
    try {
      const data = await apiFetch<{ videos: GalleryVideo[] }>('/api/visiontrace/gallery/videos');
      setVideos(data.videos);
    } catch { /* non-fatal */ }
  };

  useEffect(() => { void loadVideos(); void loadGallery(1, ''); }, []);

  const onVideoFilter = (videoId: string) => { setFilterVideoId(videoId); void loadGallery(1, videoId); };
  const onPage = (p: number) => { void loadGallery(p, filterVideoId); };

  const findSimilar = async (detection: GalleryDetection) => {
    setSearchTarget(detection); setSearchResults(null); setSearchError(''); setSearching(true); setRankingNote('');
    try {
      const data = await apiFetch<{ results: SearchByDetectionResult[]; rankingNote: string }>(
        '/api/visiontrace/search/detection',
        { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ detectionId: detection.id, topK: 20 }) },
      );
      setSearchResults(data.results); setRankingNote(data.rankingNote ?? '');
    } catch (err) { setSearchError(failureMessage(err)); } finally { setSearching(false); }
  };

  const clearSearch = () => { setSearchTarget(null); setSearchResults(null); setSearchError(''); setRankingNote(''); };

  return <div className="page">
    <div className="page-heading">
      <div><div className="eyebrow">Indexed footage · visual discovery</div><h1>Detected People</h1><p className="subtitle">Browse every person detected and indexed across your video library. Click any thumbnail to search for visual appearances.</p></div>
      <span className="runtime-pill"><Users size={14} />{total} detections</span>
    </div>

    {/* Filters */}
    <div className="panel panel-body" style={{ marginBottom: 16, display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
      <div className="field" style={{ flex: '1 1 240px', margin: 0 }}>
        <label htmlFor="gallery-video-filter" style={{ display: 'block', marginBottom: 4 }}>Filter by video</label>
        <select id="gallery-video-filter" className="control" value={filterVideoId} onChange={e => onVideoFilter(e.target.value)} data-testid="select-gallery-video">
          <option value="">All indexed videos ({videos.reduce((s, v) => s + (v.personCount ?? 0), 0)} people)</option>
          {videos.map(v => <option key={v.id} value={v.id}>{v.name} — {v.personCount ?? 0} people</option>)}
        </select>
      </div>
      <button className="btn" onClick={() => loadGallery(page, filterVideoId)} data-testid="button-refresh-gallery"><Activity size={13} />Refresh</button>
      <div className="mono" style={{ fontSize: 11, color: '#7b8682' }}>Page {page} / {pages} · {total} total</div>
    </div>

    {error && <div className="toast-error" role="alert">{error}</div>}

    <div className="workspace-grid">
      {/* Gallery grid */}
      <section className="panel" style={{ flex: '1 1 500px' }}>
        <div className="panel-head"><h2>Detections</h2><span className="mono" style={{ fontSize: 10, color: '#7b8682' }}>YOLO11n · OSNet 512-d</span></div>
        {loading ? <SkeletonBlock /> : !detections.length
          ? <div className="empty-state"><div><div className="empty-symbol"><Users size={21} /></div><h3>No detections found</h3><p>{total === 0 ? 'No indexed videos contain detected people yet. Register and index videos in the Video library, then revisit.' : 'Try removing filters to see all detections.'}</p></div></div>
          : <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: 10, padding: 14 }}>
            {detections.map(d => <GalleryCard key={d.id} detection={d} onSearch={findSimilar} isSearchTarget={searchTarget?.id === d.id} />)}
          </div>
        }
        {pages > 1 && <div style={{ display: 'flex', gap: 8, justifyContent: 'center', padding: '12px 14px', borderTop: '1px solid hsl(var(--border))' }}>
          <button className="btn" disabled={page <= 1} onClick={() => onPage(page - 1)}>← Prev</button>
          <span className="mono" style={{ lineHeight: '32px', fontSize: 12 }}>Page {page} of {pages}</span>
          <button className="btn" disabled={page >= pages} onClick={() => onPage(page + 1)}>Next →</button>
        </div>}
      </section>

      {/* Search results panel */}
      <section className="panel" style={{ flex: '0 1 380px', minWidth: 280 }}>
        <div className="panel-head">
          <div><strong>Visual Search Results</strong>{searchTarget && <div style={{ fontSize: 10, color: '#7d8985', marginTop: 4 }}>Query: {searchTarget.videoName} · frame {searchTarget.frameNumber}</div>}</div>
          {searchTarget && <button className="btn" onClick={clearSearch}>Clear</button>}
        </div>
        {rankingNote && <div className="source-note" style={{ borderTop: 0, borderBottom: '1px solid hsl(var(--border))', fontSize: 10 }}>{rankingNote}</div>}
        {searching && <div className="empty-state"><div><div className="empty-symbol"><LoaderCircle size={20} className="spin" /></div><h3>Searching FAISS index</h3><p>Loading stored 512-d OSNet embedding and computing cosine similarities…</p></div></div>}
        {searchError && <div className="toast-error" style={{ margin: '14px' }}>{searchError}</div>}
        {!searching && !searchTarget && <div className="empty-state"><div><div className="empty-symbol"><Fingerprint size={21} /></div><h3>Click a detection to search</h3><p>Select any thumbnail from the gallery. The stored re-ID embedding is retrieved and searched against the FAISS index instantly.</p></div></div>}
        {!searching && searchResults && (searchResults.length === 0
          ? <div className="empty-state"><div><div className="empty-symbol"><Search size={20} /></div><h3>No matches found</h3><p>The FAISS index returned no results above threshold. The person may not appear in other indexed videos.</p></div></div>
          : <div style={{ overflowY: 'auto', maxHeight: 680 }}>
              {searchResults.map(hit => <SearchResultRow key={hit.id} hit={hit} />)}
            </div>
        )}
      </section>
    </div>
  </div>;
}

function GalleryCard({ detection, onSearch, isSearchTarget }: { detection: GalleryDetection; onSearch: (d: GalleryDetection) => void; isSearchTarget: boolean }) {
  const hasThumb = !!(detection.thumbnailPath);
  const score = (detection.detectorScore * 100).toFixed(0);
  return <button
    className={`detect-option${isSearchTarget ? ' selected' : ''}`}
    style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'stretch', padding: 0, overflow: 'hidden', borderRadius: 8, position: 'relative' }}
    onClick={() => onSearch(detection)}
    title={`${detection.videoName} · Frame ${detection.frameNumber} · ${duration(detection.timestampSeconds)} · ${score}% conf`}
    data-testid={`gallery-card-${detection.id}`}
  >
    <div style={{ width: '100%', paddingBottom: '133%', position: 'relative', background: '#1a2e2c' }}>
      {hasThumb
        ? <img src={detection.thumbnailPath!} alt={`Person detection from ${detection.videoName}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} />
        : <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', color: '#4a7269' }}><Users size={28} /></div>}
      <span style={{ position: 'absolute', top: 4, right: 4, background: 'rgba(0,0,0,0.7)', color: '#9fc7b9', fontSize: 9, padding: '2px 5px', borderRadius: 4, fontFamily: 'monospace' }}>{score}%</span>
      <span style={{ position: 'absolute', bottom: 4, left: 4, background: 'rgba(0,0,0,0.7)', color: '#cde8e1', fontSize: 9, padding: '2px 5px', borderRadius: 4, fontFamily: 'monospace' }}>{duration(detection.timestampSeconds)}</span>
      {isSearchTarget && <div style={{ position: 'absolute', inset: 0, border: '2px solid #3fcfa0', borderRadius: 8, pointerEvents: 'none' }} />}
    </div>
    <div style={{ padding: '5px 7px', textAlign: 'left' }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: '#cde8e1', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{detection.videoName}</div>
      <div style={{ fontSize: 9, color: '#7b8682', marginTop: 2, fontFamily: 'monospace' }}>f{detection.frameNumber}</div>
    </div>
  </button>;
}

function SearchResultRow({ hit }: { hit: SearchByDetectionResult }) {
  const [open, setOpen] = useState(false);
  const playerRef = useRef<HTMLVideoElement>(null);
  return <>
    <div className="hit-card" style={{ margin: '8px 12px', cursor: 'pointer', borderRadius: 10, overflow: 'hidden' }} onClick={() => setOpen(true)} role="button" tabIndex={0} onKeyDown={e => e.key === 'Enter' && setOpen(true)} data-testid={`search-result-${hit.id}`}>
      <div className="hit-image">
        {hit.thumbnailPath ? <img src={hit.thumbnailPath} alt={`Frame from ${hit.sourceName}`} /> : <div className="empty-symbol"><FileVideo2 size={18} /></div>}
        <span className="frame-label">FRAME {hit.frameNumber}</span>
        <span className="sim-chip">Score {hit.similarity.toFixed(3)}</span>
      </div>
      <div className="hit-meta">
        <div className="hit-title">{hit.sourceName}</div>
        <div className="hit-sub"><span>{hit.cameraId ?? '—'}</span><span>{duration(hit.timestampSeconds)}</span></div>
      </div>
    </div>
    {open && <div role="dialog" aria-modal="true" onClick={() => setOpen(false)} style={{ position: 'fixed', inset: 0, zIndex: 50, background: '#172b2dcc', display: 'grid', placeItems: 'center', padding: 18 }}>
      <div className="panel" onClick={e => e.stopPropagation()} style={{ width: 'min(700px,100%)', maxHeight: '90vh', overflow: 'auto', padding: 18 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div><div className="eyebrow">Source evidence</div><strong>{hit.sourceName}</strong></div>
          <button className="btn" onClick={() => setOpen(false)}>Close</button>
        </div>
        <video ref={playerRef} src={hit.videoPath} controls preload="metadata" style={{ width: '100%', maxHeight: '55vh', background: '#1c2828' }} onLoadedMetadata={e => { e.currentTarget.currentTime = hit.timestampSeconds; }} />
        <div className="source-note" style={{ paddingLeft: 0 }}>Camera {hit.cameraId ?? 'not supplied'} · Frame {hit.frameNumber} · {duration(hit.timestampSeconds)} · cosine similarity {hit.similarity.toFixed(4)}</div>
        <div className="notice" style={{ marginTop: 10 }}>Similarity is a navigation aid only. Confirm appearance against the full source recording before any decision.</div>
      </div>
    </div>}
  </>;
}

// ─── Pipeline Status Page ─────────────────────────────────────────────────────
type PipelineStage = { name: string; description: string; done: boolean };
type PipelineJob = { id: string; status: string; stage: string; progress: number; error: string | null; updatedAt: string; pipelineStages: PipelineStage[] };
type PipelineVideo = { id: string; name: string; status: string; frameCount: number; personCount: number; durationSeconds: number | null; cameraId: string | null; createdAt: string; updatedAt: string; job: PipelineJob | null };
type PipelineStatus = { runtime: { status: string; detector: string; reid: string; device: string }; indexedVideoCount: number; videos: PipelineVideo[] };

const STAGE_MATH: Record<string, string[]> = {
  'Frame Sampling': [
    'fps_sample = VISIONTRACE_SAMPLE_FPS (default 1.0)',
    'frame_idx = round(t × fps_video)',
    'Skipped frames = video_fps / sample_fps',
  ],
  'YOLO11n Detection': [
    'Input: resize(frame, 640, 640) → normalize [0,1]',
    'Grid: 3 scales × 80×80, 40×40, 20×20 anchors',
    'Filter: conf ≥ 0.25, class=person (COCO id 0)',
    'Output: [x1, y1, x2, y2, conf] per detection',
  ],
  'IoU Tracking': [
    'IoU(A,B) = |A∩B| / |A∪B|',
    'Match threshold: 0.3 (Hungarian assignment)',
    'max_missed = 8 frames before tracklet close',
    'Each person gets stable tracklet_id (t1, t2…)',
  ],
  'OSNet Re-ID': [
    'Crop: bbox region → resize 256×128 (H×W)',
    'Normalize: μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]',
    'Network: OSNet x1.0 → 512-dim feature vector',
    'L2-norm: v̂ = v / ‖v‖₂  (unit sphere projection)',
  ],
  'FAISS Indexing': [
    'Index type: IndexIDMap2(IndexFlatIP(512))',
    'Cosine sim = inner_product(v̂_query, v̂_stored)',
    '∵ both are L2-normalized: dot = cos(θ)',
    'Score range: [0, 1] · 1.0 = identical embedding',
  ],
};

function PipelineStatusPage() {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const load = async () => {
    try {
      const data = await apiFetch<PipelineStatus>('/api/visiontrace/pipeline/status');
      setStatus(data);
    } catch (err) { setError(failureMessage(err)); }
  };

  useEffect(() => { void load(); const t = window.setInterval(() => void load(), 4000); return () => clearInterval(t); }, []);

  const toggle = (id: string) => setExpanded(prev => { const s = new Set(prev); s.has(id) ? s.delete(id) : s.add(id); return s; });

  const activeCount = status?.videos.filter(v => ['queued','processing'].includes(v.status)).length ?? 0;
  const indexedCount = status?.indexedVideoCount ?? 0;

  return <div className="page">
    <div className="page-heading">
      <div><div className="eyebrow">Real-time processing · math breakdown</div><h1>Processing Pipeline</h1><p className="subtitle">Watch videos move through each stage of the detection, tracking, embedding, and indexing pipeline.</p></div>
      <div style={{ display: 'flex', gap: 10 }}>
        <span className="runtime-pill"><Cpu size={14} />{status?.runtime.status ?? '…'}</span>
        <button className="btn" onClick={() => void load()}><Activity size={13} />Refresh</button>
      </div>
    </div>

    {error && <div className="toast-error" role="alert">{error}</div>}

    {/* Stats strip */}
    <div className="stats-strip">
      {[
        ['Indexed videos', indexedCount],
        ['Active jobs', activeCount],
        ['Detector', status?.runtime.detector ?? '—'],
        ['Device', status?.runtime.device ?? '—'],
      ].map(([label, value]) => <div className="stat-box" key={String(label)}><div className="stat-label">{label}</div><div className="stat-value mono">{value ?? '—'}</div><div className="stat-foot">Live status</div></div>)}
    </div>

    {/* Pipeline legend */}
    <div className="panel panel-body" style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', gap: 0, alignItems: 'center', flexWrap: 'wrap', rowGap: 10 }}>
        {Object.keys(STAGE_MATH).map((name, i) => <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 20 }}>
          <div style={{ width: 28, height: 28, borderRadius: '50%', background: '#1d3533', display: 'grid', placeItems: 'center', fontFamily: 'monospace', fontSize: 12, fontWeight: 700, color: '#3fcfa0' }}>{i + 1}</div>
          <div style={{ fontSize: 12, color: '#cde8e1' }}>{name}</div>
        </div>)}
      </div>
      <div className="notice" style={{ marginTop: 12, marginBottom: 0 }}>Auto-refreshing every 4 seconds. Click any video card to expand pipeline stage details with the underlying math.</div>
    </div>

    {!status ? <SkeletonBlock /> : !status.videos.length
      ? <div className="panel empty-state"><div><div className="empty-symbol"><Layers size={20} /></div><h3>No videos registered</h3><p>Register and index videos in the Video library to see live pipeline status here.</p></div></div>
      : <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {status.videos.map(video => <PipelineVideoCard key={video.id} video={video} expanded={expanded.has(video.id)} onToggle={() => toggle(video.id)} />)}
        </div>
    }
  </div>;
}

function PipelineVideoCard({ video, expanded, onToggle }: { video: PipelineVideo; expanded: boolean; onToggle: () => void }) {
  const job = video.job;
  const isActive = job && ['queued', 'running'].includes(job.status);
  const isDone = video.status === 'indexed';
  const isFailed = video.status === 'failed' || job?.status === 'failed';
  return <div className="panel" style={{ border: isActive ? '1px solid #3fcfa030' : undefined }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 14px', cursor: 'pointer', userSelect: 'none' }} onClick={onToggle} role="button" tabIndex={0} onKeyDown={e => e.key === 'Enter' && onToggle()} data-testid={`pipeline-video-${video.id}`}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <FileVideo2 size={15} color="#4a7269" />
          <strong style={{ fontSize: 14, color: '#cde8e1', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{video.name}</strong>
          <StatusBadge status={video.status} />
        </div>
        <div style={{ display: 'flex', gap: 14, marginTop: 6 }}>
          {job && <div style={{ fontSize: 11, color: '#7d8985' }}>
            <span className="mono">{job.stage || job.status}</span>
          </div>}
          <div style={{ fontSize: 11, color: '#7d8985' }}>
            {video.frameCount} frames · {video.personCount} people {video.durationSeconds ? `· ${duration(video.durationSeconds)}` : ''}
          </div>
        </div>
      </div>
      {job && <div style={{ width: 140, flexShrink: 0 }}>
        <div className="progress-track">
          <div className="progress-fill" style={{ transform: `scaleX(${Math.max(0, Math.min(100, job.progress)) / 100})` }} />
        </div>
        <div style={{ textAlign: 'right', fontSize: 10, color: '#7b8682', marginTop: 3, fontFamily: 'monospace' }}>{job.progress}%</div>
      </div>}
      <ChevronRight size={16} color="#4a7269" style={{ transform: expanded ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }} />
    </div>

    {job?.error && <div className="error-note" style={{ margin: '0 14px 10px' }}>{job.error}</div>}

    {expanded && <div style={{ borderTop: '1px solid hsl(var(--border))', padding: 14 }}>
      <div className="section-kicker" style={{ marginBottom: 12 }}>Pipeline stages · mathematical operations</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {(job?.pipelineStages ?? Object.keys(STAGE_MATH).map((name, i) => ({ name, description: '', done: isDone }))).map((stage, i) => {
          const math = STAGE_MATH[stage.name] ?? [];
          const stageActive = !stage.done && isActive && (job?.pipelineStages ?? []).slice(0, i).every(s => s.done);
          return <div key={stage.name} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
            <div style={{ marginTop: 2, flexShrink: 0 }}>
              {stage.done
                ? <CheckCircle2 size={18} color="#3fcfa0" />
                : stageActive
                  ? <LoaderCircle size={18} color="#f59e0b" className="spin" />
                  : <Circle size={18} color="#3a5451" />}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontWeight: 600, fontSize: 13, color: stage.done ? '#3fcfa0' : stageActive ? '#f59e0b' : '#4a7269' }}>{stage.name}</span>
                {stage.description && <span className="mono" style={{ fontSize: 10, color: '#566b67' }}>{stage.description}</span>}
              </div>
              {math.length > 0 && <div style={{ marginTop: 5, padding: '7px 10px', background: '#0f201f', borderRadius: 6, borderLeft: '2px solid #1d3533' }}>
                {math.map((line, li) => <div key={li} style={{ fontFamily: 'monospace', fontSize: 11, color: '#7fb8ae', lineHeight: 1.7 }}>{line}</div>)}
              </div>}
            </div>
          </div>;
        })}
      </div>

      {isDone && <div className="notice" style={{ marginTop: 14 }}>✓ Video fully indexed. All embeddings committed to FAISS IndexFlatIP. Detections browseable in the Detected People gallery.</div>}
    </div>}
  </div>;
}

export default App;
