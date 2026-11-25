// Local welcome page: extension download and setup steps.
import { useEffect, useState } from 'react'

export const STEPS = [
  { title: 'Run the local backend', body: <>You are viewing this page from it, so it is running. On a new machine, run <code>./scripts/setup.sh</code> once,
    then <code>.venv/bin/uvicorn backend.app:app --port 8765</code> (it listens on 127.0.0.1 only).</> },
  { title: 'Load the extension in Chrome', body: <>Unzip the download, open <code>chrome://extensions</code>, turn on <b>Developer mode</b>, click
    <b> Load unpacked</b> and choose the <code>macaronigraph-extension</code> folder.</> },
  { title: 'Capture your saved Reels', body: <>Open the extension, choose your saved Reels as the source and start capture. Processed Reels appear
    in the workspace graph and can be asked about.</> },
]

export default function Welcome() {
  const [health, setHealth] = useState(null)
  useEffect(() => {
    fetch('/api/health').then((r) => (r.ok ? r.json() : null)).then(setHealth).catch(() => setHealth(false))
  }, [])
  return (
    <div className="wl">
      <div className="wl-grid" aria-hidden="true" />
      <div className="wl-glow" aria-hidden="true" />
      <header className="wl-nav">
        <span className="wl-brand"><img src="/workspace/logo.svg" alt="" aria-hidden="true" />MacaroniGraph</span>
        <a className="wl-navlink" href="/workspace/">Open Workspace</a>
      </header>
      <main className="wl-hero">
        <h1>A <span className="accent">knowledge graph</span> of your saved Reels.</h1>
        <p className="wl-sub">
          MacaroniGraph turns the Instagram Reels you save into searchable passages (e.g., transcripts, captions, and
          on-screen text), links them through the concepts they mention, and answers your questions with citations to
          the original Reels.
        </p>
        <div className="wl-cta">
          <a className="wl-btn primary" href="/welcome/extension.zip" download>Download Extension (.zip)</a>
          <a className="wl-btn ghost" href="/workspace/">Open Workspace</a>
        </div>
        <p className="wl-fine">
          Not on the Chrome Web Store. The extension only works together with this local backend, which needs macOS on Apple Silicon.
          {health === false && <span className="wl-warn"> The backend did not answer just now — check that it is still running.</span>}
        </p>
        <ol className="wl-steps">
          {STEPS.map((s, i) => (
            <li key={s.title}><span className="wl-num">{i + 1}</span><div><h2>{s.title}</h2><p>{s.body}</p></div></li>
          ))}
        </ol>
      </main>
      <footer className="wl-foot">© 2026 Samvit Kundapur</footer>
    </div>
  )
}