// Welcome page: extension download and setup steps, served by the local backend or hosted on GitHub Pages.
import { useEffect, useState } from 'react'

export const REPO_URL = 'https://github.com/samvit-k/macaronigraph'
export const RELEASE_ZIP = `${REPO_URL}/releases/latest/download/macaronigraph-extension.zip`
const LOGO = `${import.meta.env.BASE_URL}logo.svg`

const RUN_LOCAL = <>You are viewing this page from it, so it is running. On a new machine, run <code>./scripts/setup.sh</code> once,
  then <code>.venv/bin/uvicorn backend.app:app --port 8765</code> (it listens on 127.0.0.1 only).</>
const RUN_HOSTED = <>Clone the repository on a Mac with Apple Silicon, run <code>./scripts/setup.sh</code> once, then
  <code> .venv/bin/uvicorn backend.app:app --port 8765</code> (it listens on 127.0.0.1 only).</>

export const steps = (hosted) => [
  { title: 'Run the local backend', body: hosted ? RUN_HOSTED : RUN_LOCAL },
  { title: 'Load the extension in Chrome', body: <>Unzip the download, open <code>chrome://extensions</code>, turn on <b>Developer mode</b>, click
    <b> Load unpacked</b> and choose the <code>macaronigraph-extension</code> folder.</> },
  { title: 'Capture your saved Reels', body: <>Open the extension, choose your saved Reels as the source and start capture. Processed Reels appear
    in the workspace graph and can be asked about.</> },
]

export default function Welcome({ hosted = false }) {
  const [health, setHealth] = useState(null)
  useEffect(() => {
    if (hosted) return
    fetch('/api/health').then((r) => (r.ok ? r.json() : null)).then(setHealth).catch(() => setHealth(false))
  }, [hosted])
  const second = hosted ? { href: REPO_URL, label: 'View on GitHub' } : { href: '/workspace/', label: 'Open Workspace' }
  return (
    <div className="wl">
      <div className="wl-grid" aria-hidden="true" />
      <div className="wl-glow" aria-hidden="true" />
      <header className="wl-nav">
        <span className="wl-brand"><img src={LOGO} alt="" aria-hidden="true" />MacaroniGraph</span>
        <a className="wl-navlink" href={second.href}>{second.label}</a>
      </header>
      <main className="wl-hero">
        <h1>A <span className="accent">knowledge graph</span> of your saved Reels.</h1>
        <p className="wl-sub">
          MacaroniGraph turns the Instagram Reels you save into searchable passages (e.g., transcripts, captions, and
          on-screen text), links them through the concepts they mention, and answers your questions with citations to
          the original Reels.
        </p>
        <div className="wl-cta">
          <a className="wl-btn primary" href={hosted ? RELEASE_ZIP : '/welcome/extension.zip'} download>Download Extension (.zip)</a>
          <a className="wl-btn ghost" href={second.href}>{second.label}</a>
        </div>
        <p className="wl-fine">
          Not on the Chrome Web Store. The extension works only with {hosted ? 'the' : 'this'} local backend, which needs macOS on Apple Silicon.
          {health === false && <span className="wl-warn"> The backend did not answer just now — check that it is still running.</span>}
        </p>
        <ol className="wl-steps">
          {steps(hosted).map((s, i) => (
            <li key={s.title}><span className="wl-num">{i + 1}</span><div><h2>{s.title}</h2><p>{s.body}</p></div></li>
          ))}
        </ol>
      </main>
      <footer className="wl-foot">© 2026 Samvit Kundapur</footer>
    </div>
  )
}