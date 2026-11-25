// Top bar (SanD Hacks layout): brand + what the canvas shows, live status, search, filters, chat toggle.
import FilterPopover from './FilterPopover.jsx'
import SearchBox from './SearchBox.jsx'

export function graphMeta(model, status) {
  if (status === 'offline') return 'Knowledge Graph Offline'
  if (status === 'loading') return 'Loading Graph…'
  const t = model?.totals
  if (!t) return ''
  const shownConcepts = Object.keys(model.concepts).length
  return `${shownConcepts}/${t.concepts} Concepts`
}

export default function TopBar({ model, graphStatus, status, query, onQuery, onPick, filters, onFilters, chatOpen, onToggleChat }) {
  const offline = graphStatus === 'offline' || graphStatus === 'error'
  return (
    <header className="topbar">
      <div className="tb-title">
        <img className="brand-mark" src="/workspace/logo.svg" alt="" aria-hidden="true" />
        <span className="tb-name">MacaroniGraph</span>
        <span className="tb-meta" data-testid="graph-meta">{graphMeta(model, graphStatus)}</span>
      </div>
      <div className="tb-actions">
        {status && (
          <span className={`pill status-${status.tone}`} title={status.detail} data-testid="status-chip">
            <span className="status-dot" aria-hidden="true" />{status.text}
          </span>
        )}
        <SearchBox query={query} onQuery={onQuery} onPick={onPick} disabled={offline} />
        <FilterPopover model={model} filters={filters} onChange={onFilters} disabled={offline} />
        <a className="btn btn-ghost" href="/welcome" title="Install the extension and set up capture">Setup</a>
        <button type="button" className={`btn ${chatOpen ? 'btn-on' : ''}`} onClick={onToggleChat} aria-pressed={chatOpen}
          aria-controls="assistant-panel" title="Ask your saved Reels">
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 3.5h11v7h-6l-3 2.5v-2.5h-2z" /></svg>
          Ask
        </button>
      </div>
    </header>
  )
}
