// Workspace shell: graph canvas, inspector, legend and assistant panel, and the wiring between them.
import { useCallback, useEffect, useMemo, useState } from 'react'
import AssistantPanel from './components/assistant/AssistantPanel.jsx'
import { citationMap } from './components/assistant/ChatMessage.jsx'
import GraphCanvas from './components/GraphCanvas.jsx'
import Inspector from './components/inspector/Inspector.jsx'
import Legend from './components/Legend.jsx'
import TopBar from './components/TopBar.jsx'
import { answerHighlight, conceptId, DEFAULT_FILTERS, hiddenIds, matchingIds, reelId } from './graph/toElements.js'
import { useConversation } from './state/useConversation.js'
import { useGraph } from './state/useGraph.js'
import { useStatus } from './state/useStatus.js'
import { ThemeContext, useTheme } from './theme.jsx'

const NARROW = '(max-width: 1100px)'

function useNarrow() {
  const get = () => typeof window !== 'undefined' && !!window.matchMedia?.(NARROW).matches
  const [narrow, setNarrow] = useState(get)
  useEffect(() => {
    const mq = window.matchMedia?.(NARROW)
    if (!mq) return undefined
    const on = () => setNarrow(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return narrow
}

function StageState({ graph }) {
  if (graph.status === 'loading') return <div className="stage-note"><span className="spinner" aria-hidden="true" /> Loading the knowledge graph…</div>
  if (graph.status === 'offline' || graph.status === 'error') {
    return (
      <div className="stage-card" role="alert">
        <h2>{graph.status === 'offline' ? 'Knowledge Graph Offline' : 'Could Not Load the Graph'}</h2>
        <p>{graph.status === 'offline'
          ? 'Neo4j is not running, so the graph cannot be shown. Asking still works: answers fall back to passage search.'
          : graph.error}</p>
        {graph.status === 'offline' && <p className="insp-small">Start it with <code>scripts/neo4j.sh start</code>, then retry.</p>}
        <button type="button" className="btn btn-sm" onClick={graph.reload}>Retry</button>
      </div>
    )
  }
  if (graph.status === 'ready' && Object.keys(graph.model.reels).length === 0) {
    return (
      <div className="stage-card">
        <h2>No Processed Reels Yet</h2>
        <p>Capture your saved Reels with the MacaroniGraph extension; they appear here once processed and added to the graph.</p>
        <a className="btn btn-sm btn-primary" href="/welcome">Set Up the Extension</a>
      </div>
    )
  }
  return null
}

export default function App() {
  const [theme, toggleTheme] = useTheme()
  const graph = useGraph()
  const chat = useConversation()
  const status = useStatus()
  const narrow = useNarrow()
  const [chatOpen, setChatOpen] = useState(() => !(typeof window !== 'undefined' && window.matchMedia?.(NARROW).matches))
  const [selection, setSelection] = useState(null)
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState(DEFAULT_FILTERS)
  const [draft, setDraft] = useState('')
  const [focusTick, setFocusTick] = useState(0)
  const [highlightTurn, setHighlightTurn] = useState(null)

  const model = graph.model
  const hidden = useMemo(() => hiddenIds(model, filters), [model, filters])
  const matches = useMemo(() => matchingIds(model, query), [model, query])
  const turn = chat.turns.find((t) => t.turn_id === highlightTurn) ?? null
  const highlight = useMemo(() => answerHighlight(turn), [turn])

  useEffect(() => { if (narrow) setChatOpen(false) }, [narrow])

  useEffect(() => { setHighlightTurn(null) }, [chat.id])

  const ensureLoaded = useCallback(async (kind, id) => {
    const present = kind === 'reel' ? model.reels[id] : model.concepts[id]
    if (!present) {
      try { await graph.expand(kind, id) } catch {  }
    }
  }, [graph, model])

  const onSelect = useCallback(async (sel) => {
    if (sel && (sel.kind === 'reel' || sel.kind === 'concept')) await ensureLoaded(sel.kind, sel.id)
    setSelection(sel)
  }, [ensureLoaded])

  const onToggleExpand = useCallback((sel) => {
    if (sel.kind !== 'reel' && sel.kind !== 'concept') return
    const nodeId = sel.kind === 'reel' ? reelId(sel.id) : conceptId(sel.id)
    if (model.expanded[nodeId]) graph.collapse(nodeId)
    else graph.expand(sel.kind, sel.id).catch(() => {})
  }, [graph, model])

  const onPick = useCallback(async (hit) => {
    setQuery('')
    try { await graph.expand(hit.kind, hit.id) } catch {  }
    setSelection({ kind: hit.kind, id: hit.id, focus: true })
  }, [graph])

  const highlightAnswer = useCallback(async (t) => {
    if (highlightTurn === t.turn_id) { setHighlightTurn(null); return }
    setSelection(null)
    setHighlightTurn(t.turn_id)

    const missing = new Set((t.cited_concepts ?? []).filter((c) => !model.concepts[c.key]).map((c) => c.key))
    const reels = [...new Set((t.citations ?? []).map((c) => c.reel_id))]
    for (const rid of reels) {
      if (!model.reels[rid] || (missing.size && !model.expanded[reelId(rid)])) {
        try { await graph.expand('reel', rid) } catch {  }
      }
    }
  }, [graph, highlightTurn, model])

  const onCite = useCallback(async (t, sid) => {
    const citation = citationMap(t)[sid]
    const source = (t.sources ?? []).find((s) => s.sid === sid)
    setHighlightTurn(t.turn_id)
    if (citation?.reel_id && !model.reels[citation.reel_id] && graph.status === 'ready') {
      try { await graph.expand('reel', citation.reel_id) } catch {  }
    }
    setSelection({ kind: 'source', sid, citation, source, turnId: t.turn_id })
  }, [graph, model])

  const onAsk = useCallback((text) => {
    setChatOpen(true)
    setDraft(text)
    setFocusTick((n) => n + 1)
  }, [])

  return (
    <ThemeContext.Provider value={theme}>
      <div className={`app${chatOpen ? ' as-open' : ''}${narrow ? ' narrow' : ''}`}>
        <main className="main">
          <TopBar model={model} graphStatus={graph.status} scope={graph.scope} onScope={(n) => { setSelection(null); graph.setScope(n) }}
            status={status} theme={theme} onTheme={toggleTheme} query={query} onQuery={setQuery} onPick={onPick}
            filters={filters} onFilters={setFilters} chatOpen={chatOpen} onToggleChat={() => setChatOpen((v) => !v)} />
          <div className="stage">
            {graph.status === 'ready' && Object.keys(model.reels).length > 0 && (
              <>
                <GraphCanvas model={model} hidden={hidden} matches={matches} selection={selection?.kind === 'source'
                  ? { kind: 'reel', id: selection.citation?.reel_id } : selection} highlight={highlight}
                  onSelect={onSelect} onToggleExpand={onToggleExpand} />
                <Legend />
                {Object.keys(model.expanded).length > 0 && (
                  <div className="stage-hint">
                    <button type="button" className="link-btn" onClick={() => { graph.resetToOverview(); setSelection(null) }}>Back to Overview</button>
                  </div>
                )}
              </>
            )}
            <StageState graph={graph} />
            <Inspector selection={selection} model={model} graph={graph} onSelect={onSelect} onClose={() => setSelection(null)} onAsk={onAsk}
              graphReady={graph.status === 'ready'} />
          </div>
        </main>
        {narrow && chatOpen && <div className="as-scrim" onClick={() => setChatOpen(false)} aria-hidden="true" />}
        <AssistantPanel open={chatOpen} onClose={() => setChatOpen(false)} chat={chat} draft={draft} setDraft={setDraft}
          focusTick={focusTick} onCite={onCite} onHighlight={highlightAnswer} activeTurn={highlightTurn}
          graphReady={graph.status === 'ready'} />
      </div>
    </ThemeContext.Provider>
  )
}