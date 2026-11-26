// One exchange: generated answers, exact-search listings and app notes are rendered in visibly distinct styles.
import { useState } from 'react'
import { contentType } from '../../graph/vocab.js'

const CITE = /\[(S\d+)\]/g

export function isListing(turn) {
  return typeof turn?.intent === 'string' && turn.intent.startsWith('listing_')
}

export function listingScope(turn) {
  const parts = (turn?.answer?.text ?? '').split('\n\n')
  return parts.length > 1 ? parts.slice(1).join('\n\n') : ''
}

export function citationMap(turn) {
  const out = {}
  for (const c of turn?.citations ?? []) out[c.sid] = c
  return out
}

function CiteChip({ sid, turn, onCite }) {
  const c = citationMap(turn)[sid]
  const label = c ? `${contentType(c.content_type).label}${c.owner ? ` · @${c.owner}` : ''}` : 'source not available'
  return (
    <button type="button" className="cite-chip" onClick={() => onCite(turn, sid)} title={`Open cited source ${sid}: ${label}`}
      aria-label={`Cited source ${sid}`}>{sid.slice(1)}</button>
  )
}

const clean = (t) => (t ?? '').replace(CITE, '').replace(/\s+([.,;:!?])/g, '$1').trim()

function Sentence({ s, turn, onCite }) {
  return (
    <span className="sent">{clean(s.text)}{' '}
      {(s.citations ?? []).map((sid) => <CiteChip key={sid} sid={sid} turn={turn} onCite={onCite} />)}{' '}</span>
  )
}

function Generated({ turn, onCite, onHighlight, active, graphReady }) {
  const a = turn.answer ?? {}
  return (
    <div className={`answer-box${active ? ' active' : ''}`}>
      <div className="answer-head">
        <span>Generated Answer · Local Model</span>
        {graphReady && (turn.citations?.length ?? 0) > 0 && (
          <button type="button" className="link-btn" onClick={() => onHighlight(turn)}>{active ? 'Highlighted in Graph' : 'Show in Graph'}</button>
        )}
      </div>
      <p className="answer-text">{(a.sentences ?? []).map((s, i) => <Sentence key={i} s={s} turn={turn} onCite={onCite} />)}</p>
      {a.missing && <div className="answer-note partial">{a.missing}</div>}
      {a.conflict && <div className="answer-note">The cited Reels may disagree; compare the sources.</div>}
    </div>
  )
}

export const PREVIEW_ROWS = 3

function Summary({ turn, onCite }) {
  const s = turn.answer?.summary
  const head = <div className="answer-head"><span>Generated Answer · Local Model</span></div>
  if (turn.status === 'queued') {
    return (
      <div className="status-card wait" role="status">
        <span className="spinner" aria-hidden="true" /> Waiting for the local model to summarize these Reels — capture or processing
        is using it. The exact matches are below.
      </div>
    )
  }
  if (turn.status === 'expired') return <div className="status-card neutral">No summary: the backend restarted. The exact matches are below.</div>
  if (turn.status === 'unavailable') return <div className="status-card neutral"><b>Summary Unavailable.</b> The local model's queue was full. The exact matches are below.</div>
  if (s?.status === 'error') return <div className="status-card bad" role="alert"><b>Summary Failed.</b> {s.error ?? ''}</div>
  if (s?.status !== 'ok' || !(s.sentences?.length)) {
    return <div className="status-card neutral">The local model could not summarize these passages; the exact matches are below.</div>
  }
  return (
    <div className="answer-box">
      {head}
      <p className="answer-text">{s.sentences.map((x, i) => <Sentence key={i} s={x} turn={turn} onCite={onCite} />)}</p>
      {s.missing && <div className="answer-note partial">{s.missing}</div>}
      {s.conflict && <div className="answer-note">The cited Reels may disagree; compare the sources.</div>}
      {s.based_on < s.of && <div className="insp-small">Summarizes the first {s.based_on} of the {s.of} matching Reels below (spoken mentions first).</div>}
    </div>
  )
}

function Listing({ turn, onCite, onHighlight, active, graphReady }) {
  const a = turn.answer ?? {}
  const [head, ...items] = (a.sentences ?? [])
  const hasHead = head && (!head.citations || head.citations.length === 0)
  const rows = hasHead ? items : a.sentences ?? []
  const summarized = a.summary != null || ['queued', 'expired', 'unavailable'].includes(turn.status)
  const [open, setOpen] = useState(false)
  const collapsible = summarized && rows.length > PREVIEW_ROWS
  const shown = collapsible && !open ? rows.slice(0, PREVIEW_ROWS) : rows
  const evidence = (
    <div className={`answer-box exact${active ? ' active' : ''}`}>
      <div className="answer-head">
        <span>{summarized ? 'Evidence · ' : ''}Exact Search · No Model</span>
        {graphReady && rows.length > 0 && <button type="button" className="link-btn" onClick={() => onHighlight(turn)}>{active ? 'Highlighted in Graph' : 'Show in Graph'}</button>}
      </div>
      {hasHead && <p className="answer-text strong">{head.text}</p>}
      {shown.length > 0 && (
        <ul className="listing">
          {shown.map((s, i) => {
            const c = citationMap(turn)[s.citations?.[0]]
            return (
              <li key={i}>
                <span className={`ev-badge${(c?.content_type ?? '').startsWith('ocr') ? ' ocr' : ''}`}>{contentType(c?.content_type).label}</span>
                <span className="listing-text">{clean(s.text)}</span>
                {(s.citations ?? []).map((sid) => <CiteChip key={sid} sid={sid} turn={turn} onCite={onCite} />)}
              </li>
            )
          })}
        </ul>
      )}
      {collapsible && (
        <button type="button" className="link-btn listing-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? 'Minimize Evidence' : `Show All ${rows.length} Matches`}
        </button>
      )}
      {(!collapsible || open) && listingScope(turn) && <div className="answer-note">{listingScope(turn)}</div>}
    </div>
  )
  if (!summarized) return evidence
  return <div className="listing-stack"><Summary turn={turn} onCite={onCite} />{evidence}</div>
}

function Clarify({ turn, onClarify }) {
  const names = turn.rewrite?.clarify ?? []
  return (
    <div className="clarify-card">
      <p>{turn.answer?.text}</p>
      {names.length > 0 && (
        <div className="chip-wrap">
          {names.map((n) => (
            <button key={n} type="button" className="chip-btn inline" onClick={() => onClarify(turn.question, n)}
              title="Puts your question back in the box with this choice; edit it if needed, then send">
              {n}
            </button>
          ))}
        </div>
      )}
      <div className="insp-small">Answered without the model: nothing was guessed.</div>
    </div>
  )
}

export function StatusCard({ turn }) {
  const a = turn.answer ?? {}
  if (turn.status === 'queued') {
    return (
      <div className="status-card wait" role="status">
        <span className="spinner" aria-hidden="true" /> Waiting for the local model — capture or processing is using it. The answer
        appears here automatically.
      </div>
    )
  }
  if (turn.status === 'insufficient_evidence') {
    return (
      <div className="status-card neutral">
        <b>Not Enough Evidence.</b> {a.text ?? 'Your processed Reels do not contain enough information to answer this.'}
        <div className="insp-small">Only processed Reels are searched; try naming the Reel or concept.</div>
      </div>
    )
  }
  if (turn.status === 'expired') return <div className="status-card neutral">This request was not answered because the backend restarted. Ask again.</div>
  if (turn.status === 'unavailable' || turn.status === 'retrieval_only') {
    return <div className="status-card neutral"><b>Answer Unavailable.</b> {a.reason ?? a.error ?? 'The local model could not be used.'}</div>
  }
  if (turn.status === 'error') return <div className="status-card bad" role="alert"><b>Generation Failed.</b> {a.error ?? a.reason ?? ''}</div>
  return null
}

function MinimizeButton({ collapsed, onToggle }) {
  return (
    <button type="button" className="chat-min-btn" onClick={onToggle} aria-expanded={!collapsed}
      title={collapsed ? 'Show this answer again' : 'Minimize this question and its answer'}>
      <svg viewBox="0 0 16 16" aria-hidden="true"><path d={collapsed ? 'M4.5 6.5 8 10l3.5-3.5' : 'M4.5 9.5 8 6l3.5 3.5'} /></svg>
      {collapsed ? 'Expand' : 'Minimize'}
    </button>
  )
}

export default function ChatMessage({ turn, onCite, onHighlight, onClarify, active, graphReady = true, collapsed = false, onToggle = null }) {
  const answered = turn.status === 'ok' && (turn.answer?.sentences?.length ?? 0) > 0
  const listing = isListing(turn) && (turn.answer?.sentences?.length ?? 0) > 0
  return (
    <div className="exchange" data-testid="exchange">
      <div className={`chat-row user${collapsed ? ' collapsed' : ''}`}>
        <div className="chat-bubble"><div className="chat-prompt" title={collapsed ? turn.question : undefined}>{turn.question}</div></div>
        {onToggle && <MinimizeButton collapsed={collapsed} onToggle={onToggle} />}
        {collapsed && <div className="chat-minimized">Answer minimized</div>}
      </div>
      {!collapsed && <div className="chat-row assistant">
        {(turn.fallback ?? []).length > 0 && (
          <div className="answer-note">Knowledge graph unavailable for this answer — it used passage search only.</div>
        )}
        {turn.status === 'clarify' ? <Clarify turn={turn} onClarify={onClarify} />
          : listing ? <Listing turn={turn} onCite={onCite} onHighlight={onHighlight} active={active} graphReady={graphReady} />
          : answered ? <Generated turn={turn} onCite={onCite} onHighlight={onHighlight} active={active} graphReady={graphReady} />
            : turn.status === 'ok'
              ? <div className="status-card neutral">{turn.answer?.text || 'No answer text was produced.'}</div>
              : <StatusCard turn={turn} />}
      </div>}
    </div>
  )
}