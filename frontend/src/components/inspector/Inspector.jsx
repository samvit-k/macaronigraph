// Floating evidence inspector (SanD Hacks layout: card on the right of the canvas, bottom sheet on narrow screens).
// Original source text is always shown as a quotation with its evidence type (transcript / caption / on-screen text)
// and a link to the Reel; anything the model produced is labelled as such.
import { useEffect, useState } from 'react'
import { conceptId, nodeLabel, reelId } from '../../graph/toElements.js'
import { conceptType, contentType, EDGE_KINDS, isOcr, predicateLabel, timestamp } from '../../graph/vocab.js'

export const reelUrl = (rid, url) => url || `https://www.instagram.com/reel/${rid}/`

export function Excerpt({ ev, compact = false }) {
  const ct = contentType(ev.content_type)
  const ts = ev.content_type === 'transcript' ? timestamp(ev.start) : null
  return (
    <figure className={`excerpt${isOcr(ev.content_type) ? ' ocr' : ''}`}>
      <blockquote className={compact ? 'clamp' : ''}>{ev.text}</blockquote>
      <figcaption>
        <span className="ev-type">{ct.label}</span>{ts ? ` · ${ts}` : ''}
        {isOcr(ev.content_type) && <span className="ev-note"> · OCR may be misread</span>}
      </figcaption>
    </figure>
  )
}

function useDetail(fetchDetail, kind, id) {
  const [st, setSt] = useState({ data: null, error: null, loading: true })
  useEffect(() => {
    let live = true
    setSt({ data: null, error: null, loading: true })
    fetchDetail(kind, id).then((data) => live && setSt({ data, error: null, loading: false }))
      .catch((e) => live && setSt({ data: null, error: e.message, loading: false }))
    return () => { live = false }
  }, [fetchDetail, kind, id])
  return st
}

function ExpandButton({ nodeId, expanded, onExpand, onCollapse }) {
  return expanded
    ? <button type="button" className="btn btn-sm" onClick={() => onCollapse(nodeId)}>Collapse</button>
    : <button type="button" className="btn btn-sm" onClick={onExpand}>Show Connected Nodes</button>
}

function ReelView({ id, model, graph, onSelect, onAsk }) {
  const { data, error, loading } = useDetail(graph.fetchDetail, 'reel', id)
  const reel = data?.reel ?? model.reels[id] ?? { reel_id: id }
  const passages = []
  const seen = new Set()
  for (const c of data?.concepts ?? []) for (const ev of c.evidence ?? []) {
    if (!seen.has(ev.chunk_id) && ev.text) { seen.add(ev.chunk_id); passages.push(ev) }
  }
  passages.sort((a, b) => (a.start ?? 0) - (b.start ?? 0))
  const nodeId = reelId(id)
  return (
    <>
      <div className="insp-kicker">Reel · {nodeLabel(model, reelId(id))}</div>
      <h2 className="insp-title">{reel.owner ? `@${reel.owner}` : id}</h2>
      {reel.caption && <p className="insp-caption">{reel.caption}</p>}
      <div className="insp-actions">
        <a className="btn btn-sm btn-primary" href={reelUrl(id, reel.source_url)} target="_blank" rel="noreferrer">Open on Instagram ↗</a>
        <ExpandButton nodeId={nodeId} expanded={!!model.expanded[nodeId]} onExpand={() => graph.expand('reel', id)} onCollapse={graph.collapse} />
        <button type="button" className="btn btn-sm" onClick={() => onAsk(`What does the Reel by @${reel.owner ?? id} say?`)}>Ask About It</button>
      </div>
      {loading && <p className="insp-empty">Loading evidence…</p>}
      {error && <div className="warn-card" role="alert">Could not load this Reel: {error}</div>}
      {data && (
        <>
          <section className="insp-section">
            <h3>Concepts <span className="insp-note">{data.concepts.length} mentioned</span></h3>
            <div className="chip-wrap">
              {data.concepts.map((c) => (
                <button key={c.key} type="button" className="concept-chip" onClick={() => onSelect({ kind: 'concept', id: c.key, focus: true }, { ensure: c })}
                  title={`${conceptType(c.type).label} · mentioned by ${c.n_reels} Reel${c.n_reels === 1 ? '' : 's'}`}>
                  <span className="node-dot sm" style={{ background: conceptType(c.type).color }} />{c.name}
                  {c.n_reels > 1 && <span className="chip-count">{c.n_reels}</span>}
                </button>
              ))}
            </div>
          </section>
          <section className="insp-section">
            <h3>Source Passages <span className="insp-note">original text from this Reel</span></h3>
            {passages.length === 0 && <p className="insp-empty">No passage excerpts.</p>}
            {passages.slice(0, 8).map((ev) => <Excerpt key={ev.chunk_id} ev={ev} compact />)}
          </section>
          {data.assertions.length > 0 && (
            <section className="insp-section">
              <h3>Extracted Relationships</h3>
              {data.assertions.map((a) => <AssertionRow key={a.id} a={a} onSelect={onSelect} />)}
            </section>
          )}
        </>
      )}
    </>
  )
}

function AssertionRow({ a, onSelect }) {
  return (
    <button type="button" className="insp-row" onClick={() => onSelect({ kind: 'assertion', id: a.id, focus: true })}>
      <StatusPill status={a.status} />
      <span>{a.subject} <i>{predicateLabel(a.predicate)}</i> {a.object}</span>
    </button>
  )
}

export function StatusPill({ status }) {
  const k = EDGE_KINDS[status] ?? EDGE_KINDS.proposed
  return <span className="status-pill" style={{ '--rel': k.color }}>{status === 'verified' ? '✓ ' : ''}{k.label.replace(' Relationship', '')}</span>
}

function ConceptView({ id, model, graph, onSelect, onAsk }) {
  const { data, error, loading } = useDetail(graph.fetchDetail, 'concept', id)
  const c = data?.concept ?? model.concepts[id] ?? { key: id, name: id }
  const t = conceptType(c.type)
  const nodeId = conceptId(id)
  return (
    <>
      <div className="insp-kicker"><span className="node-dot sm" style={{ background: t.color }} /> {t.label} · {nodeLabel(model, conceptId(id))}</div>
      <h2 className="insp-title">{c.name}</h2>
      {(c.aliases ?? []).filter((x) => x !== c.name).length > 0 && (
        <p className="insp-small">Also written as: {(c.aliases ?? []).filter((x) => x !== c.name).join(', ')}</p>
      )}
      <p className="insp-small">Mentioned in {c.n_reels ?? data?.reels?.length ?? '…'} Reel{c.n_reels === 1 ? '' : 's'}.
        {c.n_reels > 1 ? ' Reels that mention the same concept are not necessarily related to each other.' : ''}</p>
      <div className="insp-actions">
        <ExpandButton nodeId={nodeId} expanded={!!model.expanded[nodeId]} onExpand={() => graph.expand('concept', id)} onCollapse={graph.collapse} />
        <button type="button" className="btn btn-sm" onClick={() => onAsk(`What do my Reels say about ${c.name}?`)}>Ask About It</button>
      </div>
      {loading && <p className="insp-empty">Loading evidence…</p>}
      {error && <div className="warn-card" role="alert">Could not load this concept: {error}</div>}
      {data && (
        <>
          <section className="insp-section">
            <h3>Mentioned In</h3>
            {data.reels.map((r) => (
              <div className="reel-ev" key={r.reel_id}>
                <div className="reel-ev-head">
                  <button type="button" className="link-btn strong" onClick={() => onSelect({ kind: 'reel', id: r.reel_id, focus: true }, { ensure: r })}>
                    {r.owner ? `@${r.owner}` : r.reel_id}</button>
                  <a className="ext-link" href={reelUrl(r.reel_id, r.source_url)} target="_blank" rel="noreferrer">Instagram ↗</a>
                </div>
                {(r.evidence ?? []).slice(0, 2).map((ev) => <Excerpt key={ev.chunk_id} ev={ev} compact />)}
              </div>
            ))}
          </section>
          {data.assertions.length > 0 && (
            <section className="insp-section">
              <h3>Extracted Relationships <span className="insp-note">model-proposed unless marked otherwise</span></h3>
              {data.assertions.map((a) => <AssertionRow key={a.id} a={a} onSelect={onSelect} />)}
            </section>
          )}
        </>
      )}
    </>
  )
}

function MentionView({ sel, model, graph, onSelect }) {
  const { data, error, loading } = useDetail(graph.fetchDetail, 'reel', sel.reel_id)
  const reel = data?.reel ?? model.reels[sel.reel_id] ?? {}
  const c = (data?.concepts ?? []).find((x) => x.key === sel.concept_key) ?? model.concepts[sel.concept_key] ?? { name: sel.concept_key }
  return (
    <>
      <div className="insp-kicker">Mention</div>
      <h2 className="insp-title">{reel.owner ? `@${reel.owner}` : sel.reel_id} mentions {c.name}</h2>
      <div className="info-card">This Reel’s text names the concept. A mention is not a relationship: it does not say how this
        Reel relates to other Reels that mention the same concept.</div>
      <div className="insp-actions">
        <button type="button" className="btn btn-sm" onClick={() => onSelect({ kind: 'reel', id: sel.reel_id, focus: true })}>Inspect Reel</button>
        <button type="button" className="btn btn-sm" onClick={() => onSelect({ kind: 'concept', id: sel.concept_key, focus: true })}>Inspect Concept</button>
        <a className="btn btn-sm" href={reelUrl(sel.reel_id, reel.source_url)} target="_blank" rel="noreferrer">Instagram ↗</a>
      </div>
      {loading && <p className="insp-empty">Loading evidence…</p>}
      {error && <div className="warn-card" role="alert">{error}</div>}
      {data && (
        <section className="insp-section">
          <h3>Where It Is Mentioned <span className="insp-note">{c.passages ?? 0} passage{c.passages === 1 ? '' : 's'}</span></h3>
          {(c.evidence ?? []).map((ev) => <Excerpt key={ev.chunk_id} ev={ev} />)}
        </section>
      )}
    </>
  )
}

const STATUS_EXPLAIN = {
  proposed: 'Extracted by the local model from one Reel. The quote below was checked against the Reel’s text, which shows where it came from — not that it is true.',
  corroborated: 'Proposed independently from two or more Reels. Still not reviewed by a person.',
  verified: 'Reviewed and accepted by a person.',
}

function AssertionView({ id, model, onSelect }) {
  const a = model.assertions[id]
  if (!a) return <p className="insp-empty">This relationship is no longer loaded.</p>
  const reel = model.reels[a.reel_id] ?? {}
  return (
    <>
      <div className="insp-kicker">Extracted Relationship</div>
      <h2 className="insp-title">{a.subject} <span className="pred">{predicateLabel(a.predicate)}</span> {a.object}</h2>
      <div className="insp-pillrow"><StatusPill status={a.status} />{a.supporting_reels > 1 && <span className="insp-small">from {a.supporting_reels} Reels</span>}</div>
      <p className="insp-small">{STATUS_EXPLAIN[a.status]}</p>
      {a.quote && (
        <section className="insp-section">
          <h3>Quote From the Reel</h3>
          <figure className="excerpt"><blockquote>“{a.quote}”</blockquote>
            <figcaption><span className="ev-type">{contentType(a.chunk_id?.split(':')[1]).label}</span> · {reel.owner ? `@${reel.owner}` : a.reel_id}</figcaption></figure>
        </section>
      )}
      <div className="insp-actions">
        <button type="button" className="btn btn-sm" onClick={() => onSelect({ kind: 'concept', id: a.subject_key, focus: true })}>{a.subject}</button>
        <button type="button" className="btn btn-sm" onClick={() => onSelect({ kind: 'concept', id: a.object_key, focus: true })}>{a.object}</button>
        <a className="btn btn-sm" href={reelUrl(a.reel_id, reel.source_url)} target="_blank" rel="noreferrer">Instagram ↗</a>
      </div>
    </>
  )
}

/** A citation from a chat answer: the exact source passage the answer cited. */
function SourceView({ sel, onSelect, graphReady }) {
  const { source, citation } = sel
  const rid = citation?.reel_id ?? source?.reel_id
  const ev = { text: source?.text ?? '', content_type: citation?.content_type ?? source?.chunk_id?.split(':')[1], start: citation?.start }
  return (
    <>
      <div className="insp-kicker">Cited Source {sel.sid}</div>
      <h2 className="insp-title">{citation?.owner ? `@${citation.owner}` : rid}</h2>
      <p className="insp-small">Original text the answer cited. A citation shows where a statement comes from; it does not guarantee
        the answer describes it correctly.</p>
      {ev.text ? <Excerpt ev={ev} /> : <p className="insp-empty">The passage text is not available for this citation.</p>}
      <div className="insp-actions">
        <a className="btn btn-sm btn-primary" href={reelUrl(rid, citation?.source_url)} target="_blank" rel="noreferrer">Open on Instagram ↗</a>
        {graphReady && <button type="button" className="btn btn-sm" onClick={() => onSelect({ kind: 'reel', id: rid, focus: true })}>Show Reel in Graph</button>}
      </div>
    </>
  )
}

export default function Inspector({ selection, model, graph, onSelect, onClose, onAsk, graphReady = true }) {
  if (!selection) return null
  let body = null
  if (selection.kind === 'reel') body = <ReelView id={selection.id} model={model} graph={graph} onSelect={onSelect} onAsk={onAsk} />
  if (selection.kind === 'concept') body = <ConceptView id={selection.id} model={model} graph={graph} onSelect={onSelect} onAsk={onAsk} />
  if (selection.kind === 'mention') body = <MentionView sel={selection} model={model} graph={graph} onSelect={onSelect} />
  if (selection.kind === 'assertion') body = <AssertionView id={selection.id} model={model} onSelect={onSelect} />
  if (selection.kind === 'source') body = <SourceView sel={selection} onSelect={onSelect} graphReady={graphReady} />
  return (
    <aside className="inspector" aria-label="Inspector">
      <button type="button" className="icon-btn insp-close" onClick={onClose} aria-label="Close inspector">
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" /></svg>
      </button>
      <div className="insp-body" key={JSON.stringify([selection.kind, selection.id, selection.reel_id, selection.concept_key, selection.sid, selection.turnId])}>{body}</div>
    </aside>
  )
}