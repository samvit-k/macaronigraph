// Collapsible legend card (SanD Hacks pattern: bottom-left card that collapses to a pill, and tells the canvas to
// refit around it). Content: MacaroniGraph's node kinds and the four edge kinds.
import { useEffect, useRef, useState } from 'react'
import { LEGEND_EVENT } from '../graph/viewport.js'
import { CONCEPT_ORDER, CONCEPT_TYPES, EDGE_KINDS, EDGE_ORDER, REEL } from '../graph/vocab.js'

export function LineSwatch({ color, line, width = 1.6, opacity = 1 }) {
  const dash = line === 'dashed' ? '6 4' : line === 'dotted' ? '2 2.5' : undefined
  return (
    <svg className="line-swatch" viewBox="0 0 28 8" aria-hidden="true">
      <line x1="1" y1="4" x2="27" y2="4" stroke={color} strokeWidth={width} strokeDasharray={dash} strokeLinecap="round" opacity={opacity} />
    </svg>
  )
}

const startsOpen = () => typeof window === 'undefined' || !window.matchMedia?.('(max-width: 1100px)').matches

export default function Legend() {
  const [open, setOpen] = useState(startsOpen)
  const mounted = useRef(false)
  useEffect(() => {
    if (!mounted.current) { mounted.current = true; return }
    window.dispatchEvent(new Event(LEGEND_EVENT))
  }, [open])
  if (!open) return <button type="button" className="legend-pill" onClick={() => setOpen(true)}>Legend</button>
  return (
    <section className="legend" aria-label="Legend">
      <h4>Legend <button type="button" className="tg" onClick={() => setOpen(false)}>Hide</button></h4>
      <div className="legend-sub">Nodes</div>
      <div className="legend-grid">
        <span className="legend-item" title={REEL.help}><span className="node-dot" style={{ background: REEL.color }} />Reel</span>
        {CONCEPT_ORDER.map((k) => (
          <span className="legend-item" key={k}><span className="node-dot" style={{ background: CONCEPT_TYPES[k].color }} />{CONCEPT_TYPES[k].label}</span>
        ))}
        <span className="legend-item" title="Concepts mentioned by only one Reel (shown when you expand it)">
          <span className="node-dot faint" />One-Reel Concept</span>
      </div>
      <div className="legend-sub">Edges</div>
      <div className="legend-edges">
        {EDGE_ORDER.map((k) => (
          <span className="legend-item" key={k} title={EDGE_KINDS[k].help}>
            <LineSwatch color={EDGE_KINDS[k].color} line={EDGE_KINDS[k].line} width={EDGE_KINDS[k].width} />{EDGE_KINDS[k].label}
          </span>
        ))}
      </div>
      <p className="legend-foot">Note: A concept shared by two Reels means both mention it.</p>
    </section>
  )
}