// Filter popover (SanD Hacks pattern: checkbox + count per type, view toggles, reset).
import { useState } from 'react'
import { DEFAULT_FILTERS, filterCounts } from '../graph/toElements.js'
import { CONCEPT_ORDER, CONCEPT_TYPES, EDGE_KINDS, EDGE_ORDER } from '../graph/vocab.js'
import { LineSwatch } from './Legend.jsx'

export function activeFilterCount(filters) {
  return CONCEPT_ORDER.filter((k) => filters.types[k] === false).length +
    EDGE_ORDER.filter((k) => filters.edges[k] === false).length + (filters.bridgeOnly ? 1 : 0)
}

export default function FilterPopover({ model, filters, onChange, disabled = false }) {
  const [open, setOpen] = useState(false)
  const counts = filterCounts(model)
  const off = activeFilterCount(filters)
  const setType = (k, v) => onChange({ ...filters, types: { ...filters.types, [k]: v } })
  const setEdge = (k, v) => onChange({ ...filters, edges: { ...filters.edges, [k]: v } })
  return (
    <div className="filter">
      <button type="button" className={`btn ${off ? 'btn-on' : ''}`} onClick={() => setOpen((v) => !v)} aria-expanded={open} disabled={disabled}>
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 4h11M4.5 8h7M6.5 12h3" /></svg>
        Filter{off ? ` · ${off}` : ''}
      </button>
      {open && (
        <>
          <div className="backdrop" onClick={() => setOpen(false)} />
          <div className="popover filter-panel" role="dialog" aria-label="Graph filters">
            <div className="pop-title">Concept Types</div>
            {CONCEPT_ORDER.map((k) => (
              <label className="pop-row" key={k}>
                <input type="checkbox" checked={filters.types[k] !== false} onChange={(e) => setType(k, e.target.checked)} />
                <span className="node-dot sm" style={{ background: CONCEPT_TYPES[k].color }} />
                <span className="name">{CONCEPT_TYPES[k].label}</span>
                <span className="count">{counts.types[k] ?? 0}</span>
              </label>
            ))}
            <div className="pop-sep" />
            <div className="pop-title">Edges</div>
            {EDGE_ORDER.map((k) => (
              <label className="pop-row" key={k} title={EDGE_KINDS[k].help}>
                <input type="checkbox" checked={filters.edges[k] !== false} onChange={(e) => setEdge(k, e.target.checked)} />
                <LineSwatch color={EDGE_KINDS[k].color} line={EDGE_KINDS[k].line} width={EDGE_KINDS[k].width} />
                <span className="name">{EDGE_KINDS[k].label}</span>
                <span className="count">{counts.edges[k] ?? 0}</span>
              </label>
            ))}
            <div className="pop-sep" />
            <label className="pop-row plain">
              <input type="checkbox" checked={filters.bridgeOnly} onChange={(e) => onChange({ ...filters, bridgeOnly: e.target.checked })} />
              <span className="name">Only Concepts Shared by Several Reels <span className="count">(hides {counts.single})</span></span>
            </label>
            <button type="button" className="link-btn" onClick={() => onChange(DEFAULT_FILTERS)}>Reset Filters</button>
          </div>
        </>
      )}
    </div>
  )
}