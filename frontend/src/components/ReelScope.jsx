// Reel scope: the concept count in the top bar opens a picker for how many recent Reels the graph draws.
import { useState } from 'react'

export const SCOPE_STEPS = [25, 50, 100, 200, 400]

export function scopeOptions(total) {
  const steps = SCOPE_STEPS.filter((n) => !total || n < total)
  return total ? [...steps, total] : steps
}

export default function ReelScope({ label, scope, total, onScope, disabled = false }) {
  const [open, setOpen] = useState(false)
  const [custom, setCustom] = useState('')
  const pick = (n) => { onScope(n); setOpen(false); setCustom('') }
  const n = Math.round(Number(custom))
  const valid = Number.isFinite(n) && n > 0 && (!total || n <= total)
  return (
    <div className="scope">
      <button type="button" className="tb-meta scope-btn" data-testid="graph-meta" onClick={() => setOpen((v) => !v)}
        aria-expanded={open} aria-haspopup="dialog" disabled={disabled} title="Choose how many Reels the graph shows">
        {label}
      </button>
      {open && (
        <>
          <div className="backdrop" onClick={() => setOpen(false)} />
          <div className="popover scope-panel" role="dialog" aria-label="Reels in graph">
            <div className="pop-title">Reels in Graph</div>
            <label className="pop-row plain">
              <input type="radio" name="reel-scope" checked={!scope} onChange={() => pick(null)} />
              <span className="name">Overview <span className="count">(shared concepts only)</span></span>
            </label>
            {scopeOptions(total).map((k) => (
              <label className="pop-row plain" key={k}>
                <input type="radio" name="reel-scope" checked={scope === k} onChange={() => pick(k)} />
                <span className="name">{k === total ? `All ${k} Reels` : `${k} Recent Reels`} <span className="count">(all their concepts)</span></span>
              </label>
            ))}
            <div className="pop-sep" />
            <form className="scope-custom" onSubmit={(e) => { e.preventDefault(); if (valid) pick(n) }}>
              <input type="number" min="1" max={total || undefined} placeholder={total ? `1–${total}` : 'Reels'} value={custom}
                onChange={(e) => setCustom(e.target.value)} aria-label="Number of Reels" />
              <button type="submit" className="btn btn-sm" disabled={!valid}>Show</button>
            </form>
            <p className="scope-note">Most recently saved Reels first. Larger graphs take a few seconds to lay out.</p>
          </div>
        </>
      )}
    </div>
  )
}