// Top-bar search (⌘K), adapted from SanD Hacks. Typing dims non-matching nodes already on the canvas; the dropdown
// searches the WHOLE graph on the server (concepts by name/alias, Reels by creator/caption), so concepts that are not
// loaded yet can still be found — picking one loads its neighbourhood and focuses it.
import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client.js'
import { conceptType, REEL } from '../graph/vocab.js'

export default function SearchBox({ query, onQuery, onPick, client = api, disabled = false }) {
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const [res, setRes] = useState({ q: '', items: [], loading: false, error: null })
  const inputRef = useRef(null)

  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); inputRef.current?.focus(); setOpen(true) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    const text = query.trim()
    if (!text || disabled) { setRes({ q: '', items: [], loading: false, error: null }); return undefined }
    const ctrl = new AbortController()
    setRes((r) => ({ ...r, loading: true, error: null }))
    const t = setTimeout(async () => {
      try {
        const data = await client.search(text, ctrl.signal)
        const items = [
          ...(data.concepts ?? []).map((c) => ({ kind: 'concept', id: c.key, label: c.name, meta: `${conceptType(c.type).label} · ${c.n_reels} Reel${c.n_reels === 1 ? '' : 's'}`, ctype: c.type })),
          ...(data.reels ?? []).map((r) => ({ kind: 'reel', id: r.reel_id, label: r.owner ? `@${r.owner}` : r.reel_id, meta: r.caption ?? '' })),
        ]
        setRes({ q: text, items, loading: false, error: null })
        setActive(0)
      } catch (e) {
        if (!ctrl.signal.aborted) setRes({ q: text, items: [], loading: false, error: e.message })
      }
    }, 180)
    return () => { clearTimeout(t); ctrl.abort() }
  }, [query, client, disabled])

  const pick = (hit) => {
    if (!hit) return
    onPick(hit)
    setOpen(false)
    inputRef.current?.blur()
  }

  return (
    <div className="search">
      <svg className="search-icon" viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.5" /><path d="M10.5 10.5 14 14" /></svg>
      <input ref={inputRef} type="search" placeholder={disabled ? 'Graph Offline' : 'Search the Graph…'}
        aria-label="Search concepts and Reels" value={query} disabled={disabled}
        onChange={(e) => { onQuery(e.target.value); setOpen(true) }}
        onFocus={() => setOpen(true)} onBlur={() => setTimeout(() => setOpen(false), 120)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') { e.preventDefault(); setActive((a) => Math.min(a + 1, res.items.length - 1)) }
          if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)) }
          if (e.key === 'Enter') pick(res.items[active])
          if (e.key === 'Escape') { onQuery(''); inputRef.current?.blur() }
        }} />
      <kbd>⌘K</kbd>
      {open && query.trim() && (
        <ul className="search-results" role="listbox" aria-label="Search results">
          {res.loading && res.items.length === 0 && <li className="empty">Searching…</li>}
          {res.error && <li className="empty">Search failed: {res.error}</li>}
          {!res.loading && !res.error && res.items.length === 0 && <li className="empty">No concept or Reel matches “{query.trim()}”</li>}
          {res.items.map((r, i) => (
            <li key={`${r.kind}:${r.id}`} role="option" aria-selected={i === active} className={i === active ? 'active' : ''}
              onMouseDown={(e) => { e.preventDefault(); pick(r) }} onMouseEnter={() => setActive(i)}>
              {r.kind === 'reel' ? <span className="node-dot sm" style={{ background: REEL.color }} /> : <span className="node-dot sm" style={{ background: conceptType(r.ctype).color }} />}
              <span className="text">{r.label}</span>
              <span className="meta">{r.meta}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
