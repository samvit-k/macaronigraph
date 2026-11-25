// Interactive canvas. The Cytoscape lifecycle, dot background tied to the viewport, hover hints, invisible wide
// hit-edges, autofit around the legend and focus-on-selection are taken from the SanD Hacks GraphCanvas; the element
// model, layout and selection kinds are MacaroniGraph's (Reel / concept / mention / assertion).
import cytoscape from 'cytoscape'
import { useEffect, useRef, useState } from 'react'
import { computeLayout } from '../graph/layout.js'
import { buildStyle } from '../graph/style.js'
import { assertionId, buildElements, conceptId, mentionId, reelId } from '../graph/toElements.js'
import { createAutoFit, GRAPH_MAX_ZOOM, GRAPH_MIN_ZOOM, LEGEND_EVENT } from '../graph/viewport.js'
import { conceptType, EDGE_KINDS } from '../graph/vocab.js'
import ZoomControls from './ZoomControls.jsx'

const DOT_SPACING = 22
const VIEW_MS = 220

/** Label sizes for the single visible label (hover / selection): constant on screen at any zoom. */
export function labelScale(cy) {
  const z = cy.zoom() || 1
  cy.batch(() => cy.nodes().data({ lfs: 13 / z, lpad: 3 / z, lbw: 1 / z }))
}

function legendRect(container) {
  const card = container?.closest('.stage')?.querySelector('.legend')
  if (!card) return null
  const c = container.getBoundingClientRect()
  const r = card.getBoundingClientRect()
  return { left: r.left - c.left, right: r.right - c.left, top: r.top - c.top, bottom: r.bottom - c.top, width: r.width, height: r.height }
}

/** App selection ({kind, ...}) → Cytoscape element id. */
export function elementIdFor(sel) {
  if (!sel) return null
  if (sel.kind === 'reel') return reelId(sel.id)
  if (sel.kind === 'concept') return conceptId(sel.id)
  if (sel.kind === 'mention') return mentionId(sel.reel_id, sel.concept_key)
  if (sel.kind === 'assertion') return assertionId(sel.id)
  return null
}

export function selectionFor(ele) {
  const d = ele.data()
  if (d.kind === 'reel') return { kind: 'reel', id: d.reel_id }
  if (d.kind === 'concept') return { kind: 'concept', id: d.key }
  if (d.kind === 'mention') return { kind: 'mention', reel_id: d.reel_id, concept_key: d.concept_key }
  if (d.kind === 'assertion') return { kind: 'assertion', id: d.assertion_id }
  return null
}

// Hover hints name only the KIND of item (the colour already shows it); details appear on click.
function tipText(ele) {
  const d = ele.data()
  if (d.kind === 'mention') return 'Mention · Not a Relationship · Click for Evidence'
  if (d.kind === 'assertion') return `${EDGE_KINDS[d.status]?.label ?? 'Relationship'} · Click for Details`
  const kind = d.kind === 'reel' ? 'Reel' : conceptType(d.ctype).label
  return `${d.displayLabel} · ${kind} · Click for Details`
}

export default function GraphCanvas({ model, hidden, matches, selection, highlight, onSelect, onToggleExpand }) {
  const containerRef = useRef(null)
  const cyRef = useRef(null)
  const posRef = useRef({})
  const fitRef = useRef(null)
  const lastHighlight = useRef('')
  const cb = useRef({ onSelect, onToggleExpand })
  cb.current = { onSelect, onToggleExpand }
  const [zoom, setZoom] = useState(1)
  const [tip, setTip] = useState(null)

  useEffect(() => {
    const cy = cytoscape({
      container: containerRef.current, style: buildStyle(), minZoom: GRAPH_MIN_ZOOM, maxZoom: GRAPH_MAX_ZOOM,
      wheelSensitivity: 0.25, boxSelectionEnabled: false, selectionType: 'single',
      // nodes are always drawn above edges (edge z-indexes stay below the node z-index)
    })
    cyRef.current = cy
    // Browser tests (frontend/e2e) read node positions; only exposed when the page is opened with ?e2e.
    if (typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('e2e')) window.__mgCy = cy
    let frame = 0
    cy.on('viewport', () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        const z = cy.zoom()
        const { x, y } = cy.pan()
        const el = containerRef.current
        if (el) {
          el.style.backgroundSize = `${DOT_SPACING * z}px ${DOT_SPACING * z}px`
          el.style.backgroundPosition = `${x}px ${y}px`
        }
        setZoom(z)
        setTip(null)
        labelScale(cy)
      })
    })
    const autofit = createAutoFit(cy, () => legendRect(containerRef.current))
    fitRef.current = autofit
    let resizeFrame = 0
    const resizer = typeof ResizeObserver !== 'undefined'
      ? new ResizeObserver(() => { cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(() => autofit.resized(VIEW_MS)) })
      : null
    resizer?.observe(containerRef.current)
    let legendFrame = 0
    const onLegend = () => { cancelAnimationFrame(legendFrame); legendFrame = requestAnimationFrame(() => autofit.refitIfFollowing(VIEW_MS)) }
    window.addEventListener(LEGEND_EVENT, onLegend)
    const visual = (ele) => (ele.data('hitFor') ? cy.getElementById(ele.data('hitFor')) : ele)
    cy.on('tap', (evt) => { if (evt.target === cy) cb.current.onSelect(null) })
    cy.on('tap', 'node, edge.mention, edge.assertion-hit', (evt) => { setTip(null); cb.current.onSelect(selectionFor(visual(evt.target))) })
    cy.on('dbltap', 'node', (evt) => { const s = selectionFor(evt.target); if (s) cb.current.onToggleExpand?.(s) })
    cy.on('mouseover', 'node, edge.mention, edge.assertion-hit', (evt) => {
      const ele = visual(evt.target)
      ele.addClass('hover')
      if (containerRef.current) containerRef.current.style.cursor = 'pointer'
      const p = evt.renderedPosition ?? ele.renderedPosition?.()
      if (p) setTip({ x: p.x, y: p.y, text: tipText(ele) })
    })
    cy.on('mouseout', 'node, edge.mention, edge.assertion-hit', (evt) => {
      visual(evt.target).removeClass('hover')
      if (containerRef.current) containerRef.current.style.cursor = 'default'
      setTip(null)
    })
    return () => {
      cancelAnimationFrame(frame); cancelAnimationFrame(resizeFrame); cancelAnimationFrame(legendFrame)
      window.removeEventListener(LEGEND_EVENT, onLegend)
      resizer?.disconnect(); autofit.dispose(); fitRef.current = null
      cy.destroy(); cyRef.current = null
    }
  }, [])

  // Rebuild on model change; kept nodes glide from their old to their new position.
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return undefined
    const elements = buildElements(model)
    const old = {}
    cy.nodes().forEach((n) => { old[n.id()] = { ...n.position() } })
    const pos = computeLayout(elements, posRef.current)
    posRef.current = pos
    const first = Object.keys(old).length === 0
    cy.batch(() => {
      cy.elements().remove()
      cy.add(elements.nodes.map((n) => ({ ...n, position: { ...(old[n.data.id] ?? pos[n.data.id] ?? { x: 0, y: 0 }) } })))
      cy.add(elements.edges)
    })
    labelScale(cy)
    if (first) {
      fitRef.current?.fit(0)
      labelScale(cy)
      return undefined
    }
    cy.nodes().forEach((n) => { const p = pos[n.id()]; if (p) n.animate({ position: p }, { duration: 420, easing: 'ease-out-cubic' }) })
    const t = setTimeout(() => fitRef.current?.refitIfFollowing(VIEW_MS), 450)
    return () => clearTimeout(t)
  }, [model])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.batch(() => cy.elements().forEach((e) => e.toggleClass('hidden', hidden?.has(e.id()) ?? false)))
  }, [model, hidden])

  // Selection focus, search matches and answer highlights.
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    const selId = elementIdFor(selection)
    const sel = selId ? cy.getElementById(selId) : null
    cy.batch(() => {
      cy.elements().removeClass('faded match cited')
      cy.elements(':selected').unselect()
      for (const id of Object.keys(highlight ?? {})) cy.getElementById(id).addClass('cited')
      if (matches?.size) {
        cy.nodes().forEach((n) => { if (matches.has(n.id())) n.addClass('match'); else n.addClass('faded') })
        cy.edges().forEach((e) => { if (!(matches.has(e.source().id()) && matches.has(e.target().id()))) e.addClass('faded') })
      }
      const cited = cy.nodes('.cited')
      if (sel && sel.nonempty()) {
        sel.select()
        const keep = sel.isEdge() ? sel.union(sel.connectedNodes()) : sel.closedNeighborhood()
        cy.elements().not(keep).addClass('faded')
        keep.removeClass('faded')
      } else if (cited.nonempty() && !matches?.size) {
        // answer highlight: dim everything except the cited Reels/concepts and the edges between them
        const keep = cited.union(cited.edgesWith(cited))
        cy.elements().not(keep).addClass('faded')
      }
      cited.removeClass('faded') // cited nodes are never dimmed
    })
    const hKey = Object.keys(highlight ?? {}).sort().join('|')
    if (hKey && hKey !== lastHighlight.current && !(sel && sel.nonempty())) {
      lastHighlight.current = hKey
      const cited = cy.nodes('.cited')
      if (cited.nonempty()) {
        fitRef.current?.userAdjusted()
        cy.animate({ fit: { eles: cited.closedNeighborhood(), padding: 120 } }, { duration: 380, easing: 'ease-in-out-cubic' })
      }
    }
    if (!hKey) lastHighlight.current = ''
    // A selection made on the canvas is not re-centred, unless the floating inspector card would cover it.
    let covered = false
    if (sel && sel.nonempty() && !selection?.focus) {
      const card = containerRef.current?.closest('.stage')?.querySelector('.inspector')
      const rb = sel.renderedBoundingBox()
      if (card && cy.width() - card.offsetWidth - 16 > 320) covered = rb.x2 > cy.width() - card.offsetWidth - 24
    }
    if (sel && sel.nonempty() && (selection?.focus || covered)) {
      fitRef.current?.userAdjusted()
      const card = containerRef.current?.closest('.stage')?.querySelector('.inspector')
      const cardW = card ? card.offsetWidth + 16 : 0
      const sideBySide = cardW && cy.width() - cardW > 320
      const visibleW = sideBySide ? cy.width() - cardW : cy.width()
      const visibleH = !sideBySide && card ? cy.height() - card.offsetHeight - 8 : cy.height()
      const bb = sel.isNode() ? sel.closedNeighborhood().boundingBox() : sel.boundingBox()
      const fitZoom = Math.min((visibleW - 80) / Math.max(bb.w, 1), (visibleH - 80) / Math.max(bb.h, 1))
      const z = Math.max(0.3, Math.min(Math.max(cy.zoom(), 0.9), fitZoom, 1.6))
      cy.animate({ zoom: z, pan: { x: visibleW / 2 - ((bb.x1 + bb.x2) / 2) * z, y: visibleH / 2 - ((bb.y1 + bb.y2) / 2) * z } },
        { duration: 380, easing: 'ease-in-out-cubic' })
    }
  }, [model, selection, matches, highlight])

  const zoomBy = (f) => {
    const cy = cyRef.current
    if (!cy) return
    fitRef.current?.userAdjusted()
    cy.animate({ zoom: { level: cy.zoom() * f, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } } }, { duration: 160 })
  }
  const fit = () => fitRef.current?.fit(300)
  const relayout = () => {
    const cy = cyRef.current
    if (!cy) return
    const pos = computeLayout(buildElements(model), {})
    posRef.current = pos
    cy.nodes().forEach((n) => pos[n.id()] && n.animate({ position: pos[n.id()] }, { duration: 450 }))
    setTimeout(fit, 480)
  }

  return (
    <div className="canvas-wrap">
      <div ref={containerRef} className="canvas" data-testid="graph-canvas" />
      {tip && <div className="graph-tip" style={{ left: tip.x + 12, top: tip.y + 12 }} role="tooltip">{tip.text}</div>}
      <ZoomControls zoom={zoom} onZoomIn={() => zoomBy(1.25)} onZoomOut={() => zoomBy(0.8)} onFit={fit} onRelayout={relayout} />
    </div>
  )
}