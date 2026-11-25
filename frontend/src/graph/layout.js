// Deterministic d3-force layout: Reels anchor the canvas, and kept nodes start from their previous positions.
import { forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY } from 'd3-force'

const TICKS = 320
const GOLDEN = Math.PI * (3 - Math.sqrt(5))

function hash(s) {
  let h = 2166136261
  for (let i = 0; i < s.length; i += 1) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
  return (h >>> 0) / 4294967296
}

export function computeLayout(elements, prev = {}) {
  const nodes = (elements?.nodes ?? []).map((n) => n.data)
  if (!nodes.length) return {}
  const reels = nodes.filter((d) => d.kind === 'reel').sort((a, b) => a.id.localeCompare(b.id))
  const spacing = 92
  const seed = {}
  reels.forEach((d, i) => {
    const r = spacing * Math.sqrt(i + 0.5)
    seed[d.id] = { x: r * Math.cos(i * GOLDEN), y: r * Math.sin(i * GOLDEN) }
  })
  const reelsOf = {}
  for (const { data: e } of elements.edges ?? []) {
    if (e.kind === 'mention') (reelsOf[e.target] ??= []).push(e.source)
  }
  for (const d of nodes) {
    if (seed[d.id]) continue
    const rs = (reelsOf[d.id] ?? []).map((id) => prev[id] ?? seed[id]).filter(Boolean)
    const a = 2 * Math.PI * hash(d.id)
    const jitter = 30 + 40 * hash(`${d.id}#r`)
    const cx = rs.length ? rs.reduce((s, p) => s + p.x, 0) / rs.length : 0
    const cy = rs.length ? rs.reduce((s, p) => s + p.y, 0) / rs.length : 0
    seed[d.id] = { x: cx + jitter * Math.cos(a), y: cy + jitter * Math.sin(a) }
  }

  const sim = nodes.map((d) => {
    const p = prev[d.id] ?? seed[d.id]
    return { id: d.id, kind: d.kind, size: d.size ?? 14, x: p.x, y: p.y }
  })
  const ids = new Set(sim.map((n) => n.id))
  const links = []
  for (const { data: e } of elements.edges ?? []) {
    if (e.hitFor || !ids.has(e.source) || !ids.has(e.target)) continue
    if (e.kind === 'mention') links.push({ source: e.source, target: e.target, distance: 78, strength: 0.3 })
    else links.push({ source: e.source, target: e.target, distance: 90, strength: 0.15 })
  }
  const simulation = forceSimulation(sim)
    .force('link', forceLink(links).id((n) => n.id).distance((l) => l.distance).strength((l) => l.strength))
    .force('charge', forceManyBody().strength((n) => (n.kind === 'reel' ? -240 : -60)).distanceMax(520))
    .force('collide', forceCollide((n) => n.size / 2 + (n.kind === 'reel' ? 22 : 12)).strength(0.9))
    .force('x', forceX(0).strength(0.035))
    .force('y', forceY(0).strength(0.035))
    .stop()
  simulation.alpha(Object.keys(prev).length ? 0.4 : 1)
  for (let i = 0; i < TICKS; i += 1) simulation.tick()

  const out = {}
  for (const n of sim) out[n.id] = { x: Math.round(n.x * 10) / 10, y: Math.round(n.y * 10) / 10 }
  return out
}