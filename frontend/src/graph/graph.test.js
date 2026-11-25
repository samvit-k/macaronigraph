import { describe, expect, it } from 'vitest'
import { computeLayout } from './layout.js'
import { answerHighlight, buildElements, collapse, DEFAULT_FILTERS, emptyModel, filterCounts, hiddenIds, matchingIds,
  mergeNeighborhood, mergeOverview } from './toElements.js'
import { EDGE_KINDS, timestamp } from './vocab.js'

const OVERVIEW = {
  reels: [{ reel_id: 'R1', owner: 'alice', caption: 'CUDA basics' }, { reel_id: 'R2', owner: 'bob', caption: 'GPUs' }],
  concepts: [{ key: 'cuda', name: 'CUDA', type: 'technology', n_reels: 2, bridge: true },
    { key: 'arm', name: 'ARM', type: 'technology', n_reels: 1, bridge: false },
    { key: 'risc', name: 'RISC', type: 'concept', n_reels: 1, bridge: false }],
  mentions: [{ reel_id: 'R1', concept_key: 'cuda', passages: 3 }, { reel_id: 'R2', concept_key: 'cuda', passages: 1 }],
  assertions: [{ id: 'a1', subject_key: 'arm', object_key: 'risc', subject: 'ARM', object: 'RISC', predicate: 'USES', status: 'proposed', reel_id: 'R2' }],
  totals: { reels: 2, concepts: 3 },
}
const NBH = {
  kind: 'reel', reel: { reel_id: 'R1', owner: 'alice' },
  concepts: [{ key: 'cuda', name: 'CUDA', type: 'technology', n_reels: 2 }, { key: 'pytorch', name: 'PyTorch', type: 'tool', n_reels: 1 }],
  mentions: [{ reel_id: 'R1', concept_key: 'cuda', passages: 3 }, { reel_id: 'R1', concept_key: 'pytorch', passages: 1 }],
  assertions: [],
}

describe('graph model', () => {
  it('builds Reel, concept, mention and assertion elements (with a click twin for assertions)', () => {
    const m = mergeOverview(emptyModel(), OVERVIEW)
    const { nodes, edges } = buildElements(m)
    expect(nodes.map((n) => n.data.id).sort()).toEqual(['concept:arm', 'concept:cuda', 'concept:risc', 'reel:R1', 'reel:R2'])
    expect(edges.filter((e) => e.data.kind === 'mention')).toHaveLength(2)
    expect(edges.find((e) => e.data.id === 'a:a1').classes).toContain('status-proposed')
    expect(edges.find((e) => e.data.id === 'a:a1:hit').data.hitFor).toBe('a:a1')
    expect(nodes.find((n) => n.data.id === 'reel:R1').data.displayLabel).toBe('Node 1')
    expect(nodes.every((n) => /^Node \d+$/.test(n.data.displayLabel))).toBe(true)
  })

  it('never draws rejected assertions or edges whose endpoints are missing', () => {
    const m = mergeOverview(emptyModel(), { ...OVERVIEW, assertions: [{ ...OVERVIEW.assertions[0], status: 'rejected' },
      { id: 'a2', subject_key: 'arm', object_key: 'nowhere', status: 'proposed' }] })
    expect(buildElements(m).edges.filter((e) => e.data.kind === 'assertion')).toHaveLength(0)
  })

  it('expanding adds a neighbourhood, collapsing removes only what the expansion added', () => {
    const base = mergeOverview(emptyModel(), OVERVIEW)
    const exp = mergeNeighborhood(base, NBH)
    expect(exp.concepts.pytorch).toBeTruthy()
    expect(exp.expanded['reel:R1']).toEqual(['concept:cuda', 'concept:pytorch'])
    const back = collapse(exp, 'reel:R1', base)
    expect(back.concepts.pytorch).toBeUndefined()
    expect(back.concepts.cuda).toBeTruthy() // still part of the overview
    expect(back.mentions['R1|pytorch']).toBeUndefined()
    expect(back.expanded['reel:R1']).toBeUndefined()
  })

  it('numbers nodes once and keeps each number across expand, collapse and reset', () => {
    const base = mergeOverview(emptyModel(), OVERVIEW)
    expect(base.labels).toEqual({ 'reel:R1': 1, 'reel:R2': 2, 'concept:cuda': 3, 'concept:arm': 4, 'concept:risc': 5 })
    const exp = mergeNeighborhood(base, NBH)
    expect(exp.labels['concept:pytorch']).toBe(6)
    const back = collapse(exp, 'reel:R1', base)
    const again = mergeNeighborhood(back, NBH)
    expect(again.labels['concept:pytorch']).toBe(6) // same node, same number
  })

  it('filters hide concept types, edge kinds and single-Reel concepts (with their edges)', () => {
    const m = mergeNeighborhood(mergeOverview(emptyModel(), OVERVIEW), NBH)
    const h1 = hiddenIds(m, { ...DEFAULT_FILTERS, types: { ...DEFAULT_FILTERS.types, tool: false } })
    expect(h1.has('concept:pytorch') && h1.has('m:R1|pytorch')).toBe(true)
    const h2 = hiddenIds(m, { ...DEFAULT_FILTERS, edges: { ...DEFAULT_FILTERS.edges, mention: false } })
    expect(h2.has('m:R1|cuda') && !h2.has('concept:cuda')).toBe(true)
    const h3 = hiddenIds(m, { ...DEFAULT_FILTERS, bridgeOnly: true })
    expect(h3.has('concept:pytorch') && h3.has('a:a1') && h3.has('a:a1:hit') && !h3.has('concept:cuda')).toBe(true)
    expect(filterCounts(m).edges).toMatchObject({ mention: 3, proposed: 1 })
  })

  it('search matches names, aliases, creators and captions of loaded nodes', () => {
    const m = mergeOverview(emptyModel(), OVERVIEW)
    expect([...matchingIds(m, 'cud')]).toEqual(expect.arrayContaining(['concept:cuda', 'reel:R1']))
    expect(matchingIds(m, '  ')).toBeNull()
  })

  it('an answer highlights its cited Reels and the concepts of the cited passages', () => {
    expect(answerHighlight({ citations: [{ reel_id: 'R1' }], cited_concepts: [{ key: 'cuda' }] }))
      .toEqual({ 'reel:R1': 'cited', 'concept:cuda': 'cited' })
    expect(answerHighlight(null)).toEqual({})
  })
})

describe('layout', () => {
  it('is deterministic and keeps existing positions as the starting point', () => {
    const el = buildElements(mergeOverview(emptyModel(), OVERVIEW))
    const a = computeLayout(el)
    const b = computeLayout(el)
    expect(a).toEqual(b)
    expect(Object.keys(a).sort()).toEqual(el.nodes.map((n) => n.data.id).sort())
    const grown = buildElements(mergeNeighborhood(mergeOverview(emptyModel(), OVERVIEW), NBH))
    const c = computeLayout(grown, a)
    const moved = Math.hypot(c['reel:R1'].x - a['reel:R1'].x, c['reel:R1'].y - a['reel:R1'].y)
    expect(moved).toBeLessThan(150) // grows in place instead of reshuffling
  })
})

describe('vocabulary', () => {
  it('keeps mentions visually distinct from every relationship status', () => {
    expect(EDGE_KINDS.mention.line).toBe('dotted')
    expect(new Set(['proposed', 'corroborated', 'verified'].map((k) => `${EDGE_KINDS[k].line}${EDGE_KINDS[k].color}`)).size).toBe(3)
    expect(timestamp(75.4)).toBe('1:15')
    expect(timestamp(undefined)).toBeNull()
  })
})