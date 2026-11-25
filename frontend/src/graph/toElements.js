// Graph model for the workspace: a merge of the server's overview and of every neighbourhood the user expanded.
// Nothing here invents structure: every node and edge comes from /api/graph/* responses.
import { ASSERTION_STATUSES, CONCEPT_ORDER } from './vocab.js'

export const reelId = (rid) => `reel:${rid}`
export const conceptId = (key) => `concept:${key}`
export const mentionId = (rid, key) => `m:${rid}|${key}`
export const assertionId = (id) => `a:${id}`

export function emptyModel() {
  return { reels: {}, concepts: {}, mentions: {}, assertions: {}, expanded: {}, totals: null, labels: {}, nextLabel: 1 }
}

/** Every node is labelled "Node N"; details appear only in the inspector. A number, once given, stays with the
 *  node for the whole session (also after collapsing or returning to the overview). */
function assignLabels(model, ids) {
  const labels = { ...model.labels }
  let n = model.nextLabel ?? 1
  for (const id of ids) if (labels[id] == null) labels[id] = n++
  return { ...model, labels, nextLabel: n }
}
export const nodeLabel = (model, id) => (model.labels?.[id] != null ? `Node ${model.labels[id]}` : 'Node')

function addConcept(model, c, extra = {}) {
  const prev = model.concepts[c.key] ?? {}
  model.concepts[c.key] = { ...prev, ...c, ...extra, bridge: prev.bridge || c.bridge || (c.n_reels ?? 0) >= 2 }
}

/** Merge a /api/graph/overview payload (replaces totals; keeps anything expanded earlier). */
export function mergeOverview(model, data) {
  const next = { ...model, reels: { ...model.reels }, concepts: { ...model.concepts }, mentions: { ...model.mentions },
    assertions: { ...model.assertions } }
  for (const r of data.reels ?? []) next.reels[r.reel_id] = { ...next.reels[r.reel_id], ...r }
  for (const c of data.concepts ?? []) addConcept(next, c)
  for (const m of data.mentions ?? []) next.mentions[`${m.reel_id}|${m.concept_key}`] = m
  for (const a of data.assertions ?? []) next.assertions[a.id] = a
  next.totals = data.totals ?? null
  return assignLabels(next, [...(data.reels ?? []).map((r) => reelId(r.reel_id)), ...(data.concepts ?? []).map((c) => conceptId(c.key))])
}

/** Merge a /api/graph/neighborhood payload and mark that node expanded. */
export function mergeNeighborhood(model, data) {
  const next = { ...model, reels: { ...model.reels }, concepts: { ...model.concepts }, mentions: { ...model.mentions },
    assertions: { ...model.assertions }, expanded: { ...model.expanded } }
  if (data.kind === 'reel' && data.reel) {
    const rid = data.reel.reel_id
    next.reels[rid] = { ...next.reels[rid], ...data.reel }
    for (const c of data.concepts ?? []) addConcept(next, c)
    next.expanded[reelId(rid)] = (data.concepts ?? []).map((c) => conceptId(c.key))
  }
  if (data.kind === 'concept' && data.concept) {
    addConcept(next, data.concept)
    for (const r of data.reels ?? []) next.reels[r.reel_id] = { ...next.reels[r.reel_id], ...r }
    for (const c of data.related_concepts ?? []) addConcept(next, c)
    next.expanded[conceptId(data.concept.key)] = (data.reels ?? []).map((r) => reelId(r.reel_id))
  }
  for (const m of data.mentions ?? []) next.mentions[`${m.reel_id}|${m.concept_key}`] = m
  for (const a of data.assertions ?? []) next.assertions[a.id] = a
  const ids = [...Object.keys(next.reels).map(reelId), ...Object.keys(next.concepts).map(conceptId)]
  return assignLabels(next, ids)
}

/** Remove what an expansion added, unless the overview or another expansion still needs it. */
export function collapse(model, nodeId, overview) {
  const added = model.expanded[nodeId]
  if (!added) return model
  const expanded = { ...model.expanded }
  delete expanded[nodeId]
  const keep = new Set([nodeId])
  for (const ids of Object.values(expanded)) for (const id of ids) keep.add(id)
  for (const k of Object.keys(expanded)) keep.add(k)
  const base = overview ?? emptyModel()
  const next = { ...model, expanded, reels: { ...model.reels }, concepts: { ...model.concepts }, mentions: { ...model.mentions } }
  for (const id of added) {
    if (keep.has(id)) continue
    if (id.startsWith('concept:')) {
      const key = id.slice(8)
      if (!base.concepts[key] && !Object.values(next.assertions).some((a) => a.subject_key === key || a.object_key === key)) {
        delete next.concepts[key]
      }
    } else if (id.startsWith('reel:')) {
      const rid = id.slice(5)
      if (!base.reels[rid]) delete next.reels[rid]
    }
  }
  for (const k of Object.keys(next.mentions)) {
    const [rid, key] = k.split('|')
    if (!next.reels[rid] || !next.concepts[key]) delete next.mentions[k]
  }
  return next
}

const conceptSize = (n) => Math.round(Math.min(34, 9 + 5 * Math.sqrt(Math.max(1, n ?? 1))))

/** Cytoscape elements for the current model. */
export function buildElements(model) {
  const nodes = []
  const edges = []
  for (const r of Object.values(model.reels)) {
    nodes.push({ group: 'nodes', classes: 'reel', data: {
      id: reelId(r.reel_id), kind: 'reel', reel_id: r.reel_id, size: 20,
      displayLabel: nodeLabel(model, reelId(r.reel_id)), expanded: !!model.expanded[reelId(r.reel_id)] } })
  }
  for (const c of Object.values(model.concepts)) {
    nodes.push({ group: 'nodes', classes: `concept t-${c.type ?? 'concept'}${c.bridge ? ' bridge' : ' single'}`, data: {
      id: conceptId(c.key), kind: 'concept', key: c.key, ctype: c.type ?? 'concept', n_reels: c.n_reels ?? 1,
      bridge: !!c.bridge, size: conceptSize(c.n_reels), displayLabel: nodeLabel(model, conceptId(c.key)), expanded: !!model.expanded[conceptId(c.key)] } })
  }
  for (const m of Object.values(model.mentions)) {
    if (!model.reels[m.reel_id] || !model.concepts[m.concept_key]) continue
    edges.push({ group: 'edges', classes: 'mention', data: {
      id: mentionId(m.reel_id, m.concept_key), kind: 'mention', source: reelId(m.reel_id),
      target: conceptId(m.concept_key), reel_id: m.reel_id, concept_key: m.concept_key, passages: m.passages ?? 1 } })
  }
  for (const a of Object.values(model.assertions)) {
    if (!model.concepts[a.subject_key] || !model.concepts[a.object_key] || a.status === 'rejected') continue
    const data = { id: assertionId(a.id), kind: 'assertion', status: a.status, assertion_id: a.id, predicate: a.predicate,
      source: conceptId(a.subject_key), target: conceptId(a.object_key) }
    edges.push({ group: 'edges', classes: `assertion status-${a.status}`, data })
    edges.push({ group: 'edges', classes: 'assertion-hit', data: { ...data, id: `${data.id}:hit`, hitFor: data.id } })
  }
  return { nodes, edges }
}

export const DEFAULT_FILTERS = {
  types: Object.fromEntries(CONCEPT_ORDER.map((t) => [t, true])),
  edges: { mention: true, proposed: true, corroborated: true, verified: true },
  bridgeOnly: false,
}

/** Element ids hidden by the filters (concept types, edge kinds, bridge-only). */
export function hiddenIds(model, filters = DEFAULT_FILTERS) {
  const hidden = new Set()
  for (const c of Object.values(model.concepts)) {
    const typeOff = filters.types?.[c.type ?? 'concept'] === false
    if (typeOff || (filters.bridgeOnly && !c.bridge)) hidden.add(conceptId(c.key))
  }
  for (const m of Object.values(model.mentions)) {
    const id = mentionId(m.reel_id, m.concept_key)
    if (filters.edges?.mention === false || hidden.has(conceptId(m.concept_key))) hidden.add(id)
  }
  for (const a of Object.values(model.assertions)) {
    const id = assertionId(a.id)
    if (filters.edges?.[a.status] === false || hidden.has(conceptId(a.subject_key)) || hidden.has(conceptId(a.object_key))) {
      hidden.add(id)
      hidden.add(`${id}:hit`)
    }
  }
  return hidden
}

/** How many loaded items each filter row controls. */
export function filterCounts(model) {
  const types = {}
  for (const c of Object.values(model.concepts)) types[c.type ?? 'concept'] = (types[c.type ?? 'concept'] ?? 0) + 1
  const edges = { mention: Object.keys(model.mentions).length }
  for (const s of ASSERTION_STATUSES) edges[s] = Object.values(model.assertions).filter((a) => a.status === s).length
  return { types, edges, single: Object.values(model.concepts).filter((c) => !c.bridge).length }
}

/** Ids of loaded nodes that match a free-text query (used to dim the rest while typing). */
export function matchingIds(model, query) {
  const q = (query ?? '').trim().toLowerCase()
  if (!q) return null
  const out = new Set()
  for (const c of Object.values(model.concepts)) {
    if ((c.name ?? '').toLowerCase().includes(q) || (c.aliases ?? []).some((a) => a.toLowerCase().includes(q))) out.add(conceptId(c.key))
  }
  for (const r of Object.values(model.reels)) {
    if ((r.owner ?? '').toLowerCase().includes(q) || (r.caption ?? '').toLowerCase().includes(q)) out.add(reelId(r.reel_id))
  }
  return out
}

/** Graph element ids to highlight for an answer: its cited Reels and the concepts of its cited passages. */
export function answerHighlight(turn) {
  if (!turn) return {}
  const out = {}
  for (const c of turn.citations ?? []) if (c.reel_id) out[reelId(c.reel_id)] = 'cited'
  for (const c of turn.cited_concepts ?? []) if (c.key) out[conceptId(c.key)] = 'cited'
  return out
}