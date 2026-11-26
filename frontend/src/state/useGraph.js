// Graph state: the overview plus expanded neighbourhoods; stale responses are discarded.
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client.js'
import { collapse as collapseModel, conceptId, emptyModel, mergeNeighborhood, mergeOverview, reelId } from '../graph/toElements.js'

const SCOPE_KEY = 'macaronigraph.reelScope'

export function overviewParams(scope) {
  return scope ? { reel_limit: scope, within: 'true', limit_concepts: 10000 } : {}
}

function savedScope() {
  try {
    const n = Number(window.localStorage.getItem(SCOPE_KEY))
    return Number.isInteger(n) && n > 0 ? n : null
  } catch {
    return null
  }
}

export function useGraph({ client = api } = {}) {
  const [scope, setScopeState] = useState(savedScope)
  const [state, setState] = useState({ model: emptyModel(), base: emptyModel(), status: 'loading', error: null })
  const gen = useRef(0)
  const inflight = useRef(new Map())
  const details = useRef(new Map())

  const load = useCallback(async () => {
    const g = ++gen.current
    inflight.current.clear()
    details.current.clear()
    setState((s) => ({ ...s, status: 'loading', error: null }))
    try {
      const data = await client.overview(overviewParams(scope))
      if (g !== gen.current) return
      const base = mergeOverview(emptyModel(), data)
      setState({ model: base, base, status: 'ready', error: null })
    } catch (e) {
      if (g !== gen.current) return
      setState((s) => ({ ...s, status: e.status === 503 ? 'offline' : 'error', error: e.message }))
    }
  }, [client, scope])

  useEffect(() => { load() }, [load])

  const fetchDetail = useCallback(async (kind, id) => {
    const nodeId = kind === 'reel' ? reelId(id) : conceptId(id)
    if (details.current.has(nodeId)) return details.current.get(nodeId)
    if (inflight.current.has(nodeId)) return inflight.current.get(nodeId)
    const g = gen.current
    const p = client.neighborhood(kind, id).then((data) => {
      if (g === gen.current) details.current.set(nodeId, data)
      return data
    }).finally(() => inflight.current.delete(nodeId))
    inflight.current.set(nodeId, p)
    return p
  }, [client])

  const expand = useCallback(async (kind, id) => {
    const g = gen.current
    const data = await fetchDetail(kind, id)
    if (g !== gen.current) return null
    setState((s) => ({ ...s, model: mergeNeighborhood(s.model, data) }))
    return data
  }, [fetchDetail])

  const collapse = useCallback((nodeId) => {
    setState((s) => ({ ...s, model: collapseModel(s.model, nodeId, s.base) }))
  }, [])

  const resetToOverview = useCallback(() => setState((s) => ({ ...s, model: { ...s.base, labels: s.model.labels, nextLabel: s.model.nextLabel } })), [])

  const setScope = useCallback((n) => {
    const next = Number.isInteger(n) && n > 0 ? n : null
    try {
      if (next) window.localStorage.setItem(SCOPE_KEY, String(next))
      else window.localStorage.removeItem(SCOPE_KEY)
    } catch { /* storage unavailable: the choice lasts for this page only */ }
    setScopeState(next)
  }, [])

  return { ...state, scope, setScope, reload: load, fetchDetail, expand, collapse, resetToOverview }
}