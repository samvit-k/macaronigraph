import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client.js'
import { collapse as collapseModel, conceptId, emptyModel, mergeNeighborhood, mergeOverview, reelId } from '../graph/toElements.js'

// Graph state: the overview plus whatever the user expanded. Async safety follows the SanD Hacks pattern: a
// generation counter discards responses that arrive after a reload, and neighbourhood requests are de-duplicated.
export function useGraph({ client = api } = {}) {
  const [state, setState] = useState({ model: emptyModel(), base: emptyModel(), status: 'loading', error: null })
  const gen = useRef(0)
  const inflight = useRef(new Map())
  const details = useRef(new Map()) // nodeId -> neighbourhood payload (inspector cache)

  const load = useCallback(async () => {
    const g = ++gen.current
    inflight.current.clear()
    details.current.clear()
    setState((s) => ({ ...s, status: 'loading', error: null }))
    try {
      const data = await client.overview()
      if (g !== gen.current) return
      const base = mergeOverview(emptyModel(), data)
      setState({ model: base, base, status: 'ready', error: null })
    } catch (e) {
      if (g !== gen.current) return
      setState((s) => ({ ...s, status: e.status === 503 ? 'offline' : 'error', error: e.message }))
    }
  }, [client])

  useEffect(() => { load() }, [load])

  /** Fetch (cached) the neighbourhood of a node: used by the inspector and by expansion. */
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

  return { ...state, reload: load, fetchDetail, expand, collapse, resetToOverview }
}
