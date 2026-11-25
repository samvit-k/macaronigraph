// Conversation state with async safety: stale responses are discarded and queued answers are polled.
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client.js'

// A generation counter discards responses for a conversation that is no longer open, and turns merge by server id.

export const QUEUED = 'queued'
const POLL_START = 2000
const POLL_MAX = 8000

export function mergeTurns(a, b) {
  const byId = new Map()
  for (const t of [...a, ...b]) if (t?.turn_id != null) byId.set(t.turn_id, t)
  return [...byId.values()].sort((x, y) => x.turn_id - y.turn_id)
}

export function useConversation({ client = api, pollStart = POLL_START, pollMax = POLL_MAX } = {}) {
  const [list, setList] = useState({ items: [], loaded: false, error: null })
  const [conv, setConv] = useState({ id: null, turns: [], loading: false, pending: null, error: null })
  const gen = useRef(0)
  const sending = useRef(false)
  const timers = useRef(new Map())
  const idRef = useRef(null)
  idRef.current = conv.id

  const stopPolling = useCallback(() => {
    for (const t of timers.current.values()) clearTimeout(t)
    timers.current.clear()
  }, [])
  useEffect(() => stopPolling, [stopPolling])

  const refreshList = useCallback(async () => {
    try {
      const items = await client.conversations()
      setList({ items: Array.isArray(items) ? items : [], loaded: true, error: null })
    } catch (e) {
      setList((s) => ({ ...s, loaded: true, error: e.message }))
    }
  }, [client])
  useEffect(() => { refreshList() }, [refreshList])

  const poll = useCallback((turn, g, delay = pollStart) => {
    if (timers.current.has(turn.request_id)) return
    const tick = async () => {
      timers.current.delete(turn.request_id)
      if (g !== gen.current) return
      let next = null
      try {
        next = await client.request(turn.request_id)
      } catch {
        next = null
      }
      if (g !== gen.current) return
      if (next) setConv((s) => ({ ...s, turns: mergeTurns(s.turns, [next]) }))
      if (!next || next.status === QUEUED) {
        timers.current.set(turn.request_id, setTimeout(tick, Math.min(pollMax, delay * 1.5)))
        delay = Math.min(pollMax, delay * 1.5)
      } else {
        refreshList()
      }
    }
    timers.current.set(turn.request_id, setTimeout(tick, delay))
  }, [client, pollStart, pollMax, refreshList])

  const open = useCallback(async (id) => {
    const g = ++gen.current
    stopPolling()
    sending.current = false
    setConv({ id, turns: [], loading: !!id, pending: null, error: null })
    if (!id) return
    try {
      const c = await client.conversation(id)
      if (g !== gen.current) return
      const turns = mergeTurns([], c.turns ?? [])
      setConv((s) => ({ ...s, turns, loading: false }))
      for (const t of turns) if (t.status === QUEUED) poll(t, g)
    } catch (e) {
      if (g !== gen.current) return
      setConv((s) => ({ ...s, loading: false, error: { kind: 'load', message: e.message } }))
    }
  }, [client, poll, stopPolling])

  const startNew = useCallback(() => open(null), [open])

  const send = useCallback(async (text) => {
    const question = (text ?? '').trim()
    if (!question || sending.current) return false
    sending.current = true
    const g = gen.current
    setConv((s) => ({ ...s, pending: { text: question }, error: null }))
    try {
      const r = await client.ask(question, idRef.current)
      if (g !== gen.current) return false
      setConv((s) => ({ ...s, id: r.conversation_id ?? s.id, turns: mergeTurns(s.turns, [r]), pending: null }))
      if (r.status === QUEUED) poll(r, g)
      refreshList()
      return true
    } catch (e) {
      if (g !== gen.current) return false

      setConv((s) => ({ ...s, pending: null, error: { kind: 'send', message: e.message, text: question } }))
      return false
    } finally {
      if (g === gen.current) sending.current = false
    }
  }, [client, poll, refreshList])

  const retry = useCallback(() => {
    const err = conv.error
    if (err?.kind === 'send') return send(err.text)
    if (err?.kind === 'load') return open(conv.id)
    return undefined
  }, [conv.error, conv.id, send, open])

  const remove = useCallback(async (id) => {
    await client.deleteConversation(id)
    if (idRef.current === id) await open(null)
    await refreshList()
  }, [client, open, refreshList])

  const removeAll = useCallback(async () => {
    const r = await client.clearConversations()
    await open(null)
    await refreshList()
    return r
  }, [client, open, refreshList])

  const dismissError = useCallback(() => setConv((s) => ({ ...s, error: null })), [])

  return { list, ...conv, open, startNew, send, retry, remove, removeAll, refreshList, dismissError }
}