import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client.js'

// Conversation state, adapted from SanD Hacks' useChat to the MacaroniGraph conversational API.
//
// Async safety (same rules as SanD Hacks):
//  - `gen` increments whenever the open conversation changes; every request captures it, and a response is applied
//    only if the generation is unchanged — a late answer can never land in another conversation.
//  - a synchronous ref guards `send`, so rapid Enter presses never start overlapping requests;
//  - turns are merged by server turn id, so a slow reload never drops or duplicates a newer turn.
// Queued answers (capture/processing is using the model) are polled with backoff until they leave "queued", also
// after reopening the conversation.

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
  const timers = useRef(new Map()) // turn_id -> timeout
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
        next = null // transient: keep polling
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
      // Nothing was stored for a failed request, so the same question can be retried as-is.
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

  /** Delete every conversation, including the open one (the user confirmed it). */
  const removeAll = useCallback(async () => {
    const r = await client.clearConversations()
    await open(null)
    await refreshList()
    return r
  }, [client, open, refreshList])

  const dismissError = useCallback(() => setConv((s) => ({ ...s, error: null })), [])

  return { list, ...conv, open, startNew, send, retry, remove, removeAll, refreshList, dismissError }
}