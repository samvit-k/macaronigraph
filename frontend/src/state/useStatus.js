import { useEffect, useState } from 'react'
import { api } from '../api/client.js'

/** Summarise /api/status, /api/health and /api/graph/status for the top-bar chip. Pure, so it is unit-tested. */
export function summarizeStatus({ jobs, health, graph, backendDown }) {
  if (backendDown) return { tone: 'bad', text: 'Backend Unreachable', detail: 'Start it with .venv/bin/uvicorn backend.app:app --port 8765' }
  const ig = jobs?.reels?.instagram ?? {}
  const processed = ig.done ?? 0
  const waiting = ig.pending ?? 0
  const capturing = (jobs?.open_sessions ?? []).length > 0
  const backlog = health?.backlog ?? jobs?.backlog ?? 0
  const graphOnline = graph?.neo4j?.online === true
  const parts = [`${processed} Reels processed`]
  if (waiting) parts.push(`${waiting} not processed yet`)
  const activity = capturing ? 'Capture Running' : backlog ? `Processing ${backlog}` : 'Idle'
  return {
    tone: !graphOnline ? 'warn' : capturing || backlog ? 'busy' : 'ok',
    text: `${processed} Reels · ${graphOnline ? 'Graph Online' : 'Graph Offline'} · ${activity}`,
    detail: [parts.join(' · '), graphOnline ? 'Knowledge graph online' : 'Knowledge graph offline: answers use passage search only',
      capturing ? 'A capture session is open; new answers may wait for the model.' : backlog ? `${backlog} captured Reels are being processed.` : 'No capture or processing running.'].join('\n'),
    graphOnline, capturing, backlog, processed, waiting,
  }
}

export function useStatus({ client = api, interval = 15000 } = {}) {
  const [summary, setSummary] = useState(null)
  useEffect(() => {
    let stop = false
    let timer = null
    const tick = async () => {
      const [jobs, health, graph] = await Promise.allSettled([client.status(), client.health(), client.graphStatus()])
      if (stop) return
      setSummary(summarizeStatus({
        jobs: jobs.value, health: health.value, graph: graph.value,
        backendDown: jobs.status === 'rejected' && health.status === 'rejected',
      }))
      timer = setTimeout(tick, interval)
    }
    tick()
    return () => { stop = true; clearTimeout(timer) }
  }, [client, interval])
  return summary
}