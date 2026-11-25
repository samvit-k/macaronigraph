// Thin client for the local MacaroniGraph backend (same origin: the backend serves this app at /workspace/).
// Errors carry the HTTP status so the UI can tell "graph offline" (503) from other failures.

export class ApiError extends Error {
  constructor(message, status = 0, detail = null) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function request(path, { method = 'GET', body, timeout = 30000, signal } = {}) {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeout)
  const onAbort = () => ctrl.abort()
  signal?.addEventListener('abort', onAbort)
  try {
    const res = await fetch(path, {
      method, signal: ctrl.signal,
      headers: body ? { 'content-type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    })
    const text = await res.text()
    let data = null
    try { data = text ? JSON.parse(text) : null } catch { data = null }
    if (!res.ok) {
      const detail = data?.detail ?? text
      throw new ApiError(typeof detail === 'string' && detail ? detail : `Request failed (${res.status})`, res.status, detail)
    }
    return data
  } catch (e) {
    if (e instanceof ApiError) throw e
    if (e?.name === 'AbortError') {
      throw new ApiError(signal?.aborted ? 'Cancelled' : 'The backend did not answer in time.', 0)
    }
    throw new ApiError('Cannot reach the local backend. Is it running on port 8765?', 0)
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}

const q = (params) => new URLSearchParams(Object.entries(params).filter(([, v]) => v != null && v !== '')).toString()

export const api = {
  health: () => request('/api/health', { timeout: 8000 }),
  status: () => request('/api/status', { timeout: 8000 }),
  indexStatus: () => request('/api/index/status', { timeout: 8000 }),
  graphStatus: () => request('/api/graph/status', { timeout: 8000 }),
  overview: (opts = {}) => request(`/api/graph/overview?${q(opts)}`),
  neighborhood: (kind, id) => request(`/api/graph/neighborhood?${q({ kind, id })}`),
  search: (text, signal) => request(`/api/graph/search?${q({ q: text, limit: 12 })}`, { signal, timeout: 10000 }),
  conversations: () => request('/api/conversations'),
  conversation: (id) => request(`/api/conversations/${encodeURIComponent(id)}`),
  clearConversations: () => request('/api/conversations', { method: 'DELETE' }),
  deleteConversation: (id) => request(`/api/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  // Generation can wait up to RAG_HEAVY_WAIT_S (90 s) for the model before the server returns "queued".
  ask: (question, conversationId) => request('/api/ask', { method: 'POST', timeout: 240000,
    body: { question, ...(conversationId ? { conversation_id: conversationId } : {}) } }),
  request: (id) => request(`/api/ask/${id}`, { timeout: 15000 }),
}
