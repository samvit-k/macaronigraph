// Right-hand panel for asking the saved Reels (layout from SanD Hacks' AssistantPanel; no uploads — knowledge comes
// only from Reels captured by the extension). Conversation history, new conversation, delete, follow-ups,
// clarifications, queued answers and errors.
import { useEffect, useRef, useState } from 'react'
import ConfirmDialog from '../ConfirmDialog.jsx'
import ChatMessage from './ChatMessage.jsx'

const SUGGESTIONS = [
  'Which Reels mention Python?',
  'What do my Reels say about CUDA?',
  'How many Reels mention Claude?',
]

export function conversationLabel(c) {
  const when = c.updated_at ? new Date(c.updated_at) : null
  const date = when && !Number.isNaN(when.getTime()) ? when.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : ''
  return { title: c.title || 'Untitled Conversation', meta: `${c.turns ?? 0} question${c.turns === 1 ? '' : 's'}${date ? ` · ${date}` : ''}` }
}

function History({ chat, onClose, onClearAll }) {
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)
  const del = async (e, id) => {
    e.stopPropagation()
    setBusy(id)
    setError(null)
    try { await chat.remove(id) } catch (err) { setError(err.message) } finally { setBusy(null) }
  }
  return (
    <div className="history-sheet" role="dialog" aria-label="Conversation history">
      <div className="history-head">
        <span className="pop-title">Conversations</span>
        <button type="button" className="link-btn danger" onClick={onClearAll} disabled={chat.list.items.length === 0}>Clear All</button>
      </div>
      {(error || chat.list.error) && <div className="warn-card">{error ?? `Could not load: ${chat.list.error}`}</div>}
      {chat.list.loaded && chat.list.items.length === 0 && <div className="insp-empty">No conversations yet.</div>}
      <ul className="history-list">
        {chat.list.items.map((c) => {
          const l = conversationLabel(c)
          return (
            <li key={c.id} data-id={c.id} className={`history-item${c.id === chat.id ? ' on' : ''}`}>
              <button type="button" className="history-row" onClick={() => { chat.open(c.id); onClose() }}>
                <span className="h-title">{l.title}</span><span className="h-meta">{l.meta}</span>
              </button>
              <button type="button" className="history-del" onClick={(e) => del(e, c.id)} disabled={busy === c.id}
                aria-label={`Delete conversation: ${l.title}`} title="Delete this conversation">
                <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7" /></svg>
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default function AssistantPanel({ open, onClose, chat, draft, setDraft, focusTick, onCite, onHighlight, activeTurn, graphReady = true }) {
  const [showHistory, setShowHistory] = useState(false)
  // Exchanges the user minimized (view state only), keyed by turn id.
  const [collapsed, setCollapsed] = useState(() => new Set())
  const toggle = (id) => setCollapsed((prev) => { const n = new Set(prev); if (n.has(id)) n.delete(id); else n.add(id); return n })
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [confirmClear, setConfirmClear] = useState(false)
  const listRef = useRef(null)
  const inputRef = useRef(null)
  const current = chat.list.items.find((c) => c.id === chat.id)

  useEffect(() => {
    const el = listRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [chat.turns.length, chat.pending, chat.error, chat.id])
  useEffect(() => { if (focusTick) inputRef.current?.focus() }, [focusTick])

  const submit = async () => {
    const text = draft.trim()
    if (!text || chat.pending) return
    setDraft('')
    const ok = await chat.send(text)
    if (!ok) setDraft((d) => d || text)
  }
  const onClarify = (question, name) => {
    setDraft(`${question} (about ${name})`)
    inputRef.current?.focus()
  }

  return (
    <aside id="assistant-panel" className="assistant" hidden={!open} aria-label="Ask your Reels">
      <div className="as-head">
        <span className="as-title">{current ? current.title || 'Conversation' : 'New Conversation'}</span>
        <div className="as-actions">
          <div className="history-wrap">
            <button type="button" className="btn btn-sm" onClick={() => setShowHistory((v) => !v)} aria-expanded={showHistory}>History</button>
            
          </div>
          <button type="button" className="btn btn-sm" onClick={() => { chat.startNew(); setDraft(''); inputRef.current?.focus() }}
            disabled={!chat.id && chat.turns.length === 0}>New</button>
          {chat.id && (
            <button type="button" className="icon-btn" onClick={() => setConfirmDelete(true)} aria-label="Delete this conversation" title="Delete this conversation">
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 4.5h9M6.5 4.5V3h3v1.5M5 4.5l.6 8h4.8l.6-8" /></svg>
            </button>
          )}
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close panel" title="Close">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3.5 10.5 8 6 12.5" /></svg>
          </button>
        </div>
      </div>
      {showHistory && (
        <>
          <div className="history-backdrop" onClick={() => setShowHistory(false)} />
          <History chat={chat} onClose={() => setShowHistory(false)} onClearAll={() => setConfirmClear(true)} />
        </>
      )}
      <div className="chat-list" ref={listRef} aria-live="polite">
        {chat.loading && <div className="chat-thinking"><span className="spinner" aria-hidden="true" /> Loading conversation…</div>}
        {!chat.loading && chat.turns.length === 0 && !chat.pending && (
          <div className="chat-empty">
            <p>Ask about your processed Reels. Answers are written by a local model from passages of your Reels and cite
              them; listing and counting questions are answered by exact search.</p>
            {SUGGESTIONS.map((s) => (
              <button key={s} type="button" className="chip-btn" onClick={() => { setDraft(s); inputRef.current?.focus() }}>{s}</button>
            ))}
          </div>
        )}
        {chat.turns.map((t) => (
          <ChatMessage key={t.turn_id} turn={t} onCite={onCite} onHighlight={onHighlight} onClarify={onClarify}
            active={activeTurn === t.turn_id} graphReady={graphReady}
            collapsed={collapsed.has(t.turn_id)} onToggle={() => toggle(t.turn_id)} />
        ))}
        {chat.pending && (
          <div className="exchange pending">
            <div className="chat-row user"><div className="chat-bubble">{chat.pending.text}</div></div>
            <div className="chat-thinking" role="status"><span className="spinner" aria-hidden="true" /> Searching your Reels and writing an answer…</div>
          </div>
        )}
        {chat.error && (
          <div className="status-card bad" role="alert">
            {chat.error.kind === 'send' ? 'Your question was not answered: ' : 'Could not load the conversation: '}{chat.error.message}{' '}
            <button type="button" className="link-btn" onClick={() => chat.retry()}>Retry</button>{' '}
            <button type="button" className="link-btn muted" onClick={chat.dismissError}>Dismiss</button>
          </div>
        )}
      </div>
      <div className="chat-compose">
        <div className="compose-row">
          <textarea ref={inputRef} rows={2} value={draft} maxLength={2000} placeholder="Ask your saved Reels…"
            aria-label="Ask your saved Reels" onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() } }} />
          <button type="button" className="btn send-btn" disabled={!draft.trim() || !!chat.pending} onClick={submit} aria-label="Send">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 8h10M8.5 4l4 4-4 4" /></svg>
          </button>
        </div>
        <p className="compose-note">Citations do not guarantee a generated answer is correct.</p>
      </div>
      {confirmClear && (
        <ConfirmDialog title="Clear All Conversations?" confirmLabel="Clear All" busyLabel="Clearing…"
          onCancel={() => setConfirmClear(false)}
          onConfirm={async () => { await chat.removeAll(); setConfirmClear(false); setShowHistory(false) }}>
          <p>All {chat.list.items.length} conversation{chat.list.items.length === 1 ? '' : 's'}, including the one open now, are
            removed from this computer. Your Reels, the search index and the knowledge graph are not affected.</p>
        </ConfirmDialog>
      )}
      {confirmDelete && (
        <ConfirmDialog title="Delete This Conversation?" confirmLabel="Delete" busyLabel="Deleting…"
          onCancel={() => setConfirmDelete(false)}
          onConfirm={async () => { await chat.remove(chat.id); setConfirmDelete(false) }}>
          <p>“{current?.title || 'This conversation'}” and its {chat.turns.length} question{chat.turns.length === 1 ? '' : 's'} are
            removed from this computer. Your Reels, the search index and the knowledge graph are not affected.</p>
        </ConfirmDialog>
      )}
    </aside>
  )
}
