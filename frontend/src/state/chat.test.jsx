// Conversation state and chat rendering: sending, polling, history and every answer status.
import { act, cleanup, fireEvent, render, renderHook, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ChatMessage, { isListing, listingScope } from '../components/assistant/ChatMessage.jsx'
import { mergeTurns, useConversation } from './useConversation.js'
import { summarizeStatus } from './useStatus.js'

const turn = (over = {}) => ({
  turn_id: 1, request_id: 1, conversation_id: 'c1', question: 'What does the cooking Reel say?', status: 'ok', intent: 'question',
  answer: { status: 'ok', text: 'x', sentences: [{ text: 'Toast the spices first.', citations: ['S1'], kind: 'source' }], missing: '' },
  citations: [{ sid: 'S1', reel_id: 'R1', owner: 'demo.chef', content_type: 'transcript', source_url: 'https://www.instagram.com/reel/R1/' }],
  sources: [{ sid: 'S1', reel_id: 'R1', chunk_id: 'R1:transcript:003', text: 'always toast the whole spices before grinding' }],
  cited_concepts: [], rewrite: { method: 'none' }, fallback: [], ...over,
})

function fakeClient(over = {}) {
  return {
    conversations: vi.fn(async () => []),
    conversation: vi.fn(async () => ({ id: 'c1', turns: [] })),
    ask: vi.fn(async () => turn()),
    request: vi.fn(async () => turn()),
    deleteConversation: vi.fn(async () => ({})),
    ...over,
  }
}

afterEach(() => { cleanup(); vi.useRealTimers() })

describe('useConversation', () => {
  it('sends, stores the conversation id and refreshes the history', async () => {
    const client = fakeClient()
    const { result } = renderHook(() => useConversation({ client }))
    await act(async () => { await result.current.send('  What does the cooking Reel say? ') })
    expect(client.ask).toHaveBeenCalledWith('What does the cooking Reel say?', null)
    expect(result.current.id).toBe('c1')
    expect(result.current.turns).toHaveLength(1)
    expect(client.conversations).toHaveBeenCalledTimes(2)
    await act(async () => { await result.current.send('follow-up') })
    expect(client.ask).toHaveBeenLastCalledWith('follow-up', 'c1')
  })

  it('ignores a second send while one is in flight', async () => {
    let resolve
    const client = fakeClient({ ask: vi.fn(() => new Promise((r) => { resolve = r })) })
    const { result } = renderHook(() => useConversation({ client }))
    let first
    act(() => { first = result.current.send('a') })
    await act(async () => { expect(await result.current.send('b')).toBe(false) })
    await act(async () => { resolve(turn()); await first })
    expect(client.ask).toHaveBeenCalledTimes(1)
  })

  it('polls a queued answer until it is ready', async () => {
    vi.useFakeTimers()
    const queued = turn({ status: 'queued', answer: { status: 'deferred', text: null, sentences: [] } })
    const client = fakeClient({ ask: vi.fn(async () => queued),
      request: vi.fn().mockResolvedValueOnce(queued).mockResolvedValueOnce(turn()) })
    const { result } = renderHook(() => useConversation({ client, pollStart: 100, pollMax: 200 }))
    await act(async () => { await result.current.send('q') })
    expect(result.current.turns[0].status).toBe('queued')
    await act(async () => { await vi.advanceTimersByTimeAsync(100) })
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    expect(client.request).toHaveBeenCalledTimes(2)
    expect(result.current.turns[0].status).toBe('ok')
  })

  it('drops a late answer after switching conversations', async () => {
    let resolve
    const client = fakeClient({ ask: vi.fn(() => new Promise((r) => { resolve = r })),
      conversation: vi.fn(async () => ({ id: 'c2', turns: [turn({ turn_id: 9, conversation_id: 'c2' })] })) })
    const { result } = renderHook(() => useConversation({ client }))
    let sending
    act(() => { sending = result.current.send('slow') })
    await act(async () => { await result.current.open('c2') })
    await act(async () => { resolve(turn({ turn_id: 5 })); await sending })
    expect(result.current.id).toBe('c2')
    expect(result.current.turns.map((t) => t.turn_id)).toEqual([9])
  })

  it('keeps the question for retry when sending fails', async () => {
    const client = fakeClient({ ask: vi.fn().mockRejectedValueOnce(new Error('Cannot reach the local backend.')).mockResolvedValueOnce(turn()) })
    const { result } = renderHook(() => useConversation({ client }))
    await act(async () => { await result.current.send('q') })
    expect(result.current.error).toMatchObject({ kind: 'send', text: 'q' })
    await act(async () => { await result.current.retry() })
    expect(result.current.error).toBeNull()
    expect(result.current.turns).toHaveLength(1)
  })

  it('deletes one conversation, or all of them including the open one', async () => {
    const client = fakeClient({ conversations: vi.fn(async () => [{ id: 'c1' }, { id: 'c2' }]), clearConversations: vi.fn(async () => ({ deleted: 2 })) })
    const { result } = renderHook(() => useConversation({ client }))
    await act(async () => { await result.current.send('q') })
    expect(result.current.id).toBe('c1')
    await act(async () => { await result.current.remove('c2') })
    expect(client.deleteConversation).toHaveBeenCalledWith('c2')
    expect(result.current.id).toBe('c1')
    await act(async () => { await result.current.removeAll() })
    expect(client.clearConversations).toHaveBeenCalled()
    expect(result.current.id).toBeNull()
    expect(result.current.turns).toEqual([])
  })

  it('merges turns by id in order', () => {
    expect(mergeTurns([{ turn_id: 2, v: 1 }], [{ turn_id: 1 }, { turn_id: 2, v: 2 }])).toEqual([{ turn_id: 1 }, { turn_id: 2, v: 2 }])
  })
})

describe('ChatMessage', () => {
  const noop = () => {}
  it('labels generated answers and renders clickable citations', () => {
    const onCite = vi.fn()
    render(<ChatMessage turn={turn()} onCite={onCite} onHighlight={noop} onClarify={noop} />)
    expect(screen.getByText(/Generated Answer · Local Model/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Cited source S1' }))
    expect(onCite).toHaveBeenCalledWith(expect.objectContaining({ turn_id: 1 }), 'S1')
  })

  it('shows the partial-answer note', () => {
    render(<ChatMessage turn={turn({ answer: { ...turn().answer, missing: 'Part of the question is not covered by the retrieved sources.' } })}
      onCite={noop} onHighlight={noop} onClarify={noop} />)
    expect(screen.getByText(/not covered/)).toBeTruthy()
  })

  it('renders listings as exact search with evidence types and the scope note', () => {
    const t = turn({ intent: 'listing_list', answer: { status: 'ok', missing: '',
      text: "@demo.chef's Reel mentions “Saffron” in its transcript: “…” [S1]\n\nExact search of transcripts, captions and on-screen text in 12 processed Reels.",
      sentences: [{ text: '@demo.chef\'s Reel mentions “Saffron” in its transcript: “…”', citations: ['S1'] }] } })
    expect(isListing(t)).toBe(true)
    expect(listingScope(t)).toMatch(/^Exact search/)
    render(<ChatMessage turn={t} onCite={noop} onHighlight={noop} onClarify={noop} />)
    expect(screen.getByText(/Exact Search · No Model/)).toBeTruthy()
    expect(screen.getByText('Transcript')).toBeTruthy()
  })

  const listed = (n, extra = {}) => turn({ intent: 'listing_list', ...extra, answer: { status: 'ok', missing: '',
    text: 'rows\n\nExact search of transcripts, captions and on-screen text in 900 processed Reels.',
    sentences: Array.from({ length: n }, (_, i) => ({ text: `@r${i}'s Reel mentions “Python” in its transcript: “…”`, citations: [`S${i + 1}`] })),
    ...(extra.answer ?? {}) },
    citations: Array.from({ length: n }, (_, i) => ({ sid: `S${i + 1}`, reel_id: `R${i}`, content_type: 'transcript', owner: `r${i}` })) })

  it('puts the generated summary above a minimized evidence list that expands to every match', () => {
    const onCite = vi.fn()
    const t = listed(8, { answer: { summary: { status: 'ok', based_on: 8, of: 8,
      sentences: [{ text: 'Several Reels recommend Python as a first language.', citations: ['S2', 'S5'] }] } } })
    render(<ChatMessage turn={t} onCite={onCite} onHighlight={noop} onClarify={noop} />)
    const summary = screen.getByText(/Generated Answer · Local Model/)
    const evidence = screen.getByText(/Evidence · Exact Search · No Model/)
    expect(summary.compareDocumentPosition(evidence) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(screen.getByText(/recommend Python as a first language/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Cited source S5' }))
    expect(onCite).toHaveBeenCalledWith(t, 'S5')
    expect(screen.getAllByRole('listitem')).toHaveLength(3)
    expect(screen.queryByText(/^Exact search of/)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Show All 8 Matches' }))
    expect(screen.getAllByRole('listitem')).toHaveLength(8)
    expect(screen.getByText(/^Exact search of/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Minimize Evidence' }))
    expect(screen.getAllByRole('listitem')).toHaveLength(3)
  })

  it('says when the summary covers only the first matches', () => {
    render(<ChatMessage turn={listed(4, { answer: { summary: { status: 'ok', based_on: 2, of: 4,
      sentences: [{ text: 'They say Python is easy.', citations: ['S1'] }] } } })} onCite={noop} onHighlight={noop} onClarify={noop} />)
    expect(screen.getByText(/first 2 of the 4 matching Reels/)).toBeTruthy()
  })

  it.each([
    ['queued', {}, /Waiting for the local model to summarize/],
    ['expired', {}, /backend restarted/],
    ['unavailable', {}, /Summary Unavailable/],
    ['ok', { status: 'error', error: 'answer format' }, /Summary Failed/],
    ['ok', { status: 'insufficient_evidence', sentences: [] }, /could not summarize/],
  ])('keeps the evidence visible when the summary is %s', (status, summary, re) => {
    render(<ChatMessage turn={listed(5, { status, answer: { summary: status === 'ok' ? summary : { status: 'pending' } } })}
      onCite={noop} onHighlight={noop} onClarify={noop} />)
    expect(screen.getByText(re)).toBeTruthy()
    expect(screen.getAllByRole('listitem')).toHaveLength(3)
    expect(screen.getByRole('button', { name: 'Show All 5 Matches' })).toBeTruthy()
  })

  it('shows listings without a summary in full, as before', () => {
    render(<ChatMessage turn={listed(6)} onCite={noop} onHighlight={noop} onClarify={noop} />)
    expect(screen.getAllByRole('listitem')).toHaveLength(6)
    expect(screen.queryByText(/Generated Answer/)).toBeNull()
    expect(screen.queryByRole('button', { name: /Show All/ })).toBeNull()
  })

  it('offers clarification choices that are only put into the composer', () => {
    const onClarify = vi.fn()
    const t = turn({ status: 'clarify', question: 'Which store sells it?', rewrite: { method: 'clarify', clarify: ['Alpha Grinder', 'Beta Grinder'] },
      answer: { status: 'clarify', text: 'Which one do you mean: “Alpha Grinder” or “Beta Grinder”?', sentences: [] } })
    render(<ChatMessage turn={t} onCite={noop} onHighlight={noop} onClarify={onClarify} />)
    fireEvent.click(screen.getByRole('button', { name: 'Beta Grinder' }))
    expect(onClarify).toHaveBeenCalledWith('Which store sells it?', 'Beta Grinder')
  })

  it.each([
    ['queued', /Waiting for the local model/],
    ['insufficient_evidence', /Not Enough Evidence/],
    ['unavailable', /Answer Unavailable/],
    ['error', /Generation Failed/],
    ['expired', /backend restarted/],
  ])('renders the %s state as an app note, not as model output', (status, re) => {
    render(<ChatMessage turn={turn({ status, answer: { status, text: 'x', sentences: [], reason: 'r' } })} onCite={noop} onHighlight={noop} onClarify={noop} />)
    expect(screen.getByText(re)).toBeTruthy()
    expect(screen.queryByText(/Generated Answer/)).toBeNull()
  })

  it('notes when the graph was unavailable for an answer', () => {
    render(<ChatMessage turn={turn({ fallback: ['graph offline'] })} onCite={noop} onHighlight={noop} onClarify={noop} />)
    expect(screen.getByText(/passage search only/)).toBeTruthy()
  })
})

describe('status summary', () => {
  it('reports processed Reels, graph state and activity', () => {
    const s = summarizeStatus({ jobs: { reels: { instagram: { done: 59, pending: 766 } }, open_sessions: [] }, health: { backlog: 0 },
      graph: { neo4j: { online: true } } })
    expect(s.text).toBe('59 Reels · Graph Online · Idle')
    expect(s.tone).toBe('ok')
    const busy = summarizeStatus({ jobs: { reels: { instagram: { done: 1 } }, open_sessions: ['x'] }, health: {}, graph: { neo4j: { online: false } } })
    expect(busy.text).toMatch(/Graph Offline · Capture Running/)
    expect(summarizeStatus({ backendDown: true }).tone).toBe('bad')
  })
})

describe('graph offline', () => {
  it('hides graph actions that cannot work', () => {
    render(<ChatMessage turn={turn()} onCite={() => {}} onHighlight={() => {}} onClarify={() => {}} graphReady={false} />)
    expect(screen.queryByRole('button', { name: 'Show in Graph' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Cited source S1' })).toBeTruthy()
  })
})

describe('minimize', () => {
  it('collapses the answer and toggles to Expand', () => {
    const onToggle = vi.fn()
    const { rerender } = render(<ChatMessage turn={turn()} onCite={() => {}} onHighlight={() => {}} onClarify={() => {}} onToggle={onToggle} />)
    fireEvent.click(screen.getByRole('button', { name: /Minimize/ }))
    expect(onToggle).toHaveBeenCalled()
    rerender(<ChatMessage turn={turn()} onCite={() => {}} onHighlight={() => {}} onClarify={() => {}} onToggle={onToggle} collapsed />)
    expect(screen.queryByText(/Generated Answer/)).toBeNull()
    expect(screen.getByText('Answer minimized')).toBeTruthy()
    expect(screen.getByRole('button', { name: /Expand/ })).toBeTruthy()
  })
})