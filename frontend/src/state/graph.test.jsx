// Reel scope: the picker's options and choices, and the graph hook requesting and remembering the chosen scope.
import { cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ReelScope, { scopeOptions } from '../components/ReelScope.jsx'
import TopBar from '../components/TopBar.jsx'
import { DEFAULT_FILTERS, emptyModel } from '../graph/toElements.js'
import { overviewParams, useGraph } from './useGraph.js'

afterEach(() => { cleanup(); window.localStorage.clear() })

const overview = (n) => ({ reels: [{ reel_id: 'R1' }], concepts: [], mentions: [], assertions: [],
  totals: { reels: n, concepts: 10, bridge: 2, reels_shown: 1, concepts_shown: 0 } })

describe('reel scope picker', () => {
  it('offers the steps below the library size plus all Reels', () => {
    expect(scopeOptions(807)).toEqual([25, 50, 100, 200, 400, 807])
    expect(scopeOptions(60)).toEqual([25, 50, 60])
  })

  it('opens from the concept count and reports the chosen scope', () => {
    const onScope = vi.fn()
    render(<ReelScope label="257/4885 Concepts" scope={null} total={807} onScope={onScope} />)
    fireEvent.click(screen.getByRole('button', { name: '257/4885 Concepts' }))
    expect(screen.getByRole('radio', { name: /Overview/ }).checked).toBe(true)
    fireEvent.click(screen.getByRole('radio', { name: /All 807 Reels/ }))
    expect(onScope).toHaveBeenCalledWith(807)
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('accepts a custom count within the library only', () => {
    const onScope = vi.fn()
    render(<ReelScope label="x" scope={50} total={807} onScope={onScope} />)
    fireEvent.click(screen.getByRole('button', { name: 'x' }))
    expect(screen.getByRole('radio', { name: /50 Recent Reels/ }).checked).toBe(true)
    const input = screen.getByRole('spinbutton', { name: 'Number of Reels' })
    fireEvent.change(input, { target: { value: '900' } })
    expect(screen.getByRole('button', { name: 'Show' }).disabled).toBe(true)
    fireEvent.change(input, { target: { value: '120' } })
    fireEvent.click(screen.getByRole('button', { name: 'Show' }))
    expect(onScope).toHaveBeenCalledWith(120)
  })

  it('is disabled while the graph cannot load', () => {
    render(<ReelScope label="Knowledge Graph Offline" scope={null} total={undefined} onScope={() => {}} disabled />)
    expect(screen.getByRole('button', { name: 'Knowledge Graph Offline' }).disabled).toBe(true)
  })

  it('stays usable after a failed load so a bad choice can be undone', () => {
    const onScope = vi.fn()
    render(<TopBar model={emptyModel()} graphStatus="error" scope={400} onScope={onScope} query=""
      onQuery={() => {}} onPick={() => {}} filters={DEFAULT_FILTERS} onFilters={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: 'Graph Not Loaded' }))
    fireEvent.click(screen.getByRole('radio', { name: /Overview/ }))
    expect(onScope).toHaveBeenCalledWith(null)
  })
})

describe('useGraph scope', () => {
  it('keeps the default overview request unless a scope is chosen', () => {
    expect(overviewParams(null)).toEqual({})
    expect(overviewParams(100)).toEqual({ reel_limit: 100, within: 'true', limit_concepts: 10000 })
  })

  it('reloads with the chosen scope and remembers it', async () => {
    const client = { overview: vi.fn(async () => overview(807)), neighborhood: vi.fn() }
    const { result } = renderHook(() => useGraph({ client }))
    await waitFor(() => expect(result.current.status).toBe('ready'))
    expect(client.overview).toHaveBeenLastCalledWith({})
    result.current.setScope(100)
    await waitFor(() => expect(client.overview).toHaveBeenLastCalledWith(overviewParams(100)))
    expect(result.current.scope).toBe(100)
    expect(window.localStorage.getItem('macaronigraph.reelScope')).toBe('100')
    const again = renderHook(() => useGraph({ client }))
    expect(again.result.current.scope).toBe(100)
    again.result.current.setScope(null)
    await waitFor(() => expect(client.overview).toHaveBeenLastCalledWith({}))
    expect(window.localStorage.getItem('macaronigraph.reelScope')).toBeNull()
  })
})