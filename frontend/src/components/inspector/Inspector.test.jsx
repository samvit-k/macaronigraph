// Inspector views for Reels, concepts, mentions, assertions and citations.
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { emptyModel, mergeOverview } from '../../graph/toElements.js'
import Inspector, { Excerpt } from './Inspector.jsx'

const model = mergeOverview(emptyModel(), {
  reels: [{ reel_id: 'R1', owner: 'alice', source_url: 'https://www.instagram.com/reel/R1/' }],
  concepts: [{ key: 'cuda', name: 'CUDA', type: 'technology', n_reels: 2 }],
  mentions: [{ reel_id: 'R1', concept_key: 'cuda', passages: 1 }], assertions: [], totals: {},
})
const graph = {
  fetchDetail: vi.fn(async () => ({ kind: 'reel', reel: { reel_id: 'R1', owner: 'alice' },
    concepts: [{ key: 'cuda', name: 'CUDA', type: 'technology', n_reels: 2, passages: 1,
      evidence: [{ chunk_id: 'R1:ocr:001', content_type: 'ocr', text: 'CUDA ON SCREEN' }] }], assertions: [] })),
  expand: vi.fn(), collapse: vi.fn(),
}

afterEach(cleanup)

describe('Inspector', () => {
  it('labels on-screen text as OCR that may be misread', () => {
    render(<Excerpt ev={{ content_type: 'ocr', text: 'GPU' }} />)
    expect(screen.getByText('On-Screen Text')).toBeTruthy()
    expect(screen.getByText(/OCR may be misread/)).toBeTruthy()
  })

  it('shows a cited source with its Reel link and the citation caveat', () => {
    const onSelect = vi.fn()
    render(<Inspector selection={{ kind: 'source', sid: 'S2', citation: { reel_id: 'R1', owner: 'alice', content_type: 'transcript', start: 33,
      source_url: 'https://www.instagram.com/reel/R1/' }, source: { text: 'Here the toolkit takes over the heavy math.' } }}
      model={model} graph={graph} onSelect={onSelect} onClose={() => {}} onAsk={() => {}} />)
    expect(screen.getByText('Here the toolkit takes over the heavy math.')).toBeTruthy()
    expect(screen.getByText(/does not guarantee/)).toBeTruthy()
    expect(screen.getByRole('link', { name: /Open on Instagram/ }).getAttribute('href')).toBe('https://www.instagram.com/reel/R1/')
    fireEvent.click(screen.getByRole('button', { name: 'Show Reel in Graph' }))
    expect(onSelect).toHaveBeenCalledWith({ kind: 'reel', id: 'R1', focus: true })
  })

  it('describes a mention as not a relationship and lists its evidence', async () => {
    render(<Inspector selection={{ kind: 'mention', reel_id: 'R1', concept_key: 'cuda' }} model={model} graph={graph}
      onSelect={() => {}} onClose={() => {}} onAsk={() => {}} />)
    expect(screen.getByText(/A mention is not a relationship/)).toBeTruthy()
    expect(await screen.findByText('CUDA ON SCREEN')).toBeTruthy()
  })
})