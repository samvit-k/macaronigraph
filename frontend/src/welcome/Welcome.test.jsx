// Welcome page in its two modes: served by the local backend, and hosted on GitHub Pages.
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import Welcome, { RELEASE_ZIP, REPO_URL } from './Welcome.jsx'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('Welcome', () => {
  it('local: downloads from the backend, links the workspace and checks backend health', () => {
    const fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => ({ ok: true }) }))
    vi.stubGlobal('fetch', fetch)
    render(<Welcome />)
    expect(screen.getByText('Download Extension (.zip)').getAttribute('href')).toBe('/welcome/extension.zip')
    expect(screen.getAllByText('Open Workspace')[0].getAttribute('href')).toBe('/workspace/')
    expect(fetch).toHaveBeenCalledWith('/api/health')
  })

  it('hosted: downloads the latest release, links the repository and never calls a backend', () => {
    const fetch = vi.fn()
    vi.stubGlobal('fetch', fetch)
    render(<Welcome hosted />)
    expect(screen.getByText('Download Extension (.zip)').getAttribute('href')).toBe(RELEASE_ZIP)
    expect(screen.getAllByText('View on GitHub')[0].getAttribute('href')).toBe(REPO_URL)
    expect(screen.queryByText('Open Workspace')).toBeNull()
    expect(screen.getByText(/Clone the repository/)).toBeTruthy()
    expect(fetch).not.toHaveBeenCalled()
  })
})