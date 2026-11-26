// Theme: system default, toggle and remembered choice, and both theme variants of the graph stylesheet.
import { act, cleanup, fireEvent, render, renderHook, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { buildStyle, INK } from './graph/style.js'
import { edgeColor, EDGE_KINDS } from './graph/vocab.js'
import { THEME_KEY, ThemeToggle, useTheme } from './theme.jsx'

const system = (dark) => vi.spyOn(window, 'matchMedia').mockImplementation((q) => ({
  matches: dark && q.includes('dark'), media: q, addEventListener: () => {}, removeEventListener: () => {},
}))

afterEach(() => { cleanup(); vi.restoreAllMocks(); window.localStorage.clear(); delete document.documentElement.dataset.theme })

describe('useTheme', () => {
  it('follows the system until the user chooses', () => {
    system(true)
    const { result } = renderHook(() => useTheme())
    expect(result.current[0]).toBe('dark')
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(window.localStorage.getItem(THEME_KEY)).toBeNull()
  })

  it('toggles, applies and remembers the choice over the system setting', () => {
    system(true)
    const { result } = renderHook(() => useTheme())
    act(() => result.current[1]())
    expect(result.current[0]).toBe('light')
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(window.localStorage.getItem(THEME_KEY)).toBe('light')
    expect(renderHook(() => useTheme()).result.current[0]).toBe('light')
  })
})

describe('ThemeToggle', () => {
  it('names the mode it switches to', () => {
    const onToggle = vi.fn()
    render(<ThemeToggle theme="light" onToggle={onToggle} />)
    fireEvent.click(screen.getByRole('button', { name: 'Switch to Dark Mode' }))
    expect(onToggle).toHaveBeenCalledOnce()
    cleanup()
    render(<ThemeToggle theme="dark" onToggle={onToggle} />)
    expect(screen.getByRole('button', { name: 'Switch to Light Mode' })).toBeTruthy()
  })
})

describe('graph stylesheet', () => {
  const rule = (style, selector) => style.findLast((r) => r.selector === selector).style
  it('uses light ink by default and dark ink in dark mode', () => {
    expect(rule(buildStyle(), 'node').color).toBe(INK.light.label)
    expect(rule(buildStyle('dark'), 'node').color).toBe(INK.dark.label)
    expect(rule(buildStyle('dark'), 'edge.mention')['line-color']).toBe(EDGE_KINDS.mention.dark)
  })

  it('keeps every edge kind visible in both themes', () => {
    for (const k of Object.keys(EDGE_KINDS)) {
      expect(edgeColor(k)).toBe(EDGE_KINDS[k].color)
      expect(edgeColor(k, 'dark')).toBe(EDGE_KINDS[k].dark)
    }
  })
})