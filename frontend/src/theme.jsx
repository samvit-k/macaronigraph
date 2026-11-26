// Light/dark theme: follows the system until the user picks one, remembers the pick, and sets <html data-theme>.
import { createContext, useCallback, useContext, useEffect, useState } from 'react'

export const THEME_KEY = 'macaronigraph.theme'
const DARK_QUERY = '(prefers-color-scheme: dark)'

export function storedTheme() {
  try {
    const t = window.localStorage.getItem(THEME_KEY)
    return t === 'light' || t === 'dark' ? t : null
  } catch {
    return null
  }
}

export function systemTheme() {
  return typeof window !== 'undefined' && window.matchMedia?.(DARK_QUERY).matches ? 'dark' : 'light'
}

export function applyTheme(theme) {
  const root = document.documentElement
  root.dataset.theme = theme
  root.style.colorScheme = theme
}

export function useTheme() {
  const [theme, setTheme] = useState(() => storedTheme() ?? systemTheme())
  useEffect(() => { applyTheme(theme) }, [theme])
  useEffect(() => {
    const mq = window.matchMedia?.(DARK_QUERY)
    if (!mq) return undefined
    const follow = (e) => { if (!storedTheme()) setTheme(e.matches ? 'dark' : 'light') }
    mq.addEventListener?.('change', follow)
    return () => mq.removeEventListener?.('change', follow)
  }, [])
  const toggle = useCallback(() => setTheme((t) => {
    const next = t === 'dark' ? 'light' : 'dark'
    try { window.localStorage.setItem(THEME_KEY, next) } catch { /* storage unavailable: the choice lasts for this page only */ }
    return next
  }), [])
  return [theme, toggle]
}

export const ThemeContext = createContext('light')
export const useThemeName = () => useContext(ThemeContext)

export function ThemeToggle({ theme, onToggle, className = 'icon-btn theme-btn' }) {
  const dark = theme === 'dark'
  const label = dark ? 'Switch to Light Mode' : 'Switch to Dark Mode'
  return (
    <button type="button" className={className} onClick={onToggle} aria-label={label} title={label} data-testid="theme-toggle">
      {dark
        ? <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="3" /><path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1" /></svg>
        : <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8z" /></svg>}
    </button>
  )
}