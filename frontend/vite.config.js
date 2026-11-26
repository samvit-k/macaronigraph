// Vite build: workspace and welcome pages into backend/static/workspace; `--mode pages` builds the hosted welcome page.
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const PAGES = {
  base: '/macaronigraph/',
  outDir: resolve(__dirname, 'dist-pages'),
  input: { welcome: resolve(__dirname, 'welcome.html') },
}
const LOCAL = {
  base: '/workspace/',
  outDir: resolve(__dirname, '../backend/static/workspace'),
  input: { index: resolve(__dirname, 'index.html'), welcome: resolve(__dirname, 'welcome.html') },
}

// Logo links carry a content fingerprint so browsers drop a cached old logo or tab icon when the file changes.
const LOGO = `logo.svg?v=${createHash('sha256').update(readFileSync(resolve(__dirname, 'public/logo.svg'))).digest('hex').slice(0, 10)}`
const logoLinks = { name: 'logo-fingerprint', transformIndexHtml: (html) => html.replaceAll('logo.svg"', `${LOGO}"`) }

export default defineConfig(({ mode }) => {
  const target = mode === 'pages' ? PAGES : LOCAL
  return {
    base: target.base,
    plugins: [react(), logoLinks],
    define: { __LOGO__: JSON.stringify(LOGO) },
    build: { outDir: target.outDir, emptyOutDir: true, rollupOptions: { input: target.input } },
    server: {
      host: '127.0.0.1',
      port: 5173,
      proxy: { '/api': 'http://127.0.0.1:8765', '/welcome/extension.zip': 'http://127.0.0.1:8765' },
    },
    test: { environment: 'jsdom', include: ['src/**/*.test.{js,jsx}'] },
  }
})