// Vite build: workspace and welcome pages into backend/static/workspace; dev server proxies to the local backend.
import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/workspace/',
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, '../backend/static/workspace'),
    emptyOutDir: true,
    rollupOptions: { input: { index: resolve(__dirname, 'index.html'), welcome: resolve(__dirname, 'welcome.html') } },
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8765', '/welcome/extension.zip': 'http://127.0.0.1:8765' },
  },
  test: { environment: 'jsdom', include: ['src/**/*.test.{js,jsx}'] },
})