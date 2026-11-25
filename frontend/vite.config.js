import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Two pages: the workspace (index.html, served at /workspace/) and the local welcome page (welcome.html, served at
// /welcome). The build goes straight into the backend's static folder, so the backend serves it without Node.
// During `npm run dev`, /api and /welcome/extension.zip are proxied to the local backend on 127.0.0.1:8765.
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