import {defineConfig} from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Development server: requests to /api/... are forwarded to the FastAPI backend on port 8000,
  // with the "/api" prefix removed (e.g. /api/health -> http://127.0.0.1:8000/health).
  // This avoids cross-origin (CORS) problems while developing.
  server: {
    proxy: {'/api': {
      target: process.env.ROADSENSE_DEV_API || 'http://127.0.0.1:8000',
      rewrite: path => path.replace(/^\/api/, ''),
    }},
  },
})
