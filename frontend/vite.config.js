import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Dev proxy target: where the backend API server actually runs locally.
// Override with BACKEND_URL if it's not on the default port.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendTarget = env.BACKEND_URL || 'http://localhost:8000'

  return {
    plugins: [react()],
    server: {
      host: true,
      port: 5173,
      proxy: {
        // Frontend calls relative `/api/...` paths (see src/api/client.js);
        // this forwards them to the backend so local dev needs no CORS
        // configuration at all. Matches VITE_API_BASE_URL left unset.
        '/api': {
          target: backendTarget,
          changeOrigin: true
        }
      }
    }
  }
})
