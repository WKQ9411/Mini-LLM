import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const backendHost = process.env.VITE_BACKEND_HOST || '127.0.0.1'
const backendPort = Number(process.env.VITE_BACKEND_PORT || '12345')
const backendTarget = `http://${backendHost}:${backendPort}`

export default defineConfig({
  plugins: [vue()],
  build: {
    chunkSizeWarningLimit: 2000,
  },
  server: {
    proxy: {
      '/api': backendTarget,
      '/ws': { target: backendTarget, ws: true, timeout: 0 },
    }
  }
})
