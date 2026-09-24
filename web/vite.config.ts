import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/postcss';
import { defineConfig } from 'vite';
import { fileURLToPath, URL } from 'node:url';
import { browserAssetsPlugin } from './scripts/browser-assets.mjs';
export default defineConfig({
  plugins: [browserAssetsPlugin(), react()],
  resolve: { alias: { '@': fileURLToPath(new URL('.', import.meta.url)) } },
  css: { postcss: { plugins: [tailwindcss()] } },
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
  optimizeDeps: { exclude: ['@mujoco/mujoco', 'onnxruntime-web'] },
  server: { host: '0.0.0.0', port: 3000, strictPort: true },
});
