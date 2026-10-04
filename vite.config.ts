import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig, type Connect } from 'vite';

/**
 * Build configuration for a **static** site.
 *
 * There is deliberately no API middleware here any more. The previous build
 * mounted Gemini routes on the dev server and shipped an Express server to
 * serve the same routes in production, which meant the deployed app needed a
 * running Node process holding an API key. This one produces a directory of
 * files. Nothing to run, nothing to pay for, nothing to leak.
 *
 * Everything the app does — instrument detection, presence and task
 * classification, Grad-CAM, the grounded assistant — happens in the visitor's
 * browser. The one optional network call goes from their browser straight to
 * Google with a key they supplied themselves (see `src/services/byokGemini.ts`).
 */

/**
 * Ask the browser for cross-origin isolation.
 *
 * onnxruntime-web can only use WASM threads when `SharedArrayBuffer` exists,
 * and that requires the document to be cross-origin isolated. Without it the
 * detector runs single-threaded — measured at ~98 ms per frame, four to five
 * times slower than the same graph on four threads.
 *
 * `credentialless` rather than `require-corp`: the stricter value blocks every
 * cross-origin subresource that does not send CORP headers, which includes the
 * Google Fonts stylesheet this page loads. Credentialless fetches them without
 * credentials instead, which is all they need. It also leaves `fetch()` to the
 * Gemini endpoint working, since that is a CORS request carrying no cookies.
 *
 * These headers must also be set by whatever serves the built files. See
 * `vercel.json`, which repeats them — getting isolation in dev and losing it
 * in production would quietly quadruple inference latency for every visitor.
 */
const isolationHeaders: Connect.NextHandleFunction = (_req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');
  next();
};

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      name: 'cross-origin-isolation',
      configureServer: (server) => {
        server.middlewares.use(isolationHeaders);
      },
      configurePreviewServer: (server) => {
        server.middlewares.use(isolationHeaders);
      },
    },
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
      // onnxruntime-web's package exports do not list `./dist/*`, so the `?url`
      // imports the vision worker uses to locate its WASM binaries cannot
      // resolve through them. Aliasing the directory goes around the exports
      // map. The alternative — copying 23 MB of WASM into `public/` — would
      // ship it twice.
      'onnxruntime-web/dist': path.resolve(__dirname, 'node_modules/onnxruntime-web/dist'),
    },
  },
  optimizeDeps: {
    // The runtime ships its own WASM loader and must not be pre-bundled;
    // esbuild rewrites the loader's import.meta.url and it then looks for the
    // binaries in the wrong place.
    exclude: ['onnxruntime-web'],
  },
  worker: {
    format: 'es' as const,
  },
  build: {
    // The ONNX graphs are served from `public/` as plain files; only the JS is
    // chunked. 900 kB is above Vite's default warning but below what the
    // runtime actually weighs, so the warning would fire on every build and
    // train everyone to ignore it.
    chunkSizeWarningLimit: 900,
  },
});
