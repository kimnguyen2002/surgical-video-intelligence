/// <reference types="vite/client" />

/**
 * Vite resolves `?url` imports to the emitted asset path. The onnxruntime-web
 * WASM binaries are loaded that way rather than being copied into `public/`:
 * the 23 MB jsep build then stays in node_modules, which AI Studio installs
 * itself, instead of becoming 23 MB that has to be uploaded with the project.
 */
declare module '*?url' {
  const src: string;
  export default src;
}
