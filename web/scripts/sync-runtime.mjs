import { mkdir, copyFile, readdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
// Export first: compiled tracks depend on the generated MJCF and actuator config.
execFileSync('python3', ['scripts/export-assets.py'], { stdio: 'inherit' });
await mkdir('public/runtime', { recursive: true });
await copyFile(
  'node_modules/@mujoco/mujoco/mujoco.wasm',
  'public/runtime/mujoco.wasm',
);
for (const f of await readdir('node_modules/onnxruntime-web/dist'))
  if (/^ort-wasm-simd-threaded\.(wasm|mjs)$/.test(f))
    await copyFile(
      'node_modules/onnxruntime-web/dist/' + f,
      'public/runtime/' + f,
    );
await import('./compile-tracks.mjs');

await import("./asset-manifest.mjs");

await import('./build-loaders.mjs');
