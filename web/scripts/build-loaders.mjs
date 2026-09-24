import { build } from 'esbuild';
import { mkdir, writeFile } from 'node:fs/promises';
const entries = {};
for (const name of ['frontend', 'worker']) {
  const result = await build({
    entryPoints: [`loading/${name}.mjs`],
    bundle: true,
    format: 'esm',
    target: 'es2022',
    minify: true,
    write: false,
  });
  entries[name] = result.outputFiles[0].text;
}
await mkdir('.generated', { recursive: true });
await writeFile('.generated/loaders.json', JSON.stringify(entries));
