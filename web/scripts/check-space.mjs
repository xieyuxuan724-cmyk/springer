/** Check the deployed source build before enabling its immutable frontend CDN. */
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { download } from '../loading/download.mjs';
const origin = new URL(process.argv[2]);
const localRoot = new URL('../dist/', import.meta.url);
const remote = new URL('/?assetSource=local', origin);
const response = await download(remote.href);
if (!response.ok) throw Error(`Homepage HTTP ${response.status}`);
const localHtml = await readFile(new URL('index.html', localRoot), 'utf8');
const remoteHtml = await response.text();
const entries = (html) =>
  [...html.matchAll(/(?:src|href)="(\/assets\/[^"]+\.(?:js|css))"/g)]
    .map((match) => match[1])
    .sort((a, b) => a.localeCompare(b));
const localEntries = entries(localHtml);
if (
  !localEntries.length ||
  JSON.stringify(localEntries) !== JSON.stringify(entries(remoteHtml))
) {
  throw Error(
    'Source build differs from the CDN export. Do not enable this CDN snapshot.',
  );
}
console.log('Deployed entry filenames match the local export.');
const paths = new Set(localEntries);
for (const path of localEntries.filter((path) => path.endsWith('.js'))) {
  const source = await readFile(new URL('.' + path, localRoot), 'utf8');
  for (const match of source.matchAll(
    /\/assets\/sim\.worker-[A-Za-z0-9_-]+\.js/g,
  ))
    paths.add(match[0]);
}
for (const path of paths) {
  console.log('Checking', path);
  const local = await readFile(new URL('.' + path, localRoot));
  const response = await download(new URL(path, origin).href);
  if (!response.ok) throw Error(`Asset HTTP ${response.status}`);
  const remote = Buffer.from(await response.arrayBuffer());
  const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
  if (hash(local) !== hash(remote))
    throw Error(`Deployed artifact differs: ${path}`);
}
console.log(
  `Verified ${paths.size} deployed frontend/worker artifacts against the local export.`,
);
