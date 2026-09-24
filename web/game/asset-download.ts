import manifest from '../.generated/asset-manifest.json';
import { download } from '../loading/download.mjs';
import { cached, remember } from '../loading/cache.mjs';
const base = (
  (import.meta as ImportMeta & { env?: Record<string, string> }).env
    ?.VITE_ASSET_BASE_URL ?? ''
).replace(/\/$/, '');
const localOnly =
  new URL(globalThis.location.href).searchParams.get('assetSource') ===
    'local' ||
  ('name' in globalThis &&
    String((globalThis as typeof globalThis & { name?: string }).name).endsWith(
      ':local',
    ));
export function assetUrl(path: string) {
  return !localOnly && base && /^\/(robot|runtime|policies)\//.test(path)
    ? base + path
    : path;
}
const metadata: Record<string, { size: number; hash: string }> =
  manifest.assets;
const CACHE = 'openbeni-assets-v1';
const key = (path: string) =>
  new URL('/__asset_cache__/' + metadata[path].hash, location.origin).href;
let active = 0;
const waiting: (() => void)[] = [];
let progress: ((text: string) => void) | undefined;
let generation = 0;
let planned = new Map<
  string,
  { size: number; received: number; cached: boolean }
>();
let lastProgress = 0;
export function onDownloadProgress(callback: (text: string) => void) {
  progress = callback;
}
function report(force = false) {
  if (!force && performance.now() - lastProgress < 200) return;
  lastProgress = performance.now();
  const network = [...planned.values()].filter((p) => !p.cached);
  const total = network.reduce((n, p) => n + p.size, 0);
  const received = network.reduce((n, p) => n + p.received, 0);
  progress?.(
    total
      ? `Downloading… ${(received / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`
      : 'Loading cached assets…',
  );
}
export async function prepareDownloads(
  mapId: string,
  paths?: string[],
  signal?: AbortSignal,
) {
  const current = ++generation;
  const entries = await Promise.all(
    [
      ...new Set(
        paths ?? (manifest.startup as Record<string, string[]>)[mapId] ?? [],
      ),
    ]
      .filter((path) => metadata[path])
      .map(
        async (path) =>
          [
            path,
            {
              size: metadata[path].size,
              received: 0,
              cached: !!(await cached(CACHE, key(path))),
            },
          ] as const,
      ),
  );
  signal?.throwIfAborted();
  if (current !== generation) return;
  planned = new Map(entries);
  report(true);
}
async function acquire() {
  if (active >= 4) await new Promise<void>((resolve) => waiting.push(resolve));
  else active++;
}
function release() {
  const next = waiting.shift();
  if (next) next();
  else active--;
}
/** Four transfers per execution context; only complete verified files persist. */
export async function downloadAsset(
  path: string,
  cache?: RequestCache,
  signal?: AbortSignal,
): Promise<Response> {
  signal?.throwIfAborted();
  const meta = metadata[path];
  const current = generation;
  const item = planned.get(path);
  const update = (received: number) => {
    if (!item || current !== generation || signal?.aborted) return;
    item.cached = false;
    item.received = Math.min(received, item.size);
    report(received === item.size);
  };
  if (meta) {
    const hit = await cached(CACHE, key(path));
    signal?.throwIfAborted();
    if (hit) {
      // A file may have completed in another operation since planning.
      if (item && !item.cached) update(item.size);
      return hit;
    }
  }
  await acquire();
  try {
    signal?.throwIfAborted();
    const response = await download(
      new URL(assetUrl(path), location.origin).href,
      {
        cache,
        signal,
        onProgress: update,
        validate: async (blob: Blob) => {
          if (!meta) return;
          const hash = [
            ...new Uint8Array(
              await crypto.subtle.digest('SHA-256', await blob.arrayBuffer()),
            ),
          ]
            .map((n) => n.toString(16).padStart(2, '0'))
            .join('');
          if (blob.size !== meta.size || hash !== meta.hash)
            throw Error('Downloaded asset failed integrity validation');
        },
      },
    );
    if (!response) throw Error('Download failed. Please retry.');
    if (meta) await remember(CACHE, key(path), response);
    return response;
  } finally {
    release();
  }
}
