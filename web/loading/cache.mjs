const DAY = 86_400_000;
const LAST_USED = 'X-OpenBeni-Last-Used';
const opened = new Map();

async function open(name) {
  if (!opened.has(name))
    opened.set(
      name,
      (async () => {
        try {
          const cache = await caches.open(name);
          // Cleanup is infrequent and age-based: another release's files stay usable.
          const marker = new URL('/__asset_cache__/cleanup', location.origin)
            .href;
          const previous = await cache.match(marker);
          if (
            Date.now() - Number(previous?.headers.get(LAST_USED) || 0) >
            DAY
          ) {
            await cache.put(marker, stamp(new Response('')));
            void (async () => {
              for (const request of await cache.keys()) {
                if (request.url === marker) continue;
                const hit = await cache.match(request);
                const used = Number(hit?.headers.get(LAST_USED) || 0);
                if (!hit) continue;
                if (!used)
                  await cache.put(request, stamp(hit)); // Migrate existing cache entries.
                else if (Date.now() - used > 30 * DAY)
                  await cache.delete(request);
              }
            })().catch(() => {});
          }
          return cache;
        } catch {
          return undefined;
        }
      })(),
    );
  return opened.get(name);
}
function stamp(response) {
  const headers = new Headers(response.headers);
  headers.set(LAST_USED, String(Date.now()));
  return new Response(response.body, { headers });
}
export async function cached(name, key) {
  try {
    const cache = await open(name);
    const hit = await cache?.match(key);
    if (hit && Date.now() - Number(hit.headers.get(LAST_USED) || 0) > DAY) {
      // Throttle touches to avoid rewriting large files on every frame/refresh.
      void cache.put(key, stamp(hit.clone())).catch(() => {});
    }
    return hit;
  } catch {
    return undefined;
  }
}
export async function remember(name, key, response) {
  try {
    await (await open(name))?.put(key, stamp(response.clone()));
  } catch {
    /* Persistence is optional, including quota and private-mode failures. */
  }
}
