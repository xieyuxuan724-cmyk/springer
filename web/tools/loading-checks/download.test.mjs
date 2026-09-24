import { test } from 'node:test';
import assert from 'node:assert/strict';
import { download } from '../../loading/download.mjs';
import { cached, remember } from '../../loading/cache.mjs';
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const originalFetch = globalThis.fetch;
const options = { idleMs: 100, retryDelayMs: 1 };
function streaming(signal, count, gap) {
  let interval;
  return new Response(
    new ReadableStream({
      start(controller) {
        let sent = 0;
        interval = setInterval(() => {
          controller.enqueue(new Uint8Array([1]));
          if (++sent === count) {
            clearInterval(interval);
            controller.close();
          }
        }, gap);
        signal.addEventListener(
          'abort',
          () => {
            clearInterval(interval);
            controller.error(signal.reason);
          },
          { once: true },
        );
      },
      cancel() {
        clearInterval(interval);
      },
    }),
  );
}
await test('continuous data can take longer than the idle deadline', async () => {
  globalThis.fetch = async (_url, { signal }) => streaming(signal, 8, 25);
  try {
    assert.equal(
      (await (await download('https://cdn.test/file', options)).arrayBuffer())
        .byteLength,
      8,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
await test('stalls retry three times on the selected source only', async () => {
  const urls = [];
  globalThis.fetch = async (url, { signal }) => {
    urls.push(url);
    return new Promise((_, reject) =>
      signal.addEventListener('abort', () => reject(signal.reason), {
        once: true,
      }),
    );
  };
  try {
    await assert.rejects(
      download('https://cdn.test/file', { ...options, idleMs: 10 }),
    );
    assert.deepEqual(urls, Array(3).fill('https://cdn.test/file'));
  } finally {
    globalThis.fetch = originalFetch;
  }
});
await test('404 fails once, corrupt payload retries and progress restarts', async () => {
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return new Response('', { status: 404 });
  };
  try {
    await assert.rejects(download('https://cdn.test/file', options));
    assert.equal(calls, 1);
    const progress = [];
    globalThis.fetch = async () => new Response('bad');
    await assert.rejects(
      download('https://cdn.test/file', {
        ...options,
        onProgress: (n) => progress.push(n),
        validate: async () => {
          throw Error('hash');
        },
      }),
    );
    assert.deepEqual(progress, [0, 3, 0, 3, 0, 3]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
await test('cancel aborts without retry; verification is outside the idle clock', async () => {
  let calls = 0;
  const controller = new AbortController();
  globalThis.fetch = async (_url, { signal }) => {
    calls++;
    return streaming(signal, 100, 20);
  };
  try {
    const promise = download('https://cdn.test/file', {
      ...options,
      signal: controller.signal,
    });
    setTimeout(() => controller.abort(), 30);
    await assert.rejects(promise, { name: 'AbortError' });
    assert.equal(calls, 1);
    let internal;
    globalThis.fetch = async (_url, { signal }) => {
      internal = signal;
      return new Response('ok');
    };
    await download('https://cdn.test/file', {
      ...options,
      idleMs: 10,
      validate: async () => {
        await delay(35);
        assert.equal(internal.aborted, false);
      },
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});
await test('optional storage failure does not break downloaded responses', async () => {
  globalThis.caches = {
    open: async () => {
      throw Error('Storage disabled');
    },
  };
  assert.equal(await cached('blocked', 'https://cdn.test/file'), undefined);
  const response = new Response('usable');
  await remember('blocked', 'https://cdn.test/file', response);
  assert.equal(await response.text(), 'usable');
});
await test('age-based cleanup retains recent releases and reuses cached bytes', async () => {
  const entries = new Map();
  const put = async (key, value) =>
    entries.set(typeof key === 'string' ? key : key.url, value.clone());
  const match = async (key) =>
    entries.get(typeof key === 'string' ? key : key.url)?.clone();
  globalThis.location = new URL('https://game.test/');
  globalThis.caches = {
    open: async () => ({
      put,
      match,
      keys: async () => [...entries.keys()].map((url) => ({ url })),
      delete: async (key) => entries.delete(key.url),
    }),
  };
  await put(
    'https://cdn.test/old',
    new Response('old', {
      headers: { 'X-OpenBeni-Last-Used': String(Date.now() - 31 * 86400000) },
    }),
  );
  await put(
    'https://cdn.test/recent',
    new Response('recent', {
      headers: { 'X-OpenBeni-Last-Used': String(Date.now()) },
    }),
  );
  await remember('age-test', 'https://cdn.test/new', new Response('new'));
  await delay(20);
  assert.equal(await cached('age-test', 'https://cdn.test/old'), undefined);
  assert.equal(
    await (await cached('age-test', 'https://cdn.test/recent')).text(),
    'recent',
  );
  assert.equal(
    await (await cached('age-test', 'https://cdn.test/new')).text(),
    'new',
  );
});
