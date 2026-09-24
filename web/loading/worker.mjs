import { download } from './download.mjs';
import { cached, remember } from './cache.mjs';
const queued = [];
self.onmessage = (event) => {
  // Input can arrive before the physics module installs its handler.
  if (queued.length >= 1024) throw Error('Worker startup queue exceeded');
  queued.push(event);
};
try {
  const url = globalThis.__OPENBENI_BOOT__.url;
  let response = await cached('openbeni-code-v1', url);
  if (!response) {
    response = await download(url);
    await remember('openbeni-code-v1', url, response);
  }
  const moduleUrl = URL.createObjectURL(
    new Blob([await response.text()], { type: 'text/javascript' }),
  );
  try {
    await import(/* @vite-ignore */ moduleUrl);
    for (const event of queued) self.onmessage(event);
    queued.length = 0;
  } finally {
    URL.revokeObjectURL(moduleUrl);
  }
} catch (error) {
  console.error('Worker startup failed', error);
  queued.length = 0;
  self.postMessage({
    type: 'fatal',
    text: 'Unable to download game. Please retry or choose the backup source.',
  });
}
