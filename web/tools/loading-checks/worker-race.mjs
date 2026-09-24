/** Exercise an actual in-flight course cancellation, independently of UI guards. */
import { chromium } from 'playwright-core';
import { readFile, readdir } from 'node:fs/promises';
import assert from 'node:assert/strict';
const root = new URL('../../dist/', import.meta.url);
const entry = (await readdir(new URL('assets/', root))).find((name) =>
  /^sim.worker-.*\.js$/.test(name),
);
const tracks = JSON.parse(
  await readFile(new URL('runtime/tracks/manifest.json', root)),
);
const delayedPath = `/runtime/tracks/${tracks.tracks.find((t) => t.id === 'switchyard').key}.mjb.gz`;
const browser = await chromium.launch({
  executablePath: process.env.CHROME_BIN || '/usr/bin/google-chrome',
  headless: true,
  args: ['--no-sandbox'],
});
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  let started;
  const requested = new Promise((resolve) => {
    started = resolve;
  });
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  await context.route('https://huggingface.co/**', async (route) => {
    const path = new URL(route.request().url()).pathname.split('/dist')[1];
    if (path === delayedPath) {
      started();
      await gate;
    }
    try {
      await route.fulfill({
        status: 200,
        headers: { 'access-control-allow-origin': '*' },
        body: await readFile(new URL('.' + path, root)),
      });
    } catch {
      /* Superseded requests have been aborted by the worker. */
    }
  });
  await page.goto(
    (process.env.GAME_URL || 'http://127.0.0.1:17863/') + 'game-config.json',
  );
  await page.evaluate((entry) => {
    window.events = [];
    window.testWorker = new Worker('/assets/' + entry, {
      type: 'module',
      name: 'singleplayer',
    });
    window.testWorker.onmessage = (event) => {
      if (event.data.type !== 'frame') window.events.push(event.data);
    };
  }, entry);
  await page.waitForFunction(
    () => window.events.some((e) => e.type === 'ready'),
    null,
    { timeout: 60000 },
  );
  await page.evaluate(() => {
    window.events = [];
    window.testWorker.postMessage({ type: 'map', map: 'switchyard' });
  });
  await Promise.race([
    requested,
    new Promise((_, reject) => {
      const timer = setTimeout(
        () => reject(Error('Course download did not start')),
        15000,
      );
      timer.unref();
    }),
  ]);
  await page.evaluate(() =>
    window.testWorker.postMessage({ type: 'map', map: 'harbor' }),
  );
  await page.waitForFunction(
    () =>
      window.events.some((e) => e.type === 'mapReady' && e.map === 'harbor'),
    null,
    { timeout: 60000 },
  );
  release();
  const events = await page.evaluate(() => window.events);
  assert.equal(
    events.some((e) => e.type === 'mapReady' && e.map === 'switchyard'),
    false,
  );
  assert.equal(
    events.some((e) => ['error', 'fatal'].includes(e.type)),
    false,
  );
  assert.ok(
    events.some(
      (e) => e.type === 'status' && e.text.startsWith('Downloading…'),
    ),
  );
  await page.evaluate(() => window.testWorker.terminate());
  console.log(
    'PASS: in-flight old course aborted; latest course ready; no stale mapReady/error',
  );
} finally {
  await browser.close();
}
