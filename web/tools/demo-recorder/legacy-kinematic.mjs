import { chromium } from 'playwright-core';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
const base = process.env.DEMO_URL ?? 'http://127.0.0.1:3001';
const out = path.resolve(process.env.DEMO_OUTPUT ?? 'outputs/multiplayer-demo');
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: '/usr/bin/google-chrome',
  headless: true,
  args: [
    '--no-sandbox',
    '--enable-unsafe-swiftshader',
    '--disable-background-timer-throttling',
  ],
});
const context = await browser.newContext({
  viewport: { width: 1280, height: 720 },
  deviceScaleFactor: 1,
  recordVideo: { dir: out, size: { width: 1280, height: 720 } },
});
const page = await context.newPage();
const errors = [];
page.on('pageerror', (e) => {
  errors.push(String(e));
  console.error('PAGE ERROR', String(e));
});
page.on('console', (m) => {
  if (m.type() === 'error') console.error('CONSOLE', m.text());
});
try {
  await page.goto(base + '/tools/demo-recorder/index.html?kinematic=1');
  await page.waitForSelector('body[data-ready="true"]', {
    timeout: 90000,
    state: 'attached',
  });
  await page.screenshot({ path: path.join(out, 'preview.png') });
  console.log('READY: six mock players joined and ready');
  const start = Date.now();
  await page.waitForTimeout(1500);
  await page.evaluate(() => window.demo.start());
  await page.waitForTimeout(12000);
  await page.screenshot({ path: path.join(out, 'race.png') });
  await page.waitForSelector('body[data-finished="true"]', {
    timeout: 100000,
    state: 'attached',
  });
  const state = await page.evaluate(() => window.demo.state());
  if (
    state.players.length !== 6 ||
    !state.players.every((p) => p.finished != null)
  )
    throw Error('Not all six players finished');
  await page.waitForTimeout(4000);
  const liveDuration = (Date.now() - start) / 1000;
  await page.screenshot({ path: path.join(out, 'finish.png') });
  await fs.writeFile(
    path.join(out, 'race-state.json'),
    JSON.stringify(
      {
        ...state,
        errors,
        mock: 'Kinematic drivers; real WebSocket race service, original robot meshes and game renderer',
      },
      null,
      2,
    ),
  );
  await page.evaluate(() => window.demo.stop());
  const video = page.video();
  await context.close();
  const raw = await video.path();
  const probe = spawnSync(
    'ffprobe',
    [
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=nw=1:nk=1',
      raw,
    ],
    { encoding: 'utf8' },
  );
  const duration = Number(probe.stdout.trim());
  const output = path.join(out, 'OpenBeni-multiplayer-demo.mp4');
  const encode = spawnSync(
    'ffmpeg',
    [
      '-y',
      '-ss',
      String(Math.max(0, duration - liveDuration)),
      '-i',
      raw,
      '-t',
      String(liveDuration),
      '-an',
      '-c:v',
      'libx264',
      '-preset',
      'fast',
      '-crf',
      '20',
      '-pix_fmt',
      'yuv420p',
      '-movflags',
      '+faststart',
      output,
    ],
    { encoding: 'utf8' },
  );
  if (encode.status !== 0) throw Error(encode.stderr);
  console.log('VIDEO', output, 'seconds', liveDuration, 'errors', errors);
  if (errors.length) throw Error(errors.join('\n'));
} catch (error) {
  await page.screenshot({ path: path.join(out, 'error.png') }).catch(() => {});
  throw error;
} finally {
  await browser.close();
}
