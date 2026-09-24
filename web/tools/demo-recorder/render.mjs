import { chromium } from 'playwright-core';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
const out = path.resolve(process.env.DEMO_OUTPUT ?? 'outputs/physics-demo');
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.CHROME_BIN ?? '/usr/bin/google-chrome',
  headless: true,
  args: [
    '--no-sandbox',
    '--enable-unsafe-swiftshader',
    '--use-angle=swiftshader',
    '--disable-dev-shm-usage',
    '--disable-background-timer-throttling',
  ],
});
const page = await browser.newPage({
  viewport: { width: 1280, height: 720 },
  deviceScaleFactor: 1,
});
const errors = [];
page.on('pageerror', (e) => {
  errors.push(String(e));
  console.error(String(e));
});
let encoder;
const cancel = async () => {
  encoder?.kill();
  await browser.close();
  process.exit(130);
};
process.once('SIGINT', cancel);
process.once('SIGTERM', cancel);
try {
  await page.route('**/__demo-replay.json', async route => route.fulfill({
    contentType: 'application/json', body: await fs.readFile(path.join(out, 'replay.json'), 'utf8'),
  }));
  await page.goto(
    (process.env.DEMO_URL ?? 'http://127.0.0.1:3001') +
      '/tools/demo-recorder/index.html?physics=1&tour=' +
      (process.env.DEMO_TOUR === '1' ? '1' : '0') +
      '&replay=' +
      encodeURIComponent('/__demo-replay.json'),
  );
  await page.waitForSelector('body[data-ready="true"]', {
    state: 'attached',
    timeout: 90000,
  });
  const frames = await page.evaluate(() => window.demo.beginReplay());
  const players = frames[0].players.length;
  if (!players || frames.some((frame) => frame.players.length !== players))
    throw Error('Inconsistent player count in physics replay');
  if (process.env.DEMO_FOCUS !== undefined) {
    const focus = Number(process.env.DEMO_FOCUS);
    if (!Number.isInteger(focus) || focus < 0 || focus >= players)
      throw Error('DEMO_FOCUS must be a player index within the replay');
    await page.evaluate((index) => window.demo.focus(index), focus);
  }
  await fs.writeFile(path.join(out, 'replay.json'), JSON.stringify(frames));
  const fps = Number(process.env.DEMO_FPS ?? 60),
    start = frames[0].now,
    duration = (frames.at(-1).now - start) / 1000,
    count = Math.ceil(duration * fps);
  const output = path.join(
    out,
    `OpenBeni-physics-${players}players-${fps}fps.mp4`,
  );
  encoder = spawn('ffmpeg', [
    '-y',
    '-hide_banner',
    '-loglevel',
    'error',
    '-f',
    'image2pipe',
    '-framerate',
    String(fps),
    '-vcodec',
    'mjpeg',
    '-i',
    'pipe:0',
    '-an',
    '-c:v',
    'libx264',
    '-preset',
    'fast',
    '-crf',
    '18',
    '-pix_fmt',
    'yuv420p',
    '-movflags',
    '+faststart',
    output,
  ]);
  let encodingErrors = '';
  encoder.stderr.on('data', (d) => (encodingErrors += d));
  const done = once(encoder, 'close');
  console.log('Rendering', count, 'unique frames at', fps, 'fps');
  for (let i = 0; i < count; i++) {
    await page.evaluate(
      (t) => window.demo.renderFrame(t),
      start + (i * 1000) / fps,
    );
    const frame = await page.screenshot({ type: 'jpeg', quality: 92 });
    if (!encoder.stdin.write(frame)) await once(encoder.stdin, 'drain');
    if (i === 0 || i === 600)
      await fs.writeFile(path.join(out, `frame-${i}.jpg`), frame);
    if (i % 120 === 0) console.log(`Frames ${i}/${count}`);
  }
  encoder.stdin.end();
  const [exitCode] = await done;
  if (exitCode !== 0) throw Error(encodingErrors);
  await fs.writeFile(
    path.join(out, 'render-verification.json'),
    JSON.stringify(
      {
        map: frames[0].map,
        fps,
        count,
        duration,
        errors,
        effects: await page.evaluate(() => window.demo.state().effects),
        players: frames.at(-1).players,
        mock: false,
        method:
          'Fixed-time replay, independently rendered screenshots, no frame duplication or motion-estimation interpolation',
      },
      null,
      2,
    ),
  );
  if (errors.length) throw Error(errors.join('\n'));
  console.log('COMPLETE', output);
} finally {
  encoder?.kill();
  await browser.close();
}
