#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import net from 'node:net';
import { fileURLToPath } from 'node:url';
import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';
import { HELP, parseConfig } from './config.mjs';
const root = fileURLToPath(new URL('../../', import.meta.url));
let config, server, active;
const children = new Set();
async function stop(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  const closed = once(child, 'close');
  child.kill('SIGTERM');
  const timer = setTimeout(() => child.kill('SIGKILL'), 5000);
  try {
    await closed;
  } finally {
    clearTimeout(timer);
  }
}
let interrupted = false;
async function interrupt() {
  if (interrupted) return;
  interrupted = true;
  await stop(active);
  await stop(server);
  process.exit(130);
}
process.once('SIGINT', interrupt);
process.once('SIGTERM', interrupt);
function start(args, env) {
  const child = spawn(process.execPath, args, {
    cwd: root,
    env,
    stdio: 'inherit',
  });
  children.add(child);
  child.once('close', () => children.delete(child));
  return child;
}
async function run(args, env) {
  active = start(args, env);
  try {
    const [code] = await once(active, 'close');
    if (code !== 0) throw Error(`${args[0]} exited with code ${code}`);
  } finally {
    active = null;
  }
}
async function availablePort(port) {
  const probe = net.createServer();
  probe.listen(port, '127.0.0.1');
  await once(probe, 'listening');
  const result = probe.address().port;
  await new Promise((r) => probe.close(r));
  return result;
}
try {
  config = parseConfig(process.argv.slice(2), root);
  if (config.help) {
    console.log(HELP);
  } else {
    for (const cmd of ['ffmpeg', 'ffprobe']) {
      const r = spawnSync(cmd, ['-version'], { stdio: 'ignore' });
      if (r.error || r.status !== 0) throw Error(`${cmd} is required on PATH`);
    }
    const chrome = process.env.CHROME_BIN ?? '/usr/bin/google-chrome';
    await fs.access(chrome);
    await fs.mkdir(config.output, { recursive: true });
    if (config.renderOnly) {
      const frames = JSON.parse(
        await fs.readFile(path.join(config.output, 'replay.json'), 'utf8'),
      );
      const verification = JSON.parse(
        await fs.readFile(
          path.join(config.output, 'verification.json'),
          'utf8',
        ),
      );
      if (
        !Array.isArray(frames) ||
        frames.length < 2 ||
        verification.physics !== 'MuJoCo WASM' ||
        verification.manualPositionUpdatesAfterSpawn !== 0
      )
        throw Error('A verified real-physics replay is required');
    }
    const env = {
      ...process.env,
      CHROME_BIN: chrome,
      DEMO_MAP: config.map,
      DEMO_TOUR: config.tour ? '1' : '0',
      DEMO_PLAYERS: String(config.players),
      DEMO_SECONDS: String(config.seconds),
      DEMO_FPS: String(config.fps),
      DEMO_OUTPUT: config.output,
    };
    // Bind-check first so an occupied user port fails before expensive simulation.
    const port = await availablePort(config.port);
    await run(['scripts/sync-runtime.mjs'], env);
    if (!config.renderOnly) await run(['tools/demo-recorder/capture.mjs'], env);
    server = start(['server/index.mjs', '--dev'], {
      ...env,
      PORT: String(port),
      HOST: '127.0.0.1',
      DEMO_FREEZE_SOURCE: '1',
    });
    const url = `http://127.0.0.1:${port}`;
    const deadline = Date.now() + 30000;
    let ready = false;
    while (Date.now() < deadline) {
      if (server.exitCode !== null || server.signalCode !== null)
        throw Error('Recording server failed to start');
      try {
        ready =
          (
            await (
              await fetch(url + '/api/health', {
                signal: AbortSignal.timeout(1000),
              })
            ).json()
          ).ok === true;
      } catch {}
      if (ready) break;
      await new Promise((r) => setTimeout(r, 150));
    }
    if (!ready) throw Error('Recording server startup timed out');
    await run(['tools/demo-recorder/render.mjs'], { ...env, DEMO_URL: url });
    const report = JSON.parse(
      await fs.readFile(
        path.join(config.output, 'render-verification.json'),
        'utf8',
      ),
    );
    if (report.errors.length || report.fps !== config.fps)
      throw Error('Render verification failed');
    const video = path.join(
      config.output,
      `OpenBeni-physics-${report.players.length}players-${config.fps}fps.mp4`,
    );
    const probe = spawnSync(
      'ffprobe',
      [
        '-v',
        'error',
        '-show_entries',
        'stream=nb_frames,avg_frame_rate,width,height',
        '-of',
        'json',
        video,
      ],
      { encoding: 'utf8' },
    );
    if (probe.status !== 0) throw Error('ffprobe could not read the video');
    const stream = JSON.parse(probe.stdout).streams[0];
    if (
      Number(stream.nb_frames) !== report.count ||
      stream.avg_frame_rate !== `${config.fps}/1`
    )
      throw Error('Video frame count/rate mismatch');
    console.log(
      `Verified video: ${video}\n${report.players.length} players, ${report.duration.toFixed(2)} s, ${config.fps} fps, ${stream.width}×${stream.height}`,
    );
  }
} catch (error) {
  console.error(`demo-recorder: ${error.message}`);
  process.exitCode = 1;
} finally {
  await stop(active);
  await stop(server);
}
