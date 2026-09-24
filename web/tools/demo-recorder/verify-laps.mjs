#!/usr/bin/env node
import { assetPath } from '../../scripts/browser-assets.mjs';
// A full-course capability check using the production physics and race rules.
// The driver only sends vx/wz and jump requests. No rescue or state editing.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';
import load from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web';
import config from '../../game/robot-config.json' with { type: 'json' };
import { Simulation, Policy } from '../../game/physics.mjs';
import {
  MAPS,
  getMap,
  getTrack,
  surfaceAt,
  atDistance,
  spawn,
  crossPoint,
  branchOffset,
  routeLengths,
} from '../../game/maps.mjs';
import { RaceService } from '../../server/race.mjs';
import { jumpApproach } from '../../game/course-driver.mjs';
import {
  boostedForwardSpeed,
  terrainTurboFactor,
} from '../../game/drive-command.mjs';
import { seededRandom } from './seed.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const { values } = parseArgs({
  options: {
    maps: { type: 'string', default: MAPS.map((m) => m.id).join(',') },
    branches: { type: 'string', default: 'left,right' },
    slots: { type: 'string', default: '0,1,2' },
    laps: { type: 'string', default: '2' },
    speed: { type: 'string', default: '1.5' },
    seed: { type: 'string', default: '20260914' },
    seconds: { type: 'string', default: '240' },
    output: { type: 'string', default: 'outputs/lap-verification' },
    help: { type: 'boolean' },
  },
});
if (values.help) {
  console.log(`Usage: npm run test:laps -- [options]
  --maps IDS          Courses to check (default: all)
  --branches left,right  Split routes to verify (default: both)
  --slots 0,1,2        Real starting grid slots, 0–7 (default: front row)
  --laps 2             Complete physical circuits, 1 or 2 (default: 2)
  --speed 1.5          Maximum forward command, 0.2–2 m/s (default: 1.5)
  --seconds 240        Timeout per run, 10–600 simulated seconds
  --seed 20260914      Repeatable gem placement and pickup bonus rolls
  --output DIR         Report/trajectory directory inside web/outputs

No server, browser or GPU required. Uses actual MuJoCo WASM + ONNX.
Nonzero exit on timeout, stall, off-track, reset, or incomplete checkpoints.
The driver slows in bends, jumps the marked trenches and receives automatic pickup boosts. This checks course
traversability, not all driving styles, uploaded policies.`);
  process.exit(0);
}
const mapIds = values.maps.split(','),
  slots = values.slots.split(',').map(Number);
const branches = values.branches.split(',');
if (!branches.length || branches.some((b) => !['left', 'right'].includes(b)))
  throw Error('Branches must be left/right');
const laps = Number(values.laps),
  speed = Number(values.speed),
  timeout = Number(values.seconds);
const seed = Number(values.seed),
  random = seededRandom(seed);
if (!mapIds.length || mapIds.some((id) => !MAPS.some((m) => m.id === id)))
  throw Error('Unknown map');
if (
  !slots.length ||
  values.slots.split(',').some((s) => !/^\d$/.test(s)) ||
  slots.some((n) => n > 7)
)
  throw Error('Slots must be 0–7');
if (![1, 2].includes(laps)) throw Error('Laps must be 1 or 2');
if (!Number.isFinite(speed) || speed < 0.2 || speed > 2)
  throw Error('Speed must be 0.2–2 m/s');
if (!Number.isFinite(timeout) || timeout < 10 || timeout > 600)
  throw Error('Timeout must be 10–600 seconds');
const out = path.resolve(root, values.output),
  relative = path.relative(path.join(root, 'outputs'), out);
if (relative.startsWith('..') || path.isAbsolute(relative))
  throw Error('Output must be inside web/outputs');
process.chdir(root);
fs.mkdirSync(out, { recursive: true });
const hash = (value) => createHash('sha256').update(value).digest('hex');
const manifest = JSON.parse(fs.readFileSync(assetPath('robot/manifest.json')));
const xml = fs.readFileSync(assetPath('robot/beni.xml'), 'utf8');
const bytes = new Uint8Array(fs.readFileSync(assetPath('policies/beni.onnx')));
const sourcePolicySha256 = hash(
  fs.readFileSync('../assets/models/policy.onnx'),
);
if (hash(bytes) !== sourcePolicySha256)
  throw Error(
    'Browser policy differs from current upstream assets; export assets first',
  );
const dt = config.physicsDt * config.decimation;
const results = [];
const cases = mapIds.flatMap((mapId) =>
  slots.flatMap((slot) =>
    (getMap(mapId).splits?.length ? branches : ['centre']).map((branch) => ({
      mapId,
      slot,
      branch,
    })),
  ),
);
const report = {
  seed,
  createdAt: new Date().toISOString(),
  physics: 'MuJoCo WASM + ONNX Runtime WASM',
  policySha256: hash(bytes),
  sourcePolicySha256,
  robotAssetsSha256: Object.fromEntries(
    manifest.map((f) => [f, hash(fs.readFileSync(assetPath('robot/' + f)))]),
  ),
  mapsSha256: hash(fs.readFileSync('game/maps.mjs')),
  terrainReliefSha256: hash(fs.readFileSync('game/terrain-relief.mjs')),
  robotConfig: config,
  requestedLaps: laps,
  cruiseForwardCommand: speed,
  maximumForwardCommand: 2,
  driver:
    'heading feedback and branch steering; 0.45 m lookahead on the compact crossing course, 0.65 m elsewhere; 2 s settling, slows in bends',
  manualPositionUpdatesAfterSpawn: 0,
  jumps: 0,
  itemsUsed: 0,
  scope:
    'Independent runs; ordered RaceService checkpoints and full physical circuit distance. Does not test arbitrary steering, other policies or realtime rendering. Continues integrating to the full circuit distance if the server finish radius awards the lap slightly earlier.',
  results,
};
const saveReport = () => {
  report.jumps = results.reduce((n, r) => n + (r.jumps ?? 0), 0);
  fs.writeFileSync(
    path.join(out, 'report.json'),
    JSON.stringify(report, null, 2),
  );
};
ort.env.wasm.numThreads = 1;
const mj = await load();
for (const dir of ['/robot', '/robot/meshes', '/robot/textures'])
  mj.FS.mkdir(dir);
for (const f of manifest)
  mj.FS.writeFile('/robot/' + f, fs.readFileSync(assetPath('robot/' + f)));
const policy = await Policy.create(ort, bytes);
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
try {
  for (const { mapId, slot, branch } of cases) {
    const map = getMap(mapId),
      track = getTrack(map);
    const sim = new Simulation(mj, xml, mapId);
    let clock = 1000000,
      resets = 0;
    const game = new RaceService({ now: () => clock, random });
    const player = game.connect('Lap Verification', null, (m) => {
      if (m.type === 'reset') resets++;
    });
    game.handle(player, { type: 'create', map: mapId });
    // Pick a real starting grid position before the start/reset, then audit all resets.
    player.slot = slot;
    game.handle(player, { type: 'ready', ready: true });
    game.handle(player, { type: 'start' });
    const room = game.rooms.get(player.room);
    sim.reset(spawn(map, slot));
    policy.reset();
    resets = 0;
    sim.reset = () => {
      throw Error('Reset during lap verification');
    };
    let prevS = surfaceAt(map, ...sim.data.qpos.subarray(0, 3)).s;
    const visitedSplits = new Set();
    let progress = 0,
      best = 0,
      lastProgress = 0;
    let maxLateral = 0,
      contactSteps = 0,
      bodySteps = 0,
      maxSpeed = 0;
    let jumps = 0,
      maxForwardCommand = 0;
    const trace = [],
      checkpoints = [],
      circuitSeconds = [];
    let status = 'timeout',
      seconds = 0,
      errorMessage;
    try {
      for (let tick = 0; tick <= Math.ceil(timeout / dt); tick++) {
        seconds = tick * dt;
        clock = room.startAt + seconds * 1000;
        game.tick();
        const p = sim.data.qpos.subarray(0, 3),
          q = sim.data.qpos.subarray(3, 7);
        const surface = surfaceAt(map, p[0], p[1], p[2]);
        const along = atDistance(map, surface.s);
        for (const split of map.splits ?? [])
          if (
            Math.abs(along.u - split.u) < split.halfU * 0.4 &&
            surface.lateral * (branch === 'right' ? -1 : 1) > 0.3
          )
            visitedSplits.add(
              `${Math.max(0, Math.floor(progress / track.length)) + 1}:${split.name}`,
            );
        const delta = surface.s - prevS;
        const advance = delta - Math.round(delta / track.length) * track.length;
        if (
          Math.abs(advance) > 12 * dt ||
          Math.abs(sim.data.time - seconds) > 1e-6
        ) {
          status = 'invalid-physics-continuity';
          break;
        }
        progress += advance;
        prevS = surface.s;
        if (progress > best + 0.2) {
          best = progress;
          lastProgress = seconds;
        }
        maxLateral = Math.max(maxLateral, Math.abs(surface.lateral));
        if (progress >= (circuitSeconds.length + 1) * track.length)
          circuitSeconds.push(seconds);
        const targetS = surface.s + (map.allowCrossings ? 0.45 : 0.65);
        const target = crossPoint(
          atDistance(map, targetS),
          branchOffset(map, targetS, branch),
        );
        const yaw = Math.atan2(
          2 * (q[0] * q[3] + q[1] * q[2]),
          1 - 2 * (q[2] ** 2 + q[3] ** 2),
        );
        const error = wrap(
          Math.atan2(target[1] - p[1], target[0] - p[0]) - yaw,
        );
        const command = {
          vx:
            seconds < 2 ? 0 : speed * Math.max(0.3, 1 - Math.abs(error) * 0.6),
          wz: seconds < 2 ? 0 : clamp(error * 2.5, -1.5, 1.5),
        };
        if ((player.effects.boost ?? 0) > clock)
          command.vx = boostedForwardSpeed(
            command.vx,
            command.wz,
            true,
            terrainTurboFactor(map, surface.s),
          );
        if (
          jumpApproach(map, surface.s, command.vx, 1.2) ||
          (sim.jumpController.tracking && sim.jumpController.age < 100)
        )
          command.vx = Math.min(command.vx, 1.4);
        if (
          seconds >= 2 &&
          // Automated driver chooses one stable takeoff per obstacle; manual
          // jump input is unconditional and would restart the reference here.
          sim.jumpController.readySteps >= 20 &&
          !sim.jumpController.tracking &&
          jumpApproach(
            map,
            surface.s,
            Math.max(command.vx, sim.snapshot().speed),
          ) &&
          sim.jump()
        )
          jumps++;
        maxForwardCommand = Math.max(maxForwardCommand, command.vx);
        // Same 20 Hz state cadence as the game; samples remain raw MuJoCo poses.
        if (tick % 5 === 0) {
          const s = sim.snapshot(true),
            next = player.next;
          game.handle(player, { type: 'state', p: s.p, q: s.q, j: s.j });
          if (next !== player.next)
            checkpoints.push({ checkpoint: next, lap: player.lap, seconds });
          trace.push({
            seconds,
            p: s.p,
            q: s.q,
            j: s.j,
            speed: s.speed,
            s: surface.s,
            progress,
            lateral: surface.lateral,
            height: p[2] - surface.z,
            command,
            next: player.next,
            lap: player.lap,
            bodyContact: sim.bodyContact,
          });
        }
        if (resets) {
          status = 'server-reset';
          break;
        }
        if (!surface.onRoad) {
          status = 'left-track';
          break;
        }
        if (player.lap >= laps && circuitSeconds.length >= laps) {
          const ordered =
            checkpoints.length >= map.points.length * laps &&
            checkpoints.every(
              (c, i) => c.checkpoint === (i + 1) % map.points.length,
            );
          status = !ordered
            ? 'missing-checkpoint'
            : visitedSplits.size !== (map.splits?.length ?? 0) * laps
              ? 'missed-branch'
              : 'passed';
          break;
        }
        if (seconds - lastProgress > 15) {
          status = 'stalled';
          break;
        }
        if (tick % 2000 === 0)
          console.log(
            `${mapId} ${branch} grid ${slot}: ${seconds.toFixed(0)} s, ${progress.toFixed(1)}/${(track.length * laps).toFixed(1)} m, checkpoint ${player.next}`,
          );
        await sim.step(policy, command);
        maxSpeed = Math.max(
          maxSpeed,
          Math.hypot(sim.data.qvel[0], sim.data.qvel[1]),
        );
        contactSteps += sim.data.ncon > 0 ? 1 : 0;
        bodySteps += sim.bodyContact ? 1 : 0;
      }
    } catch (e) {
      status = 'error';
      errorMessage = String(e);
    } finally {
      sim.dispose();
    }
    const result = {
      map: mapId,
      slot,
      branch,
      visitedSplits: [...visitedSplits],
      status,
      jumps,
      maxForwardCommand,
      error: errorMessage,
      seconds,
      circuitSeconds,
      trackLength: track.length,
      routeLengths: routeLengths(map),
      progress,
      serverLaps: player.lap,
      serverFinishSeconds:
        player.finished === null ? null : player.finished / 1000,
      checkpoints,
      resets,
      penalty: player.penalty,
      contactSteps,
      bodySteps,
      maxLateral,
      maxSpeed,
      final: trace.at(-1),
    };
    results.push(result);
    fs.writeFileSync(
      path.join(
        out,
        `${mapId}-${slot}${branch === 'centre' ? '' : '-' + branch}.json`,
      ),
      JSON.stringify({ result, trace }),
    );
    saveReport();
    console.log(
      `${status.toUpperCase()} ${mapId} ${branch} grid ${slot}: ${seconds.toFixed(2)} s, laps ${player.lap}, resets ${resets}, body contacts ${bodySteps}`,
    );
  }
} finally {
  await policy.dispose();
  saveReport();
}
console.log(`Report: ${path.join(out, 'report.json')}`);
if (
  results.length !== cases.length ||
  results.some((r) => r.status !== 'passed')
)
  process.exitCode = 1;
