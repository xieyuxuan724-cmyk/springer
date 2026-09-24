import { assetPath } from '../../scripts/browser-assets.mjs';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import config from '../../game/robot-config.json' with { type: 'json' };
import path from 'node:path';
import { once } from 'node:events';
import { WebSocketServer, WebSocket } from 'ws';
import load from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web';
import { Matrix4, Quaternion } from 'three';
import { Simulation, Policy } from '../../game/physics.mjs';
import {
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

const count = Number(process.env.DEMO_PLAYERS ?? 3),
  seconds = Number(process.env.DEMO_SECONDS ?? 45);
const tour = process.env.DEMO_TOUR === '1';
const out = path.resolve(process.env.DEMO_OUTPUT ?? 'outputs/physics-demo');
fs.mkdirSync(out, { recursive: true });
ort.env.wasm.numThreads = 1;
const mj = await load();
mj.FS.mkdir('/robot');
mj.FS.mkdir('/robot/meshes');
mj.FS.mkdir('/robot/textures');
for (const f of JSON.parse(fs.readFileSync(assetPath('robot/manifest.json'))))
  mj.FS.writeFile('/robot/' + f, fs.readFileSync(assetPath('robot/' + f)));
const xml = fs.readFileSync(assetPath('robot/beni.xml'), 'utf8'),
  bytes = new Uint8Array(fs.readFileSync(assetPath('policies/beni.onnx')));
let clock = 1000000;
const seed = Number(process.env.DEMO_SEED ?? 20260914);
const game = new RaceService({ now: () => clock, random: seededRandom(seed) }),
  server = new WebSocketServer({ port: 0, host: '127.0.0.1' });
await once(server, 'listening');
server.on('connection', (ws) => {
  let player;
  ws.on('message', (data) => {
    const m = JSON.parse(data);
    try {
      if (m.type === 'hello')
        player = game.connect(m.name, null, (x) => ws.send(JSON.stringify(x)));
      else game.handle(player, m);
      ws.send(JSON.stringify({ type: 'ack', seq: m.seq }));
    } catch (e) {
      ws.send(JSON.stringify({ type: 'ack', seq: m.seq, error: String(e) }));
    }
  });
  ws.on('close', () => {
    if (player) game.disconnect(player);
  });
});
const bots = [],
  frames = [],
  map = getMap(process.env.DEMO_MAP ?? 'harbor'),
  track = getTrack(map);
const names = [
  'Captain Lime',
  'Blue Comet',
  'Orange Rush',
  'Violet Bolt',
  'Mint Racer',
  'Cherry Dash',
];
try {
  for (let i = 0; i < count; i++) {
    const ws = new WebSocket(`ws://127.0.0.1:${server.address().port}`);
    await once(ws, 'open');
    const waits = new Map();
    let seq = 0;
    const bot = {
      ws,
      sim: new Simulation(mj, xml, map.id),
      policy: await Policy.create(ort, bytes),
      id: null,
      resets: 0,
      jumps: 0,
      contacts: 0,
      lastJump: -10,
      jumpRequests: 0,
      itemsUsed: 0,
      maxForwardCommand: 0,
      maxSpeed: 0,
      maxClearance: 0,
      bodySteps: 0,
      branch: i % 2 ? 'right' : 'left',
      progress: 0,
      previousS: 0,
      visitedSplit: false,
      circuitSeconds: null,
    };
    ws.on('message', (data) => {
      const m = JSON.parse(data);
      if (m.type === 'welcome') bot.id = m.id;
      if (m.type === 'ack') {
        const w = waits.get(m.seq);
        waits.delete(m.seq);
        m.error ? w.reject(Error(m.error)) : w.resolve();
      }
      if (m.type === 'reset') bot.resets++;
    });
    bot.send = (m) =>
      new Promise((resolve, reject) => {
        waits.set(++seq, { resolve, reject });
        ws.send(JSON.stringify({ ...m, seq }));
      });
    await bot.send({ type: 'hello', name: names[i] });
    bots.push(bot);
    if (i === 0) await bot.send({ type: 'create', map: map.id });
    else
      await bot.send({ type: 'join', code: game.players.get(bots[0].id).room });
    await bot.send({ type: 'ready', ready: true });
    console.log('Physics player ready', i + 1);
  }
  const room = game.rooms.get(game.players.get(bots[0].id).room);
  await bots[0].send({ type: 'start' });
  for (let i = 0; i < count; i++) {
    bots[i].sim.reset(spawn(map, i));
    bots[i].policy.reset();
    bots[i].resets = 0;
    bots[i].previousS = surfaceAt(
      map,
      ...bots[i].sim.data.qpos.subarray(0, 3),
    ).s;
  }
  // Drain each connection's countdown reset before auditing in-race resets.
  await Promise.all(
    bots.map(async (b) => {
      const s = b.sim.snapshot(true);
      await b.send({ type: 'state', p: s.p, q: s.q, j: s.j });
      b.resets = 0;
    }),
  );
  let completedAt = null;
  for (let tick = 0; tick < Math.ceil(seconds * 100); tick++) {
    clock = 1000000 + tick * 10;
    game.tick();
    if (room.phase === 'racing') {
      await Promise.all(
        bots.map(async (bot, i) => {
          const sim = bot.sim,
            snap = sim.snapshot(true),
            p = snap.p,
            q = snap.q;
          const surface = surfaceAt(map, p[0], p[1], p[2]);
          const targetS =
            surface.s + (tour ? (map.allowCrossings ? 0.45 : 0.65) : 0.55);
          const target = atDistance(map, targetS);
          const yaw = Math.atan2(
            2 * (q[0] * q[3] + q[1] * q[2]),
            1 - 2 * (q[2] ** 2 + q[3] ** 2),
          );
          // Split courses require a branch target even outside full-course tours.
          // Fixed centerline offsets would steer into the divider at the fork.
          const lane = crossPoint(
            target,
            tour || map.splits?.length
              ? branchOffset(map, targetS, bot.branch)
              : ((i % 3) - 1) * 0.42,
          );
          const desired = Math.atan2(lane[1] - p[1], lane[0] - p[0]);
          const error = Math.atan2(
            Math.sin(desired - yaw),
            Math.cos(desired - yaw),
          );
          const t = (clock - room.startAt) / 1000;
          let vx =
            t < (tour ? 2 : 2.2)
              ? 0
              : tour
                ? (1.5 - i * 0.06) * Math.max(0.3, 1 - Math.abs(error) * 0.6)
                : (0.92 + i * 0.025) * Math.max(0.5, 1 - Math.abs(error) * 0.4);
          const player = game.players.get(bot.id);
          if ((player.effects.boost ?? 0) > clock)
            vx = boostedForwardSpeed(
              vx,
              error * 2.5,
              true,
              terrainTurboFactor(map, surface.s),
            );
          if (
            tour &&
            (jumpApproach(map, surface.s, vx, 1.2) ||
              (sim.jumpController.tracking && sim.jumpController.age < 100))
          )
            vx = Math.min(vx, 1.4);
          const firstJump = t >= 0.8 && t < 1.2 && bot.jumps === 0;
          const nearHazard = map.hazards.some((h) => {
            const hf = track.frames.findIndex(
              (f) => Math.abs(f.u - h.u) < 1e-8,
            );
            const distance =
              (track.lengths[hf] - surface.s + track.length) % track.length;
            return distance > 0.1 && distance < 0.45;
          });
          if (
            (tour
              ? jumpApproach(map, surface.s, Math.max(vx, snap.speed))
              : firstJump || nearHazard) &&
            t - bot.lastJump > 1 &&
            sim.jumpController.readySteps >= 20 &&
            !sim.jumpController.tracking
          ) {
            bot.jumpRequests++;
            if (sim.jump()) {
              bot.lastJump = t;
              bot.jumps++;
            }
          }
          await sim.step(bot.policy, {
            vx,
            wz:
              t < (tour ? 2 : 2.2)
                ? 0
                : Math.max(
                    tour ? -1.5 : -1.8,
                    Math.min(tour ? 1.5 : 1.8, error * 2.5),
                  ),
          });
          bot.maxForwardCommand = Math.max(bot.maxForwardCommand, vx);
          bot.maxSpeed = Math.max(bot.maxSpeed, sim.snapshot().speed);
          bot.contacts += sim.data.ncon;
          bot.bodySteps += sim.bodyContact ? 1 : 0;
          const after = surfaceAt(map, ...sim.data.qpos.subarray(0, 3));
          const delta = after.s - bot.previousS;
          const advance =
            delta - Math.round(delta / track.length) * track.length;
          if (tour && (Math.abs(advance) > 0.12 || !after.onRoad || bot.resets))
            throw Error(
              `Invalid physical continuity or off-track driver: ${names[i]}`,
            );
          bot.progress += advance;
          bot.previousS = after.s;
          const along = atDistance(map, after.s);
          for (const split of map.splits ?? [])
            if (
              Math.abs(along.u - split.u) < split.halfU * 0.4 &&
              after.lateral * (bot.branch === 'right' ? -1 : 1) > 0.3
            )
              bot.visitedSplit = true;
          if (bot.progress >= track.length && bot.circuitSeconds === null)
            bot.circuitSeconds = t;
          bot.maxClearance = Math.max(
            bot.maxClearance,
            Math.min(...sim.jumpMeasurement().clearance),
          );
        }),
      );
    }
    if (tick % 5 === 0) {
      await Promise.all(
        bots.map(async (b) => {
          const s = b.sim.snapshot(true);
          await b.send({ type: 'state', p: s.p, q: s.q, j: s.j });
        }),
      );
      game.tick();
      const state = game.view(room);
      if (tour && bots.every((b) => b.circuitSeconds !== null)) {
        completedAt ??= clock;
        state.tourComplete = true;
      }
      for (const p of state.players) {
        const bot = bots.find((b) => b.id === p.id),
          snap = bot.sim.snapshot(true);
        p.visual = snap.geomIds.map((id, i) => {
          const a = snap.geomMat,
            k = i * 9,
            m = new Matrix4();
          m.set(
            a[k],
            a[k + 1],
            a[k + 2],
            0,
            a[k + 3],
            a[k + 4],
            a[k + 5],
            0,
            a[k + 6],
            a[k + 7],
            a[k + 8],
            0,
            0,
            0,
            0,
            1,
          );
          const q = new Quaternion().setFromRotationMatrix(m);
          return {
            id,
            p: Array.from(snap.geomPos.slice(i * 3, i * 3 + 3)),
            q: [q.w, q.x, q.y, q.z],
          };
        });
      }
      frames.push(structuredClone(state));
      // Keep the finish visible without recording a partial second circuit.
      if (completedAt !== null && clock - completedAt >= 2000) break;
    }
    if (tick % 500 === 0)
      console.log(
        'Simulated seconds',
        tick / 100,
        bots.map((b) => {
          const p = b.sim.snapshot(true).p;
          return {
            s: +surfaceAt(map, p[0], p[1], p[2]).s.toFixed(1),
            z: +p[2].toFixed(2),
            jumps: b.jumps,
          };
        }),
      );
  }
  const report = {
    seed,
    seconds: (frames.at(-1).now - frames[0].now) / 1000,
    durationLimit: seconds,
    count,
    map: map.id,
    routeLengths: routeLengths(map),
    tour,
    mapsSha256: createHash('sha256')
      .update(fs.readFileSync('game/maps.mjs'))
      .digest('hex'),
    terrainReliefSha256: createHash('sha256')
      .update(fs.readFileSync('game/terrain-relief.mjs'))
      .digest('hex'),
    physics: 'MuJoCo WASM',
    policy: 'BENI-Jump-Flat/policy.onnx',
    policySha256: config.policySha256,
    robot: 'beni.xml',
    dt: config.physicsDt,
    controlDt: 0.01,
    manualPositionUpdatesAfterSpawn: 0,
    interRobotCollisions: false,
    players: bots.map((b) => ({
      name: game.players.get(b.id).name,
      branch: b.branch,
      progress: b.progress,
      circuitSeconds: b.circuitSeconds,
      laps: game.players.get(b.id).lap,
      bodySteps: b.bodySteps,
      visitedSplit: b.visitedSplit,
      contacts: b.contacts,
      jumps: b.jumps,
      jumpRequests: b.jumpRequests,
      itemsUsed: b.itemsUsed,
      gems: game.players.get(b.id).gems,
      pickups: game.players.get(b.id).lastPickup?.seq ?? 0,
      maxForwardCommand: b.maxForwardCommand,
      maxSpeed: b.maxSpeed,
      maxWheelClearance: b.maxClearance,
      resets: b.resets,
      final: b.sim.snapshot(true).p,
    })),
  };
  fs.writeFileSync(path.join(out, 'replay.json'), JSON.stringify(frames));
  fs.writeFileSync(
    path.join(out, 'verification.json'),
    JSON.stringify(report, null, 2),
  );
  console.log('CAPTURED', frames.length, report);
  if (
    tour &&
    report.players.some(
      (p) =>
        p.circuitSeconds === null ||
        p.laps < 1 ||
        p.resets ||
        p.bodySteps ||
        (map.splits?.length && !p.visitedSplit),
    )
  )
    throw Error(
      'Course tour did not complete a clean physical lap for every driver; inspect verification.json',
    );
} finally {
  for (const b of bots) {
    b.ws.close();
    b.sim.dispose();
    await b.policy.dispose();
  }
  server.close();
}
