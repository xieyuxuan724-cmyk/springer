import { Vector3 } from 'three';
import { sampleReplay } from './replay.mjs';
import { Viewer } from '../../game/viewer';
import {
  getMap,
  getTrack,
  atDistance,
  crossPoint,
  surfaceAt,
} from '../../game/maps.mjs';
import { getSkin } from '../../game/skins.mjs';

const physical = !new URLSearchParams(location.search).has('kinematic');
const el = (id: string) => document.getElementById(id)!;
const viewer = new Viewer(el('scene'), 'high');
const courseTour = new URLSearchParams(location.search).get('tour') === '1';
const loadedReplay: any = physical
  ? await (
      await fetch(
        new URLSearchParams(location.search).get('replay') ??
          '/outputs/physics-demo/replay.json',
      )
    ).json()
  : null;
let map = getMap(loadedReplay?.[0]?.map ?? 'harbor'),
  track = getTrack(map);
viewer.setMap(map.id);
let room: any,
  started = false,
  finishedAt = 0,
  focus = 0;
const bots: any[] = [];
const recording: any[] = [];
let replay = false,
  directorId = 0;
const names = [
  'Captain Lime',
  'Blue Comet',
  'Orange Rush',
  'Violet Bolt',
  'Mint Racer',
  'Cherry Dash',
];
const worker = new Worker(
  new URL('../../game/sim.worker.ts', import.meta.url),
  { type: 'module', name: 'multiplayer' },
);
const ready = new Promise<void>((resolve, reject) => {
  worker.onmessage = ({ data }) => {
    if (data.type === 'model') {
      viewer.setModel(data.geoms);
      viewer.robot.visible = false;
    }
    if (data.type === 'ready') worker.postMessage({ type: 'map', map: map.id });
    if (data.type === 'mapReady' && data.map === map.id) {
      worker.terminate();
      resolve();
    }
    if (data.type === 'error') reject(Error(data.text));
    if (data.type === 'fatal') reject(Error(data.text));
  };
});
function drawRoom(r: any) {
  if (started && !replay) recording.push(structuredClone(r));
  room = r;
  viewer.updateRoom(r, 'demo-spectator');
  for (const remote of viewer.remotes.values())
    remote.userData.label.scale.set(0.38, 0.12, 1);
  const sorted = [...r.players].sort(
    (a, b) =>
      (a.finished ?? Infinity) - (b.finished ?? Infinity) ||
      b.lap - a.lap ||
      (b.next || 10) - (a.next || 10),
  );
  const followed =
    r.players.find((p: any) => p.id === bots[focus]?.id) ?? sorted[0];
  el('course-name').textContent = map.name;
  el('race-position').textContent =
    `${sorted.findIndex((p: any) => p.id === followed?.id) + 1}/${sorted.length}`;
  el('race-lap').textContent = courseTour
    ? r.tourComplete
      ? 'Everyone completed a lap'
      : 'Course tour'
    : `${Math.min((followed?.lap ?? 0) + 1, 2)}/2 laps`;
  el('race-score').textContent = `◆ ${followed?.gems ?? 0}`;
  el('board').replaceChildren(
    ...sorted.map((p: any, i: number) => {
      const row = document.createElement('div');
      row.className = 'row';
      for (const [cls, text] of [
        ['rank', String(i + 1)],
        ['chip', ''],
        ['name', p.name],
        [
          'lap',
          p.finished != null
            ? (p.finished / 1000).toFixed(1) + 's'
            : `${Math.min(2, p.lap + 1)}/2 laps · ◆ ${p.gems ?? 0}`,
        ],
      ]) {
        const span = document.createElement('span');
        span.className = cls;
        span.textContent = text;
        if (cls === 'chip') span.style.background = getSkin(p.skin).primary;
        row.appendChild(span);
      }
      return row;
    }),
  );
  el('event').textContent =
    r.events.at(-1)?.text ??
    `${r.players.length} drivers connected and ready to race`;
  if (courseTour)
    el('event').textContent = r.players.every((p: any) => p.lap >= 1)
      ? `${r.players.length} drivers have completed their first lap`
      : map.splits?.length
        ? 'Left · Garden bridge  |  Right · Candy tunnel'
        : 'Linked S-bends · Cloudtop loop · Elevation changes';
  el('phase').textContent =
    r.phase === 'racing'
      ? 'RACE IN PROGRESS'
      : r.phase === 'finished'
        ? 'FINISHED'
        : 'READY TO RACE';
  if (r.phase === 'finished' && !finishedAt) {
    finishedAt = performance.now();
    el('winner').textContent = `${sorted[0].name} wins`;
    el('finish').style.display = 'block';
    document.body.dataset.finished = 'true';
  }
}
async function connect(name: string) {
  const ws = new WebSocket(
    `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`,
  );
  const bot: any = {
    ws,
    name,
    s: 0,
    send: (m: any) => ws.send(JSON.stringify(m)),
  };
  await new Promise<void>((resolve, reject) => {
    ws.onopen = () => bot.send({ type: 'hello', name });
    ws.onerror = () => reject(Error('Demo connection failed'));
    ws.onmessage = ({ data }) => {
      const m = JSON.parse(data);
      if (m.type === 'welcome') {
        bot.id = m.id;
        resolve();
      }
      if (m.type === 'joined') {
        bot.joined = true;
        bot.s = surfaceAt(map, m.spawn.p[0], m.spawn.p[1])!.s - track.length;
      }
      if (m.type === 'room' && bots[0]?.id === bot.id) drawRoom(m.room);
      if (m.type === 'reset')
        bot.s = surfaceAt(map, m.spawn.p[0], m.spawn.p[1])!.s;
      if (m.type === 'error') {
        document.body.dataset.error = m.text;
        console.error(m.text);
      }
    };
  });
  bots.push(bot);
  return bot;
}
const until = async (predicate: () => boolean) => {
  const end = performance.now() + 20000;
  while (!predicate()) {
    if (performance.now() > end) throw Error('Demo setup timed out');
    await new Promise((r) => setTimeout(r, 50));
  }
};
await ready;
let host: any;
if (!physical) {
  host = await connect(names[0]);
  host.send({ type: 'create', map: 'harbor' });
  await until(() => !!room);
  for (const name of names.slice(1)) {
    const bot = await connect(name);
    bot.send({ type: 'join', code: room.code });
    await until(() => bot.joined);
  }
  for (const bot of bots) bot.send({ type: 'ready', ready: true });
  await until(
    () => room.players.length === 6 && room.players.every((p: any) => p.ready),
  );
} else {
  recording.push(...loadedReplay);
  map = getMap(recording[0].map);
  track = getTrack(map);
  viewer.setMap(map.id);
  bots.push(...recording[0].players.map((p: any) => ({ id: p.id })));
  if (courseTour) {
    focus = bots.length - 1;
    viewer.debug.freeCamera = true;
    // Physics replay poses are world-space; remote roots are zero during drawing.
    viewer.effectPosition = (id, target) => {
      const p = room?.players.find((p: any) => p.id === id);
      if (!p) return false;
      target.fromArray(p.p);
      return true;
    };
  }
  replay = true;
  started = true;
  cancelAnimationFrame(viewer.animation);
  viewer.started = 0;
  viewer.lastRender = 0;
  drawRoom(recording[0]);
  document.querySelector('.pill')!.textContent =
    `${bots.length} AI drivers · MuJoCo + ONNX`;
  document.querySelector('footer .title')!.textContent =
    `One robot, ${bots.length} personalities.`;
  document.querySelector('footer .eyebrow')!.textContent =
    `${map.name} / Real-physics multiplayer demo`;
  document.querySelector('footer .detail')!.textContent =
    'Latest BENI · Current jump policy · Real terrain collisions';
  document.querySelector('#finish p')!.textContent =
    `Two-lap race · ${bots.length} AI drivers`;
}
await until(() =>
  [...viewer.skinMaps.values()].every(
    (t) => (t.image as HTMLImageElement | undefined)?.complete === true,
  ),
);
el('loading').remove();
document.body.dataset.ready = 'true';
let lastTick = performance.now();
const driverTimer = setInterval(() => {
  const now = performance.now(),
    dt = Math.min(0.1, (now - lastTick) / 1000);
  lastTick = now;
  if (replay || room?.phase !== 'racing') return;
  bots.forEach((bot, i) => {
    const p = room.players.find((p: any) => p.id === bot.id);
    if (!p || p.finished !== null) return;
    const clock = Date.now() + (room.now - Date.now());
    const speed =
      (2.3 + 0.09 * i + 0.22 * Math.sin(now / 2600 + i)) *
      ((p.effects?.boost ?? 0) > clock ? 1.5 : 1);
    bot.s += speed * dt;
    const f = atDistance(map, bot.s);
    const lateral = 0.24 * Math.sin(bot.s * 0.45 + i * 1.9);
    // Kinematic mock driver: score/boost/skin/ranking messages use the real server.
    const lift = f.hazard
      ? 0.14 *
        Math.max(
          0,
          Math.cos((((f.u - f.hazard.u) / f.hazard.halfU) * Math.PI) / 2),
        )
      : 0;
    bot.send({
      type: 'state',
      p: crossPoint(f, lateral, 0.115 + lift),
      q: [Math.cos(f.yaw / 2), 0, 0, Math.sin(f.yaw / 2)],
      j: [
        0.3,
        0.3,
        0.5,
        0.5,
        bot.s / 0.038,
        bot.s / 0.038,
        0.12 * Math.sin(now / 1000 + i),
      ],
    });
  });
}, 50);
function director(now = Date.now()) {
  if (room) {
    const elapsed =
      room.phase === 'finished'
        ? Math.max(...room.players.map((p: any) => p.finished ?? 0)) / 1000
        : room.startAt
          ? Math.max(0, (now - room.startAt) / 1000)
          : 0;
    viewer.world.traverse((object: any) => {
      if (object.isSprite)
        object.visible =
          !courseTour &&
          object.position.distanceTo(viewer.camera.position) > 2.5;
    });
    el('clock').textContent =
      `${String(Math.floor(elapsed / 60)).padStart(2, '0')}:${(elapsed % 60).toFixed(1).padStart(4, '0')}`;
    el('countdown').textContent =
      room.phase === 'countdown'
        ? String(Math.max(1, Math.ceil((room.startAt - now) / 1000)))
        : room.phase === 'racing' && elapsed < 1
          ? 'GO!'
          : '';
    const target = room.players.find((p: any) => p.id === bots[focus]?.id);
    viewer.preview = !started;
    if (target && started) {
      viewer.frame = { p: target.p, q: target.q };
    }
  }
  if (!replay) directorId = requestAnimationFrame(() => director());
}
director();
let lastTourFrame: number | null = null;
// Explicit record control for the local harness, never enabled in the normal app.
(window as any).demo = {
  start() {
    started = true;
    recording.push({ ...structuredClone(room), now: Date.now() });
    host.send({ type: 'start' });
  },
  state() {
    return {
      phase: room?.phase,
      players: room?.players,
      code: room?.code,
      events: room?.events,
      effects: viewer.effects.stats(),
      pickupLabels: viewer.effects.labels.filter((l) => l.sprite.visible)
        .length,
    };
  },
  focus(index: number) {
    focus = index % bots.length;
  },
  beginReplay() {
    replay = true;
    clearInterval(driverTimer);
    cancelAnimationFrame(directorId);
    cancelAnimationFrame(viewer.animation);
    for (const bot of bots) {
      if (bot.ws) {
        bot.ws.onmessage = null;
        bot.ws.close();
      }
    }
    viewer.started = 0;
    viewer.lastRender = 0;
    lastTourFrame = null;
    return recording;
  },
  renderFrame(time: number) {
    const r = sampleReplay(recording, time);
    drawRoom(r);
    el('finish').style.display = r.phase === 'finished' ? 'block' : 'none';
    director(time);
    // The same interpolated pose drives the camera and robot, with no second filter.
    for (const remote of viewer.remotes.values()) {
      const p = remote.userData.target;
      remote.position.fromArray(p.p);
      remote.quaternion.set(p.q[1], p.q[2], p.q[3], p.q[0]);
    }
    viewer.renderAt(time - recording[0].now, Date.now(), () => {
      if (!physical) return;
      if (courseTour) {
        const seconds = (time - recording[0].now) / 1000;
        const racers = r.players;
        const last =
          racers.find((p: any) => p.id === bots[focus]?.id) ?? racers[0];
        let target: Vector3, camera: Vector3;
        const overview = seconds < 3;
        if (overview) {
          const b = track.bounds,
            radius = Math.max(b.maxX - b.minX, b.maxY - b.minY) * 0.7;
          target = new Vector3(
            (b.minX + b.maxX) / 2,
            (b.minY + b.maxY) / 2,
            0.6,
          );
          camera = target
            .clone()
            .add(
              new Vector3(
                radius * Math.cos(seconds * 0.06),
                -radius * 0.85,
                radius * 0.9,
              ),
            );
        } else {
          const q = last.q;
          const yaw = Math.atan2(
            2 * (q[0] * q[3] + q[1] * q[2]),
            1 - 2 * (q[2] ** 2 + q[3] ** 2),
          );
          target = new Vector3(
            last.p[0],
            last.p[1],
            Math.max(last.p[2], -0.1) + 0.12,
          );
          camera = target
            .clone()
            .add(new Vector3(-1.1 * Math.cos(yaw), -1.1 * Math.sin(yaw), 0.38));
        }
        // Cut from the establishing shot to the chase camera before launch.
        // Following one robot also keeps split-route cameras out of tunnel roofs.
        if (lastTourFrame === null || (seconds >= 3 && lastTourFrame < 3)) {
          viewer.camera.position.copy(camera);
          viewer.target.copy(target);
        } else {
          const dt = Math.max(0, Math.min(0.1, seconds - lastTourFrame));
          viewer.camera.position.lerp(camera, 1 - Math.pow(1 - 0.09, dt * 60));
          viewer.target.lerp(target, 1 - Math.pow(1 - 0.16, dt * 60));
        }
        lastTourFrame = seconds;
        viewer.camera.lookAt(viewer.target);
      }
      for (const remote of viewer.remotes.values()) {
        const p = remote.userData.target;
        remote.position.set(0, 0, 0);
        remote.quaternion.identity();
        for (const mesh of remote.children) {
          if (!mesh.userData.geom) continue;
          const pose = p.visual.find(
            (v: any) => v.id === mesh.userData.geom.id,
          );
          mesh.position.fromArray(pose.p);
          mesh.quaternion.set(pose.q[1], pose.q[2], pose.q[3], pose.q[0]);
        }
        remote.userData.label.position.set(p.p[0], p.p[1], p.p[2] + 0.34);
      }
    });
  },
  stop() {
    clearInterval(driverTimer);
    cancelAnimationFrame(directorId);
    for (const bot of bots) bot.ws?.close();
    viewer.dispose();
  },
};
