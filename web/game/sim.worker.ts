/// <reference lib="webworker" />
import {
  downloadAsset,
  onDownloadProgress,
  prepareDownloads,
} from './asset-download';
import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web/wasm';
import { Policy, Simulation } from './physics.mjs';
import { BENI_JUMP_REFERENCE } from './jump-controller.mjs';
import { spawn, getMap, DEFAULT_MAP_ID } from './maps.mjs';
import { applyTuning } from './debug-tuning.mjs';
import {
  fingerprint,
  robotFiles,
  compiledModelKey,
  modelFromBinary,
} from './compiled-model.mjs';
import type { Command, WorkerCommand, WorkerEvent } from '../shared/protocol';
const bundledPolicy = {
  label: 'BENI · Jump and recovery',
  url: '/policies/beni.onnx',
};
const scope = self as unknown as DedicatedWorkerGlobalScope;
let simulation: Simulation | null = null,
  policy: Policy | null = null;
let command: Command = { vx: 0, wz: 0 },
  running = false,
  stopped = false;
// Debug single-stepping: policy intervals to integrate while paused.
let pendingSteps = 0;
const tasks: WorkerCommand[] = [];
let xml = '';
let assetsKey = '';
// Keep at most two compressed downloads; inactive MuJoCo models are released.
const packedTracks = new Map<string, ArrayBuffer>();
const unavailableTracks = new Set<string>();
const tunedMaps = new Set<string>();
let mj: Awaited<ReturnType<typeof loadMujoco>>;
let epoch = 0;
let mapGeneration = 0;
let mapAbort: AbortController | undefined;
const send = (x: WorkerEvent, transfer: Transferable[] = []) =>
  scope.postMessage(x, transfer);
async function checkedFetch(
  url: string,
  cache?: RequestCache,
  signal?: AbortSignal,
) {
  return downloadAsset(url, cache, signal);
}
async function loadSimulation(mapId: string, signal?: AbortSignal) {
  const key = compiledModelKey(xml, mapId, assetsKey, mj.mj_versionString());
  let model = null;
  if (!tunedMaps.has(mapId) && !unavailableTracks.has(key)) {
    try {
      let packed = packedTracks.get(key);
      if (!packed)
        packed =
          await // Every compilation input is part of this URL; cached bytes stay valid.
          (
            await checkedFetch(
              `/runtime/tracks/${key}.mjb.gz`,
              'force-cache',
              signal,
            )
          ).arrayBuffer();
      signal?.throwIfAborted();
      packedTracks.delete(key);
      packedTracks.set(key, packed);
      if (packedTracks.size > 2)
        packedTracks.delete(packedTracks.keys().next().value!);
      const stream = new Response(packed).body!.pipeThrough(
        new DecompressionStream('gzip'),
      );
      const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
      signal?.throwIfAborted();
      model = modelFromBinary(mj, bytes);
    } catch (error) {
      signal?.throwIfAborted();
      packedTracks.delete(key);
      unavailableTracks.add(key);
      console.info('Compiled track unavailable; rebuilding from XML.', error);
    }
  }
  signal?.throwIfAborted();
  return {
    simulation: new Simulation(
      mj,
      xml,
      mapId,
      true,
      BENI_JUMP_REFERENCE,
      model,
    ),
    source: model ? ('binary' as const) : ('compiled' as const),
  };
}
async function setPolicy(bytes: Uint8Array, label: string) {
  const next = await Policy.create(ort, bytes);
  const previous = policy;
  policy = next;
  await previous?.dispose();
  if (simulation) simulation.jumpController.reference = BENI_JUMP_REFERENCE;
  simulation?.reset(spawn(simulation.map));
  if (simulation) simulation.observationSize = next.observationSize;
  epoch++;
  send({
    type: 'policy',
    label,
    jumpSupported: true,
    inputs: next.observationSize,
    actions: 7,
    recurrent: !!next.hidden,
  });
}
async function initialize() {
  try {
    const configUrl = scope.name.startsWith('multiplayer')
      ? '/api/config'
      : '/game-config.json';
    onDownloadProgress((text) => send({ type: 'status', text }));
    send({ type: 'status', text: 'Checking cached assets…' });
    await prepareDownloads(DEFAULT_MAP_ID);
    const robotDownload = Promise.all([
      checkedFetch('/robot/manifest.json').then(
        (r) => r.json() as Promise<string[]>,
      ),
      checkedFetch('/robot/beni.xml').then((r) => r.text()),
    ]).then(async ([manifest, source]) => ({
      source,
      files: await Promise.all(
        robotFiles(source, manifest).map(async (file: string) => ({
          file,
          bytes: new Uint8Array(
            await (await checkedFetch('/robot/' + file)).arrayBuffer(),
          ),
        })),
      ),
    }));
    const trackDownload = checkedFetch('/runtime/tracks/manifest.json')
      .then(
        (r) => r.json() as Promise<{ tracks: { id: string; key: string }[] }>,
      )
      .then(async (manifest) => {
        const entry = manifest.tracks.find(
          (item: { id: string }) => item.id === DEFAULT_MAP_ID,
        );
        if (entry)
          packedTracks.set(
            entry.key,
            await (
              await checkedFetch(
                `/runtime/tracks/${entry.key}.mjb.gz`,
                'force-cache',
              )
            ).arrayBuffer(),
          );
      })
      .catch((error) => console.info('Course prefetch unavailable.', error));
    const [config, wasmBinary, ortBinary, policyBytes, robot] =
      await Promise.all([
        checkedFetch(configUrl).then(
          (r) => r.json() as Promise<{ multiplayerEnabled?: boolean }>,
        ),
        checkedFetch('/runtime/mujoco.wasm').then((r) => r.arrayBuffer()),
        checkedFetch('/runtime/ort-wasm-simd-threaded.wasm').then((r) =>
          r.arrayBuffer(),
        ),
        checkedFetch(bundledPolicy.url).then((r) => r.arrayBuffer()),
        robotDownload,
        trackDownload,
      ]);
    send({
      type: 'configuration',
      multiplayerEnabled: config.multiplayerEnabled === true,
    });
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = new URL('/runtime/', scope.location.origin).href;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmBinary = ortBinary;
    send({ type: 'status', text: 'Initializing physics and policy…' });
    const [, engine] = await Promise.all([
      setPolicy(new Uint8Array(policyBytes), bundledPolicy.label),
      loadMujoco({
        wasmBinary: new Uint8Array(wasmBinary),
        locateFile: (p: string) =>
          p.endsWith('.wasm') ? '/runtime/mujoco.wasm' : p,
      }),
    ]);
    mj = engine;
    mj.FS.mkdir('/robot');
    mj.FS.mkdir('/robot/meshes');
    mj.FS.mkdir('/robot/textures');
    xml = robot.source;
    const assets = robot.files.map(({ file, bytes }) => {
      mj.FS.writeFile('/robot/' + file, bytes);
      return [file, fingerprint(bytes)];
    });
    assetsKey = fingerprint(...assets.flat());
    send({ type: 'status', text: 'Preparing course…' });
    simulation = (await loadSimulation(DEFAULT_MAP_ID)).simulation;
    if (policy) simulation.observationSize = (policy as Policy).observationSize;
    simulation.reset(spawn(simulation.map));
    send({ type: 'model', geoms: simulation.visualModel() });
    send({ type: 'ready' });
    await loop();
  } catch (e) {
    send({ type: 'fatal', text: String(e instanceof Error ? e.message : e) });
  }
}
async function loop() {
  let deadline = performance.now(),
    nextFrame = 0;
  while (!stopped) {
    const batch = tasks.splice(0);
    const batchGeneration = mapGeneration;
    for (const task of batch) {
      if (stopped || (task.type === 'map' && batchGeneration !== mapGeneration))
        continue;
      try {
        if (task.type === 'map') {
          // Debug tuning must land before the collision prisms are regenerated.
          if (task.tuning) {
            applyTuning(task.map, task.tuning);
            tunedMaps.add(task.map);
          }
          const current = mapGeneration;
          mapAbort = new AbortController();
          const signal = mapAbort.signal;
          const previous = simulation;
          let source: 'reuse' | 'binary' | 'compiled' = 'reuse';
          // Joining/rejoining the current course only needs a fresh robot and
          // policy state. Keep the compiled collision model unless it changed.
          if (
            !simulation ||
            simulation.map.id !== task.map ||
            task.tuning !== undefined
          ) {
            simulation?.dispose();
            simulation = null;
            const key = compiledModelKey(
              xml,
              task.map,
              assetsKey,
              mj.mj_versionString(),
            );
            const paths =
              packedTracks.has(key) ||
              tunedMaps.has(task.map) ||
              unavailableTracks.has(key)
                ? []
                : [`/runtime/tracks/${key}.mjb.gz`];
            await prepareDownloads(task.map, paths, signal);
            const loaded = await loadSimulation(task.map, signal);
            if (stopped || current !== mapGeneration) {
              loaded.simulation.dispose();
              continue;
            }
            simulation = loaded.simulation;
            source = loaded.source;
          }
          simulation.reset(task.spawn ?? spawn(getMap(task.map)));
          epoch++;
          policy?.reset();
          if (simulation !== previous)
            send({ type: 'model', geoms: simulation.visualModel() });
          send({ type: 'mapReady', map: task.map, source });
        }
        if (task.type === 'reset') {
          simulation?.reset(task.spawn);
          epoch++;
          policy?.reset();
          command = { vx: 0, wz: 0 };
        }
        if (task.type === 'policy') {
          send({ type: 'status', text: 'Checking custom policy…' });
          await setPolicy(new Uint8Array(task.bytes), task.label);
          send({ type: 'ready' });
        }
        if (task.type === 'default') {
          await setPolicy(
            new Uint8Array(
              await (await checkedFetch(bundledPolicy.url)).arrayBuffer(),
            ),
            bundledPolicy.label,
          );
          send({ type: 'ready' });
        }
        if (task.type === 'jump' && running && simulation) {
          const accepted = simulation.jump();
          send({
            type: 'jumpResult',
            accepted,
          });
        }
      } catch (e) {
        if (stopped || (task.type === 'map' && mapAbort?.signal.aborted))
          continue;
        send({
          type: 'error',
          text: String(e instanceof Error ? e.message : e),
        });
        send({ type: 'ready' });
      }
    }
    if (simulation && policy) {
      if (running || pendingSteps > 0) {
        if (!running) pendingSteps--;
        try {
          await simulation.step(policy, command);
        } catch (e) {
          running = false;
          pendingSteps = 0;
          send({
            type: 'fatal',
            text: String(e instanceof Error ? e.message : e),
          });
        }
      }
      const now = performance.now();
      if (now >= nextFrame) {
        const frame = simulation.snapshot(true);
        send(
          {
            type: 'frame',
            ...frame,
            running,
            epoch,
            wallTime: performance.timeOrigin + now,
          },
          [frame.geomPos.buffer, frame.geomMat.buffer],
        );
        nextFrame += 1000 / 60;
        if (nextFrame <= now) nextFrame = now + 1000 / 60;
      }
    }
    deadline += 10;
    const wait = deadline - performance.now();
    if (wait < -100) deadline = performance.now();
    await new Promise((r) => setTimeout(r, Math.max(0, wait)));
  }
  simulation?.dispose();
  await policy?.dispose();
}
scope.onmessage = ({ data }: MessageEvent<WorkerCommand>) => {
  if (data.type === 'command') command = data.command;
  else if (data.type === 'running') {
    running = data.value;
    if (running) pendingSteps = 0;
  } else if (data.type === 'step') pendingSteps += data.count ?? 1;
  else if (data.type === 'stop') {
    stopped = true;
    mapAbort?.abort();
  } else if (data.type === 'map') {
    mapGeneration++;
    mapAbort?.abort();
    for (let i = tasks.length - 1; i >= 0; i--)
      if (tasks[i].type === 'map') tasks.splice(i, 1);
    tasks.push(data);
  } else tasks.push(data);
};
void initialize();
