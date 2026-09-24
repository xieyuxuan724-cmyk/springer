import { assetPath } from './browser-assets.mjs';
import {
  readFile,
  writeFile,
  mkdir,
  readdir,
  stat,
  rename,
  unlink,
} from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import loadMujoco from '@mujoco/mujoco';
import { MAPS } from '../game/maps.mjs';
import { Simulation } from '../game/physics.mjs';
import {
  fingerprint,
  robotFiles,
  compiledModelKey,
} from '../game/compiled-model.mjs';

const output = 'public/runtime/tracks';
await mkdir(output, { recursive: true });
const xml = await readFile(assetPath('robot/beni.xml'), 'utf8');
const manifest = JSON.parse(
  await readFile(assetPath('robot/manifest.json'), 'utf8'),
);
const assets = await Promise.all(
  robotFiles(xml, manifest).map(async (name) => ({
    name,
    bytes: await readFile(assetPath('robot/' + name)),
  })),
);
const assetsKey = fingerprint(
  ...assets.flatMap(({ name, bytes }) => [name, fingerprint(bytes)]),
);
const { version } = JSON.parse(
  await readFile('node_modules/@mujoco/mujoco/package.json', 'utf8'),
);
const entries = MAPS.map(({ id }) => ({
  id,
  key: compiledModelKey(xml, id, assetsKey, version),
}));
let mj;
for (const entry of entries) {
  const path = `${output}/${entry.key}.mjb.gz`;
  if (
    await stat(path).then(
      (s) => s.size > 0,
      () => false,
    )
  )
    continue;
  if (!mj) {
    mj = await loadMujoco();
    if (mj.mj_versionString() !== version)
      throw Error('MuJoCo package/WASM version mismatch');
    for (const dir of ['/robot', '/robot/meshes', '/robot/textures'])
      mj.FS.mkdir(dir);
    for (const { name, bytes } of assets)
      mj.FS.writeFile('/robot/' + name, bytes);
  }
  const started = performance.now();
  const sim = new Simulation(mj, xml, entry.id);
  try {
    mj.mj_saveModel(sim.model, '/track.mjb', null);
    const packed = gzipSync(mj.FS.readFile('/track.mjb'));
    const temporary = `${path}.${process.pid}.tmp`;
    await writeFile(temporary, packed);
    await rename(temporary, path);
    console.log(
      `Compiled ${entry.id}: ${(packed.length / 1048576).toFixed(1)} MB, ${((performance.now() - started) / 1000).toFixed(1)} s`,
    );
  } finally {
    sim.dispose();
    mj.FS.unlink('/track.mjb');
  }
}
// This directory is generated; retain only the current content-addressed assets.
const keep = new Set(entries.map(({ key }) => key + '.mjb.gz'));
for (const name of await readdir(output))
  if (name.endsWith('.mjb.gz') && !keep.has(name))
    await unlink(`${output}/${name}`);
await writeFile(
  `${output}/manifest.json`,
  JSON.stringify({ version, assetsKey, tracks: entries }, null, 2) + '\n',
);
