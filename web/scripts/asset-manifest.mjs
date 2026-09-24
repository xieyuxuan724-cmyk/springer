import { readFile, writeFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { browserAssets } from './browser-assets.mjs';
import { robotFiles } from '../game/compiled-model.mjs';
const files = new Map([...browserAssets()].map(([key,value]) => ['/'+key,value]));
for (const name of await readdir('public/runtime')) {
  if (/\.(wasm|mjs)$/.test(name)) files.set('/runtime/'+name,'public/runtime/'+name);
}
for (const name of await readdir('public/runtime/tracks')) files.set('/runtime/tracks/'+name,'public/runtime/tracks/'+name);
const assets = {};
for (const [url,path] of files) {
  const bytes=await readFile(path);
  assets[url]={size:bytes.length,hash:createHash('sha256').update(bytes).digest('hex')};
}
const xml=await readFile(files.get('/robot/beni.xml'),'utf8');
const manifest=JSON.parse(await readFile(files.get('/robot/manifest.json'),'utf8'));
const common=['/runtime/mujoco.wasm','/runtime/ort-wasm-simd-threaded.wasm','/policies/beni.onnx','/robot/beni.xml','/robot/manifest.json','/runtime/tracks/manifest.json',...robotFiles(xml,manifest).map(f=>'/robot/'+f)];
const tracks=JSON.parse(await readFile('public/runtime/tracks/manifest.json','utf8'));
const startup=Object.fromEntries(tracks.tracks.map(t=>[t.id,[...common,`/runtime/tracks/${t.key}.mjb.gz`]]));
await writeFile('.generated/asset-manifest.json',JSON.stringify({assets,startup}));
