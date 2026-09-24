import { terrainXML, terrainAssetsXML, getMap } from './maps.mjs';

export function simulationXML(xml, mapId) {
  const map = getMap(mapId);
  return xml
    .replace('<asset>', '<asset>' + terrainAssetsXML(map))
    .replace('<worldbody>', '<worldbody>' + terrainXML(map));
}

// Cache identity, not an authentication hash. Works on LAN HTTP, where browser
// crypto.subtle is unavailable. Include part lengths to preserve boundaries.
export function fingerprint(...parts) {
  let a = 0x811c9dc5,
    b = 0x9e3779b9;
  const encoder = new TextEncoder();
  for (const part of parts) {
    const bytes = typeof part === 'string' ? encoder.encode(part) : part;
    for (let i = 0; i < bytes.length; i++) {
      a = Math.imul(a ^ bytes[i], 0x01000193);
      b = Math.imul(b ^ bytes[i], 0x85ebca6b);
    }
    a = Math.imul(a ^ bytes.length, 0x01000193);
    b = Math.imul(b ^ bytes.length, 0xc2b2ae35);
  }
  return [a, b].map((n) => (n >>> 0).toString(16).padStart(8, '0')).join('');
}

export function robotFiles(xml, manifest) {
  return manifest.filter((file) => xml.includes(file.split('/').at(-1))).sort();
}

export function compiledModelKey(xml, mapId, assetsKey, version) {
  return fingerprint('mjb-v1', version, assetsKey, simulationXML(xml, mapId));
}

export function modelFromBinary(mj, bytes) {
  // MjVFS.addBuffer converts every byte through Embind and takes seconds for
  // these models. MEMFS uses a typed-array copy; an empty VFS falls back to it.
  const path = '/compiled-track.mjb';
  mj.FS.writeFile(path, bytes);
  const vfs = new mj.MjVFS();
  try {
    return mj.MjModel.from_binary_path(path, vfs);
  } finally {
    vfs.delete();
    mj.FS.unlink(path);
  }
}
