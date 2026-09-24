// Live map tuning shared by the main thread and the simulation worker.
// Each thread owns its own copy of maps.mjs, so the same tuning has to be
// applied on both sides or the rendered road and the MuJoCo collision tiles
// would drift apart.
import { MAPS, getMap, invalidateTrack, setHazards } from './maps.mjs';

// Scalars and colours a debug session may edit. Control points stay read-only:
// they feed the periodic smoothing that runs once when maps.mjs loads.
export const TUNABLE = [
  'width',
  'railHeight',
  'friction',
  'color',
  'sky',
  'ground',
  'road',
  'railColor',
  'railAccent',
];

const clone = (map) => ({
  ...Object.fromEntries(TUNABLE.map((k) => [k, map[k]])),
  hazards: map.hazards.map(({ name, u, height, length, requiresJump }) => ({
    name,
    u,
    height,
    length,
    requiresJump,
  })),
});

// Captured before any tuning is applied, so "reset" always restores maps.mjs.
const PRISTINE = new Map(MAPS.map((m) => [m.id, clone(m)]));

export const defaultTuning = (id) =>
  structuredClone(PRISTINE.get(getMap(id).id));
export const readTuning = (id) => clone(getMap(id));

// Only these fields change the swept surface or its friction, so only they
// require paying for a MuJoCo rebuild in the worker. Colours are render-only.
export function geometrySignature(tuning) {
  return JSON.stringify([
    tuning.width,
    tuning.railHeight,
    tuning.friction,
    tuning.hazards.map((h) => [h.u, h.height, h.length, !!h.requiresJump]),
  ]);
}

export function applyTuning(id, tuning) {
  const map = getMap(id);
  for (const key of TUNABLE) if (key in tuning) map[key] = tuning[key];
  if (tuning.hazards) setHazards(map, tuning.hazards);
  else invalidateTrack(map.id);
  return map;
}

const num = (n) => Number(n.toFixed(4));

// Pasteable fields for the map's MAPS entry plus its hazardSets entry.
// Control points are omitted: the exported scene tuning does not change the curve.
export function mapSnippet(id) {
  const map = getMap(id);
  const quoted = (v) => (typeof v === 'string' ? `'${v}'` : num(v));
  const fields = TUNABLE.map((k) => `    ${k}: ${quoted(map[k])},`).join('\n');
  const hazards = map.hazards
    .map(
      (h) =>
        `    ['${h.name}', ${num(h.u)}, ${num(h.height)}, ${num(h.length)}${h.requiresJump ? ', true' : ''}],`,
    )
    .join('\n');
  return `// maps.mjs → MAPS entry '${map.id}' (points unchanged)\n${fields}\n\n// maps.mjs → hazardSets.${map.id}\n  ${map.id}: [\n${hazards}\n  ],\n`;
}
