// One periodic track definition drives rendering, convex collision tiles and race logic.
// Units: metres, Z up. A closed road mesh must NOT be used as one MuJoCo convex hull.
import {
  createRelief,
  reliefSamples,
  reliefSection,
  sectionHeight,
} from './terrain-relief.mjs';
export const DEFAULT_MAP_ID = 'cloud';
export const MAPS = [
  {
    id: 'harbor',
    name: 'Toy Harbor',
    tag: 'Wide overtakes · Low jumps · Plenty of gems',
    description:
      'Race for the inside line around a toy freight yard. Wide lanes allow side-by-side overtakes, with gems scattered across the road.',
    pickupLanes: [-0.8, 0, 0.8],
    color: '#e5ff75',
    sky: '#243d54',
    ground: '#264c61',
    road: '#43505d',
    friction: 0.9,
    width: 3.0,
    railHeight: 0.38,
    railColor: '#e7edf0',
    railAccent: '#ec6952',
    theme: 'port',
    points: [
      [0, 0, 0],
      [5, 0, 0],
      [9, 2, 0.02],
      [10, 6, 0.06],
      [7, 9, 0.12],
      [2, 9, 0.06],
      [-3, 7, 0],
      [-6, 4, 0],
      [-6, 1, 0],
      [-4, 0, 0],
    ],
  },
  {
    id: 'cloud',
    name: 'Cloud Climb',
    tag: 'Linked S-bends · Twin hairpins · Cloudtop loop',
    description:
      'The default course. Climb narrow bridges through S-bends and hairpins, loop around the cloudtop, then descend toward the finish.',
    pickupLanes: [0],
    color: '#ffe5aa',
    sky: '#b7dcee',
    ground: '#89bdd8',
    road: '#d4dae1',
    friction: 0.95,
    width: 1.85,
    railHeight: 0.48,
    railColor: '#b5d8e5',
    railAccent: '#5686bd',
    theme: 'viaduct',
    points: [
      [0, 0, 0],
      [5, 0, 0],
      [9, 2, 0.35],
      [11, 5, 0.9],
      [10, 8, 1.6],
      [6, 9, 2.3],
      [5, 12, 2.5],
      [9, 15, 2.5],
      [6, 18, 2.1],
      [1, 17, 1.8],
      [-2, 13, 1.3],
      [-1, 10, 0.8],
      [-4, 8, 0.45],
      [-8, 10, 0.3],
      [-12, 7, 0.1],
      [-10, 3, 0],
      [-6, 2, 0],
      [-4, -1, 0],
    ],
  },
  {
    id: 'switchyard',
    name: 'Candy Junction',
    tag: 'Equal routes · Garden bridge / tunnel · Figure-eight crossing',
    description:
      'Garden bridge or candy tunnel? Pick one of two separate routes of similar length, merge again, then tackle the figure-eight crossing.',
    pickupLanes: [0],
    color: '#ffd67d',
    sky: '#d3c3ec',
    ground: '#b9a6cf',
    road: '#675c83',
    friction: 0.95,
    width: 1.22,
    railHeight: 0.36,
    railColor: '#fff1d9',
    railAccent: '#eb8bb1',
    theme: 'candy',
    allowCrossings: true,
    // Short ramps need continuous pitch; per-knot easing is too abrupt here.
    smoothHeight: true,
    checkpointUs: [1, 2, 6, 7, 9, 11, 13, 15],
    splits: [
      {
        name: 'Split routes',
        kind: 'fork',
        u: 4,
        halfU: 1.0,
        approachU: 0.4,
        lane: 1.145,
        halfWidth: 0.53,
        roadWidth: 3.52,
        routes: ['Garden bridge', 'Candy tunnel'],
      },
    ],
    points: Array.from({ length: 16 }, (_, i) => {
      const t = (i * Math.PI) / 8;
      const right = [
        [0, 0],
        [2, 3.7],
        [4.2, 3.3],
        [4.2, 2.1],
        [4.2, 0],
        [4.2, -2.1],
        [4.2, -3.3],
        [2, -3.7],
      ];
      const xy = i < 8 ? right[i] : [3.4 * Math.sin(t), 1.5 * Math.sin(2 * t)];
      return [...xy, 0.525 * (1 + Math.cos(t))];
    }),
  },
];
// Heights are offsets from the local graded road, never world-space heights.
const hazardSets = {
  harbor: [
    ['Fault trench · Space to jump', 0.7, -0.08, 0.3, true],
    ['Toy speed bump', 2.55, 0.04, 0.65],
    ['Harbor trough', 5.55, -0.04, 0.8],
    ['Finish ramp', 8.55, 0.05, 0.65],
  ],
  cloud: [
    ['Cloud gap · Space to jump', 0.7, -0.09, 0.28, true],
    // The summit is level: a bump on the descent compounds both exit grades.
    ['Bridge seam', 2.0, -0.04, 1.2],
    ['Sunken bridge deck', 15.55, -0.05, 1.2],
    ['Cloud hollow', 11.2, -0.025, 1.0],
    ['Cloud ripples', 13.5, 0.025, 1.0],
  ],
  switchyard: [
    ['Candy gap · Space to jump', 7.8, -0.08, 0.26, true],
    ['Underpass trough', 9, -0.025, 0.7],
    ['Return slope', 13, 0.025, 0.7],
    ['Frosting hollow', 14.5, -0.025, 0.7],
  ],
};
export const getMap = (id) =>
  MAPS.find((m) => m.id === id) ??
  MAPS.find((m) => m.id === DEFAULT_MAP_ID) ??
  MAPS[0];
// Debug tuning edits a map in place, so every derived track has to be dropped.
export const invalidateTrack = (id) => cache.delete(id);
export const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const STEPS = 16,
  THICKNESS = 0.18,
  RAIL_WIDTH = 0.12;
const cache = new Map();
const smooth = (t) => t * t * t * (10 + t * (-15 + 6 * t));
function cat(a, b, c, d, t) {
  return (
    ((1 - t) ** 3 * a +
      (3 * t * t * t - 6 * t * t + 4) * b +
      (-3 * t * t * t + 3 * t * t + 3 * t + 1) * c +
      t * t * t * d) /
    6
  );
}
function curve(map, u) {
  const nodes = map.knots ?? map.points;
  const n = nodes.length;
  u = ((u % n) + n) % n;
  const i = Math.floor(u),
    t = u - i;
  const a = nodes[(i + n - 1) % n],
    b = nodes[i],
    c = nodes[(i + 1) % n],
    d = nodes[(i + 2) % n];
  return [
    cat(a[0], b[0], c[0], d[0], t),
    cat(a[1], b[1], c[1], d[1], t),
    map.smoothHeight
      ? cat(a[2], b[2], c[2], d[2], t)
      : b[2] + (c[2] - b[2]) * smooth(t),
  ];
}
function forkAmount(split, u) {
  const t = Math.abs(u - split.u) / split.halfU;
  return t < 1 ? Math.cos((t * Math.PI) / 2) ** 2 : 0;
}
function frame(map, u) {
  const p = curve(map, u),
    a = curve(map, u - 0.001),
    b = curve(map, u + 0.001),
    yaw = Math.atan2(b[1] - a[1], b[0] - a[0]);
  const phase = u / (map.knots ?? map.points).length;
  // Broad glacier turns develop a shallow U profile; both ends join smoothly.
  const bowl =
    map.theme === 'glacier' ? 0.105 * Math.sin(Math.PI * phase) ** 4 : 0;
  const bank =
    map.theme === 'glacier' ? 0.045 * Math.sin(2 * Math.PI * phase) ** 3 : 0;
  let width =
    map.width *
    (map.theme === 'viaduct'
      ? 1 - 0.22 * Math.sin(2 * Math.PI * phase) ** 8
      : 1);
  let gap = 0;
  for (const split of map.splits ?? []) {
    if (split.kind === 'fork') {
      const amount = forkAmount(split, u);
      width += (split.roadWidth - map.width) * amount;
      gap = split.halfWidth * Math.sqrt(amount);
      continue;
    }
    const span = split.halfU + split.approachU;
    const f = clamp((span - Math.abs(u - split.u)) / split.approachU, 0, 1);
    width += (split.roadWidth - map.width) * smooth(f);
  }
  const baseZ = p[2];
  let hazard = null;
  for (const h of map.hazards ?? []) {
    const d = Math.abs(u - h.u) / h.halfU;
    if (d < 1) {
      // Jump obstacles have a 3 mm bevel, ordinary uneven patches a broad ramp.
      const bevel = h.requiresJump ? Math.min(0.1, 0.006 / h.length) : 0.5;
      p[2] += h.height * clamp((1 - d) / bevel, 0, 1);
      hazard = h;
    }
  }
  const section = reliefSection(map.relief ?? [], u, width, gap);
  return { p, yaw, width, gap, bowl, bank, u, baseZ, hazard, section };
}
export function crossPoint(f, lateral, zOffset = 0) {
  const edge = clamp(lateral / (f.width / 2), -1, 1);
  return [
    f.p[0] - Math.sin(f.yaw) * lateral,
    f.p[1] + Math.cos(f.yaw) * lateral,
    f.p[2] +
      f.bank * clamp(lateral, -f.width / 2, f.width / 2) +
      f.bowl * edge * edge +
      zOffset +
      (f.section ? sectionHeight(f.section, lateral) : 0),
  ];
}
// Indexed top, bottom and side faces share vertices, including the wraparound seam.
export function solidGrid(rows, depth = THICKNESS, closed = true) {
  const n = rows.length,
    w = rows[0].length,
    vertices = rows.flat(2),
    faces = [],
    topFaces = [];
  const offset = n * w;
  for (const row of rows)
    for (const p of row) vertices.push(p[0], p[1], p[2] - depth);
  const tri = (a, b, c) => faces.push(a, b, c);
  for (let i = 0; i < n - (closed ? 0 : 1); i++) {
    const k = (i + 1) % n;
    for (let j = 0; j < w - 1; j++) {
      const a = i * w + j,
        b = a + 1,
        c = k * w + j,
        d = c + 1;
      tri(a, c, d);
      tri(a, d, b);
      topFaces.push(a, c, d, a, d, b);
      tri(a + offset, d + offset, c + offset);
      tri(a + offset, b + offset, d + offset);
    }
    const a = i * w,
      c = k * w,
      b = i * w + w - 1,
      d = k * w + w - 1;
    tri(a, a + offset, c + offset);
    tri(a, c + offset, c);
    tri(b, d, d + offset);
    tri(b, d + offset, b + offset);
  }
  if (!closed)
    for (let j = 0; j < w - 1; j++) {
      const a = j,
        b = j + 1,
        c = (n - 1) * w + j,
        d = c + 1;
      tri(a, b, b + offset);
      tri(a, b + offset, a + offset);
      tri(c, c + offset, d + offset);
      tri(c, d + offset, d);
    }
  return { vertices, faces, topFaces, rows: n, columns: w, depth };
}
// Extrude a welded top mesh, deriving every outer and hole wall from boundary edges.
function extrudeTop(points, triangles, depth = THICKNESS, attributes) {
  const vertices = [],
    uv = [],
    relief = [],
    topFaces = [],
    ids = new Map(),
    edges = new Map();
  const index = (p, source) => {
    const key = p.map((v) => v.toFixed(9)).join(',');
    if (!ids.has(key)) {
      ids.set(key, vertices.length / 3);
      vertices.push(...p);
      if (attributes) {
        uv.push(...attributes[source].uv);
        relief.push(attributes[source].relief);
      }
    }
    return ids.get(key);
  };
  for (const triangle of triangles) {
    const p = triangle.map((i) => points[i]);
    const area =
      (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) -
      (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
    if (Math.abs(area) < 1e-10) continue;
    const [a, b, c] = p.map((v, i) => index(v, triangle[i]));
    topFaces.push(a, b, c);
    for (const [x, y] of [
      [a, b],
      [b, c],
      [c, a],
    ]) {
      const key = x < y ? `${x}:${y}` : `${y}:${x}`;
      if (edges.has(key)) edges.delete(key);
      else edges.set(key, [x, y]);
    }
  }
  const n = vertices.length / 3,
    faces = [...topFaces];
  for (let i = 0; i < n; i++)
    vertices.push(
      vertices[i * 3],
      vertices[i * 3 + 1],
      vertices[i * 3 + 2] - depth,
    );
  for (let i = 0; i < topFaces.length; i += 3)
    faces.push(topFaces[i] + n, topFaces[i + 2] + n, topFaces[i + 1] + n);
  for (const [a, b] of edges.values()) faces.push(a, a + n, b + n, a, b + n, b);
  return {
    vertices,
    faces,
    topFaces,
    depth,
    uv: [...uv, ...uv],
    relief: [...relief, ...relief],
  };
}
function reliefRoad(frames, lengths) {
  const points = [],
    attributes = [],
    triangles = [];
  const rows = frames.map((f, i) =>
    [-1, 1].map((side) => {
      const lo = side < 0 ? -f.width / 2 : f.gap;
      const hi = side < 0 ? -f.gap : f.width / 2;
      return f.section
        .filter(([x]) => x >= lo - 1e-8 && x <= hi + 1e-8)
        .map(([x, z]) => {
          const id = points.length;
          points.push(crossPoint(f, x));
          attributes.push({
            uv: [x, lengths[i]],
            relief: f.p[2] - f.baseZ + z,
          });
          return { id, t: (x - lo) / (hi - lo) };
        });
    }),
  );
  for (let i = 0; i < frames.length; i++) {
    const next = (i + 1) % frames.length;
    for (let side = 0; side < 2; side++) {
      const a = rows[i][side],
        b = rows[next][side];
      let j = 0,
        k = 0;
      // Zipper unequal cross sections: no T-junctions where local refinement ends.
      while (j < a.length - 1 || k < b.length - 1) {
        if ((a[j + 1]?.t ?? Infinity) < (b[k + 1]?.t ?? Infinity)) {
          triangles.push([a[j].id, b[k].id, a[j + 1].id]);
          j++;
        } else {
          triangles.push([a[j].id, b[k].id, b[k + 1].id]);
          k++;
        }
      }
    }
  }
  return extrudeTop(points, triangles, THICKNESS, attributes);
}
export function getTrack(map) {
  if (cache.has(map.id)) return cache.get(map.id);
  const us = Array.from(
    { length: map.knots.length * STEPS },
    (_, i) => i / STEPS,
  ).filter(
    (u) =>
      !(map.relief ?? []).some(
        (h) => h.kind === 'crater' && Math.abs(u - h.u) < h.halfU,
      ),
  );
  us.push(...reliefSamples(map.relief ?? []));
  for (const h of map.hazards) {
    for (let j = -8; j <= 8; j++) us.push(h.u + (h.halfU * j) / 8);
    if (h.requiresJump)
      for (const side of [-1, 1])
        us.push(h.u + side * h.halfU * (1 - Math.min(0.1, 0.006 / h.length)));
  }
  us.push(...map.checkpointUs);
  for (const split of map.splits ?? []) {
    for (let j = -32; j <= 32; j++) us.push(split.u + (split.halfU * j) / 32);
    us.push(
      split.u - split.halfU - split.approachU,
      split.u + split.halfU + split.approachU,
    );
  }
  us.sort((a, b) => a - b);
  const anchors = new Set([
    ...map.checkpointUs,
    ...map.hazards.map((h) => h.u),
    ...(map.relief ?? []).map((h) => h.u),
  ]);
  const samples = [];
  for (const u of us) {
    if (!samples.length || u - samples.at(-1) > 0.00025) samples.push(u);
    else if (anchors.has(u)) samples[samples.length - 1] = u;
  }
  const frames = samples.map((u) => frame(map, u));
  const count = frames.length;
  const lengths = [0];
  for (let i = 0; i < count; i++)
    lengths.push(lengths[i] + distance(frames[i].p, frames[(i + 1) % count].p));
  const road = reliefRoad(frames, lengths);
  const rails = [-1, 1].map((side) =>
    solidGrid(
      frames.map((f) => {
        const edge = (side * f.width) / 2;
        return (
          side < 0 ? [edge - RAIL_WIDTH, edge] : [edge, edge + RAIL_WIDTH]
        ).map((x) => crossPoint(f, x, map.railHeight));
      }),
      map.railHeight + 0.06,
    ),
  );
  const islands = (map.splits ?? [])
    .filter((s) => s.kind !== 'fork')
    .map((split) =>
      solidGrid(
        Array.from({ length: 17 }, (_, i) => {
          const t = (i - 8) / 8,
            f = frame(map, split.u + split.halfU * t);
          const half = Math.max(0.012, split.halfWidth * Math.sqrt(1 - t * t));
          return [-half, half].map((lateral) =>
            crossPoint(f, lateral, split.height),
          );
        }),
        split.height + 0.06,
        false,
      ),
    );
  const innerRails = (map.splits ?? [])
    .filter((s) => s.kind === 'fork')
    .map((split) => {
      const rows = frames.filter(
        (f) => Math.abs(f.u - split.u) <= split.halfU + 1e-8,
      );
      const outline = [
        ...rows.map((f) => crossPoint(f, -f.gap, map.railHeight)),
        ...rows
          .slice(1, -1)
          .reverse()
          .map((f) => crossPoint(f, f.gap, map.railHeight)),
      ];
      const centre = frame(map, split.u).p;
      return solidGrid(
        outline.map((p) => [
          p,
          [
            centre[0] + (p[0] - centre[0]) * 0.84,
            centre[1] + (p[1] - centre[1]) * 0.84,
            p[2],
          ],
        ]),
        map.railHeight + 0.06,
      );
    });
  const structures = [];
  for (const split of map.splits ?? []) {
    if (split.kind !== 'fork') continue;
    for (let i = -2; i <= 2; i++) {
      const f = frame(map, split.u + i * 0.32);
      const lane = -(f.width / 2 + f.gap) / 2,
        laneWidth = f.width / 2 - f.gap;
      const block = (lateral, halfWidth, halfLength, height, depth, color) => {
        const rows = [-halfLength, halfLength].map((d) =>
          [-halfWidth, halfWidth].map((w) => {
            const p = crossPoint(f, lateral + w, height);
            return [
              p[0] + Math.cos(f.yaw) * d,
              p[1] + Math.sin(f.yaw) * d,
              p[2],
            ];
          }),
        );
        structures.push({ mesh: solidGrid(rows, depth, false), color });
      };
      block(
        lane,
        laneWidth / 2 + 0.26,
        0.43,
        0.9,
        0.1,
        i % 2 ? '#e390c9' : '#af86cc',
      );
      for (const side of [-1, 1])
        block(
          lane + side * (laneWidth / 2 + 0.2),
          0.05,
          0.05,
          0.85,
          0.9,
          '#fff2dc',
        );
    }
  }
  const all = frames.flatMap((f) =>
      [-1, 1].map((side) => crossPoint(f, side * (f.width / 2 + RAIL_WIDTH))),
    ),
    minX = Math.min(...all.map((p) => p[0])) - 0.3,
    maxX = Math.max(...all.map((p) => p[0])) + 0.3,
    minY = Math.min(...all.map((p) => p[1])) - 0.3,
    maxY = Math.max(...all.map((p) => p[1])) + 0.3;
  const result = {
    frames,
    lengths,
    checkpointLengths: [
      ...map.checkpointUs.map(
        (u) => lengths[frames.findIndex((f) => Math.abs(f.u - u) < 1e-8)],
      ),
      lengths.at(-1) +
        lengths[
          frames.findIndex((f) => Math.abs(f.u - map.checkpointUs[0]) < 1e-8)
        ],
    ],
    length: lengths.at(-1),
    road,
    rails,
    islands,
    innerRails,
    structures,
    bounds: { minX, maxX, minY, maxY },
    minimap: frames.filter((_, i) => i % 3 === 0).map((f) => f.p.slice(0, 2)),
  };
  cache.set(map.id, result);
  return result;
}
export function atDistance(map, s) {
  const t = getTrack(map);
  s = ((s % t.length) + t.length) % t.length;
  let i = 0;
  while (i < t.frames.length - 1 && t.lengths[i + 1] < s) i++;
  const f = (s - t.lengths[i]) / (t.lengths[i + 1] - t.lengths[i]);
  return frame(
    map,
    t.frames[i].u +
      f * ((t.frames[i + 1]?.u ?? map.knots.length) - t.frames[i].u),
  );
}
export function heading(map, index = 0) {
  return frame(map, map.checkpointUs[index]).yaw;
}
// Both branches reconnect to the same ordered course. Only the steering target
// changes; the island and its guard are real collision geometry.
export function branchOffset(map, s, branch = 'left') {
  const u = atDistance(map, s).u;
  for (const split of map.splits ?? []) {
    const d = Math.abs(u - split.u);
    if (split.kind === 'fork') {
      const amount = forkAmount(split, u);
      const lead = smooth(
        clamp((split.halfU + split.approachU - d) / split.approachU, 0, 1),
      );
      return (
        (branch === 'right' ? -1 : 1) *
        ((map.width / 4) * lead +
          ((split.roadWidth - map.width) / 4) * amount +
          (split.halfWidth / 2) * Math.sqrt(amount))
      );
    }
    const amount = smooth(
      clamp((split.halfU + split.approachU - d) / split.approachU, 0, 1),
    );
    if (amount > 0) return (branch === 'right' ? -1 : 1) * split.lane * amount;
  }
  return 0;
}
/** @returns {number[][][]} */
export function routePolylines(map) {
  const track = getTrack(map);
  return (track.routes ??= (
    map.splits?.length ? ['left', 'right'] : ['left']
  ).map((branch) => {
    const points = track.frames
      .filter((_, i) => i % 2 === 0)
      .map((f) => {
        const i = track.frames.indexOf(f);
        return crossPoint(f, branchOffset(map, track.lengths[i], branch));
      });
    return [...points, points[0]];
  }));
}
export function routeLengths(map) {
  const track = getTrack(map);
  return Object.fromEntries(
    ['left', 'right'].map((branch) => {
      const path = track.lengths
        .slice(0, -1)
        .map((s) =>
          crossPoint(atDistance(map, s), branchOffset(map, s, branch)),
        );
      const full = path.reduce(
        (sum, p, i) =>
          sum +
          Math.hypot(...p.map((v, j) => v - path[(i + 1) % path.length][j])),
        0,
      );
      const split = map.splits?.[0];
      let fork = 0;
      if (split)
        for (let i = 0; i < path.length - 1; i++)
          if (
            track.frames[i].u >= split.u - split.halfU - split.approachU &&
            track.frames[i + 1].u <= split.u + split.halfU + split.approachU
          )
            fork += Math.hypot(...path[i].map((v, j) => v - path[i + 1][j]));
      return [branch, { full, fork }];
    }),
  );
}
export function islandAt(map, u, lateral = 0) {
  return (map.splits ?? []).some((split) => {
    if (split.kind === 'fork') return false;
    const t = (u - split.u) / split.halfU;
    return (
      Math.abs(t) <= 1 &&
      Math.abs(lateral) <=
        Math.max(0.012, split.halfWidth * Math.sqrt(1 - t * t))
    );
  });
}
export function spawn(map, slot = 0, checkpoint = 0) {
  const t = getTrack(map),
    back =
      checkpoint === 0
        ? -0.4 - Math.floor(slot / 3) * 0.4
        : 0.15 + Math.floor(slot / 3) * 0.28;
  const f = atDistance(map, t.checkpointLengths[checkpoint] + back);
  return { p: crossPoint(f, ((slot % 3) - 1) * 0.32, 0.115), yaw: f.yaw };
}
export function pickups(map) {
  const t = getTrack(map);
  const boxes = map.points.flatMap((_, i) => {
    // Wide harbor rows offer parallel pickup routes; sky bridges put supplies at landings.
    if (map.theme === 'viaduct' && i % 2 === 1) return [];
    const a = t.checkpointLengths[i],
      b = t.checkpointLengths[i + 1];
    const f = atDistance(map, map.theme === 'viaduct' ? a : (a + b) / 2);
    if (f.gap > 0) return [];
    return (map.pickupLanes ?? [0]).map((lateral, lane) => ({
      id: `${i}:${lane}`,
      p: crossPoint(f, lateral, 0.2),
    }));
  });
  for (const [i, split] of (map.splits ?? []).entries()) {
    const f = frame(map, split.u);
    for (const side of [-1, 1])
      boxes.push({
        id: `split:${i}:${side}`,
        p: crossPoint(f, side * split.lane, 0.2),
      });
  }
  return boxes;
}
export function surfaceAt(map, x, y, z) {
  const t = getTrack(map);
  let best = Infinity,
    result = null;
  for (let i = 0; i < t.frames.length; i++) {
    const a = t.frames[i],
      b = t.frames[(i + 1) % t.frames.length],
      dx = b.p[0] - a.p[0],
      dy = b.p[1] - a.p[1],
      f = clamp(
        ((x - a.p[0]) * dx + (y - a.p[1]) * dy) / (dx * dx + dy * dy),
        0,
        1,
      );
    const px = a.p[0] + f * dx,
      py = a.p[1] + f * dy,
      d = Math.hypot(x - px, y - py);
    // At an overpass XY alone cannot identify the deck underneath the robot.
    const height = z - (a.p[2] + f * (b.p[2] - a.p[2]));
    // Use height to reject another deck, not to pull the XY projection uphill.
    // The dead band includes the body/wheels and the normal jump clearance.
    const verticalError =
      height < -0.08 ? height + 0.08 : Math.max(0, height - 0.45);
    const score =
      d * d +
      (map.allowCrossings && Number.isFinite(z) ? 16 * verticalError ** 2 : 0);
    if (score >= best) continue;
    best = score;
    const along = frame(
      map,
      a.u + f * ((t.frames[i + 1]?.u ?? map.knots.length) - a.u),
    );
    const lateral =
      -(x - px) * Math.sin(along.yaw) + (y - py) * Math.cos(along.yaw);
    result = {
      z: crossPoint(along, lateral)[2],
      lateral,
      width: along.width,
      gap: along.gap,
      onRoad:
        Math.abs(lateral) >= along.gap && Math.abs(lateral) <= along.width / 2,
      s: t.lengths[i] + f * (t.lengths[i + 1] - t.lengths[i]),
      distance: d,
    };
  }
  return result;
}
function prism(vertices, depth) {
  // MuJoCo recentres each mesh and stores its vertices as float32. A tiny
  // coplanar overlap seals the resulting micrometre-wide numerical seams.
  // The rendered road remains an exactly welded manifold; this is collision-only.
  const centre = [0, 1, 2].map(
    (axis) => vertices.reduce((s, p) => s + p[axis], 0) / vertices.length,
  );
  vertices = vertices.map((p) =>
    p.map((v, axis) => centre[axis] + (v - centre[axis]) * 1.001),
  );
  const v = [...vertices, ...vertices.map((p) => [p[0], p[1], p[2] - depth])];
  const n = vertices.length,
    faces = [];
  for (let i = 1; i < n - 1; i++) faces.push(0, i, i + 1, n, n + i + 1, n + i);
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    faces.push(i, i + n, j + n, i, j + n, j);
  }
  return {
    vertices: v.flat(),
    faces,
  };
}
function tiles(mesh, prefix) {
  const out = [],
    triangles = [],
    edges = new Map(),
    used = new Set();
  const point = (id) => mesh.vertices.slice(id * 3, id * 3 + 3);
  for (let i = 0; i < mesh.topFaces.length; i += 3) {
    const ids = mesh.topFaces.slice(i, i + 3),
      keys = [];
    for (let j = 0; j < 3; j++) {
      const a = ids[j],
        b = ids[(j + 1) % 3];
      const key = a < b ? `${a}:${b}` : `${b}:${a}`;
      keys.push(key);
      const adjacent = edges.get(key) ?? [];
      adjacent.push(triangles.length);
      edges.set(key, adjacent);
    }
    triangles.push({ ids, keys });
  }
  for (let i = 0; i < triangles.length; i++) {
    if (used.has(i)) continue;
    let top = triangles[i].ids.map(point);
    // Exactly coplanar neighbours can share one convex prism. Never convexify a
    // crater: a non-planar pair retains the two original collision triangles.
    for (const key of triangles[i].keys) {
      const other = edges.get(key).find((j) => j !== i && !used.has(j));
      if (other === undefined) continue;
      const ids = [...new Set([...triangles[i].ids, ...triangles[other].ids])];
      const p = ids.map(point);
      if (p.length !== 4) continue;
      const [a, b, c, d] = p;
      const ab = b.map((v, k) => v - a[k]),
        ac = c.map((v, k) => v - a[k]);
      const normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
      ];
      if (
        Math.abs(normal.reduce((sum, n, k) => sum + n * (d[k] - a[k]), 0)) >
        1e-9 * Math.hypot(...normal)
      )
        continue;
      const x = p.reduce((s, v) => s + v[0], 0) / 4,
        y = p.reduce((s, v) => s + v[1], 0) / 4;
      p.sort(
        (a, b) =>
          Math.atan2(a[1] - y, a[0] - x) - Math.atan2(b[1] - y, b[0] - x),
      );
      if (
        p.some((a, j) => {
          const b = p[(j + 1) % 4],
            c = p[(j + 2) % 4];
          return (
            (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <=
            1e-10
          );
        })
      )
        continue;
      top = p;
      used.add(other);
      break;
    }
    used.add(i);
    out.push({ name: `${prefix}_${i}`, ...prism(top, mesh.depth) });
  }
  return out;
}
export function collisionTiles(map) {
  const t = getTrack(map);
  const guards = (mesh, prefix) =>
    Array.from({ length: mesh.rows }, (_, i) => {
      const j = (i + 1) % mesh.rows;
      // One convex guard segment per cross-section pair. Its side wall and shared
      // ends are unchanged; only a possibly twisted top is enclosed by the hull.
      const top = [i * 2, j * 2, j * 2 + 1, i * 2 + 1].map((id) =>
        mesh.vertices.slice(id * 3, id * 3 + 3),
      );
      return { name: `${prefix}_${i}`, ...prism(top, mesh.depth) };
    });
  return (t.collision ??= [
    ...tiles(t.road, 'road'),
    ...t.rails.flatMap((r, i) => guards(r, 'rail' + i)),
    ...t.structures.flatMap((r, i) => tiles(r.mesh, 'rail_tunnel' + i)),
    ...t.innerRails.flatMap((r, i) => guards(r, 'rail_fork' + i)),
    ...t.islands.flatMap((r, i) => tiles(r, 'rail_split' + i)),
  ]);
}
const fmt = (n) => Number(n.toFixed(9));
export function terrainAssetsXML(map) {
  return collisionTiles(map)
    .map(
      (t) =>
        `<mesh name="${t.name}" vertex="${t.vertices.map(fmt).join(' ')}" face="${t.faces.join(' ')}"/>`,
    )
    .join('');
}
export function terrainXML(map) {
  return collisionTiles(map)
    .map(
      (t) =>
        `<geom name="${t.name}" type="mesh" mesh="${t.name}" group="3" friction="${t.name.startsWith('rail') ? 0.7 : map.friction} 0.005 0.0001" condim="3"/>`,
    )
    .join('');
}

// Control polygons are smoothed periodically; checkpoints lie ON the resulting curve.
for (const map of MAPS) {
  map.knots = map.points;
  map.checkpointUs ??= map.knots.map((_, i) => i);
  map.points = map.checkpointUs.map((u) => curve(map, u));
  for (const split of map.splits ?? []) {
    if (split.kind === 'fork') continue;
    const speed =
      distance(curve(map, split.u - 0.001), curve(map, split.u + 0.001)) /
      0.002;
    split.halfU = split.length / (2 * speed);
    split.approachU = split.approach / speed;
  }
}

// halfU converts a hazard's metre length into curve parameter at the local speed.
export function setHazards(map, list) {
  map.hazards = list.map(
    ({ name, u, height, length, requiresJump = false }) => {
      const speed =
        distance(curve(map, u - 0.001), curve(map, u + 0.001)) / 0.002;
      return {
        name,
        u,
        height: clamp(height, -0.1, 0.1),
        length,
        requiresJump,
        halfU: length / (2 * speed),
      };
    },
  );
  invalidateTrack(map.id);
  return map.hazards;
}

for (const map of MAPS) {
  map.relief = createRelief(map, curve);
  setHazards(
    map,
    hazardSets[map.id].map(([name, u, height, length, requiresJump]) => ({
      name,
      u,
      height: clamp(height, -0.1, 0.1),
      length,
      requiresJump,
    })),
  );
}
