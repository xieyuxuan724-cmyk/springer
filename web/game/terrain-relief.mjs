// Small, deterministic terrain combinations. Distances and heights are metres.
// Kept separate from the mandatory full-width jump trenches in maps.mjs.
const layouts = {
  harbor: {
    craters: [1.65, 3.55, 6.1, 9.25],
    stairs: [4.65],
    bars: [7.05],
    wave: 4.1,
    troughs: 5.25,
    pads: 8.1,
  },
  cloud: {
    craters: [1.55, 5.5, 9.0, 12.4, 16.6],
    stairs: [6.5],
    bars: [14.7],
    wave: 17.5,
    troughs: 8.0,
    pads: 3.0,
  },
  switchyard: {
    craters: [1.0, 5.6, 15.0],
    stairs: [0.35],
    bars: [12.0],
    wave: 2.5,
    troughs: 6.8,
    pads: 10.6,
  },
};

export function createRelief(map, curve) {
  const features = [];
  const add = (kind, u, length, height, extra = {}) => {
    const a = curve(map, u - 0.001),
      b = curve(map, u + 0.001);
    const speed = Math.hypot(b[0] - a[0], b[1] - a[1]) / 0.002;
    features.push({
      kind,
      u,
      length,
      height,
      halfU: length / (2 * speed),
      ...extra,
    });
  };
  const shift = (u, metres) => {
    const a = curve(map, u - 0.001),
      b = curve(map, u + 0.001);
    return u + metres / (Math.hypot(b[0] - a[0], b[1] - a[1]) / 0.002);
  };
  const layout = layouts[map.id];
  for (const [cluster, u] of layout.craters.entries()) {
    // Stagger both wheel lines and leave room to choose a line around the bowls.
    for (const [i, lane] of [-0.42, 0.28, -0.08, 0.48, -0.32].entries()) {
      const diameter = 0.44 + ((i + cluster) % 3) * 0.07;
      add(
        'crater',
        shift(u, (i - 2) * 0.48),
        diameter,
        -(0.025 + ((i + cluster) % 3) * 0.005),
        { lane, rim: 0.006 },
      );
    }
  }
  for (const u of layout.stairs) add('stairs', u, 1.8, 0.075);
  for (const u of layout.bars)
    for (let i = -1; i <= 1; i++) add('bar', shift(u, i * 0.52), 0.24, 0.025);
  add('wave', layout.wave, 1.4, 0.025);
  for (const [i, lane] of [-0.38, 0.38].entries()) {
    add('trough', shift(layout.troughs, (i - 0.5) * 0.5), 0.4, -0.025, {
      lane,
      span: map.width * 0.5,
    });
    add('pad', shift(layout.pads, (i - 0.5) * 0.55), 0.55, 0.025, {
      lane,
      span: map.width * 0.55,
    });
  }
  return features;
}

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
function heightAt(h, u, lateral, width) {
  const t = (u - h.u) / h.halfU;
  if (Math.abs(t) >= 1) return 0;
  if (h.kind === 'crater') {
    const r = Math.hypot(t, (lateral - (h.lane * width) / 2) / (h.length / 2));
    if (r >= 1) return 0;
    const bowl =
      r < 0.78 ? (h.height * (1 + Math.cos((Math.PI * r) / 0.78))) / 2 : 0;
    const rim =
      r > 0.68 ? h.rim * Math.sin((Math.PI * (r - 0.68)) / 0.32) ** 2 : 0;
    return bowl + rim;
  }
  if (h.kind === 'stairs') {
    const x = ((t + 1) * h.length) / 2;
    // Three 2.5 cm risers, 32 cm treads, a landing, then an easy exit ramp.
    const rises = [0, 0.32, 0.64].reduce(
      (sum, start) => sum + clamp((x - start) / 0.1, 0, 1),
      0,
    );
    return (h.height / 3) * rises * clamp((h.length - x) / 0.5, 0, 1);
  }
  if (h.kind === 'wave')
    return h.height * Math.sign(t) * Math.sin(Math.PI * t) ** 2;
  if (h.kind === 'trough' || h.kind === 'pad') {
    const side = Math.abs(lateral - (h.lane * width) / 2) / (h.span / 2);
    return (
      h.height *
      clamp((1 - Math.abs(t)) / 0.45, 0, 1) *
      clamp((1 - side) / 0.4, 0, 1)
    );
  }
  // A rounded transverse rib, embedded in the road with no underside seam.
  return h.height * Math.cos((Math.PI * t) / 2) ** 2;
}

export function reliefSamples(features) {
  return features.flatMap((h) => {
    const samples = (
      h.kind === 'crater'
        ? [-1, -0.55, 0, 0.55, 1]
        : [-1, -0.85, -0.65, -0.4, 0, 0.4, 0.65, 0.85, 1]
    ).map((t) => h.u + h.halfU * t);
    if (h.kind === 'stairs')
      for (const x of [0, 0.1, 0.32, 0.42, 0.64, 0.74, 1.3, 1.8])
        samples.push(h.u + h.halfU * ((2 * x) / h.length - 1));
    if (h.kind === 'wave')
      for (let i = -8; i <= 8; i++) samples.push(h.u + (h.halfU * i) / 8);
    return samples;
  });
}

// Refine only the small crater patches, rather than subdividing the entire road.
// crossPoint and the mesh builder share this piecewise-linear cross section.
export function reliefSection(features, u, width, gap) {
  const active = features.filter((h) => Math.abs(u - h.u) <= h.halfU + 1e-8);
  const boundary = [-width / 2, -gap, gap, width / 2];
  const lanes = [...boundary];
  for (const h of active) {
    if (!['crater', 'trough', 'pad'].includes(h.kind)) continue;
    const radius = (h.span ?? h.length) / 2;
    const t = (u - h.u) / h.halfU;
    const outer = Math.sqrt(Math.max(0, 1 - t * t));
    const rim = Math.sqrt(Math.max(0, 0.82 ** 2 - t * t));
    const samples =
      h.kind === 'crater'
        ? [-outer, -rim, 0, rim, outer]
        : [-1, -0.6, 0, 0.6, 1];
    for (const r of samples) {
      const x = clamp((h.lane * width) / 2 + r * radius, -width / 2, width / 2);
      if (boundary.every((b) => Math.abs(b - x) > 0.003)) lanes.push(x);
    }
  }
  lanes.sort((a, b) => a - b);
  const section = lanes
    .filter(
      (x, i) =>
        (!i ||
          x - lanes[i - 1] > 0.003 ||
          (x !== lanes[i - 1] && boundary.includes(x))) &&
        Math.abs(x) >= gap - 1e-8,
    )
    .map((x) => [
      x,
      clamp(
        active.reduce((z, h) => z + heightAt(h, u, x, width), 0),
        -0.1,
        0.1,
      ),
    ]);
  const simplified = [];
  for (const point of section) {
    while (simplified.length > 1) {
      const b = simplified.at(-1),
        a = simplified.at(-2);
      if (
        boundary.includes(b[0]) ||
        Math.abs(
          (b[1] - a[1]) * (point[0] - b[0]) - (point[1] - b[1]) * (b[0] - a[0]),
        ) > 1e-10
      )
        break;
      simplified.pop();
    }
    simplified.push(point);
  }
  return simplified;
}

export function sectionHeight(section, lateral) {
  if (lateral <= section[0][0]) return section[0][1];
  for (let i = 1; i < section.length; i++) {
    const [b, bz] = section[i],
      [a, az] = section[i - 1];
    if (lateral <= b) return az + ((bz - az) * (lateral - a)) / (b - a);
  }
  return section.at(-1)[1];
}
