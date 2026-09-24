import {
  atDistance,
  crossPoint,
  getTrack,
  surfaceAt,
  distance,
} from '../game/maps.mjs';

// Each slot roams within one sector, keeping supplies spread around the course.
// Race rules own positions, lifetimes and rewards; the renderer draws snapshots.
export function renewGem(
  map,
  gem,
  count,
  now,
  random,
  players = [],
  others = [],
) {
  const length = getTrack(map).length;
  for (let attempt = 0; attempt < 32; attempt++) {
    const s = ((gem.slot + 0.08 + random() * 0.84) * length) / count;
    const f = atDistance(map, s);
    if (f.hazard?.requiresJump) continue;
    const edge = f.width / 2 - 0.18;
    const lateral =
      f.gap > 0
        ? (random() < 0.5 ? -1 : 1) *
          (f.gap + 0.18 + random() * (edge - f.gap - 0.18))
        : (random() * 2 - 1) * edge;
    const p = crossPoint(f, lateral, 0.2);
    const floor = surfaceAt(map, ...p);
    if (
      !floor.onRoad ||
      Math.abs(floor.lateral) > floor.width / 2 - 0.16 ||
      (floor.gap > 0 && Math.abs(floor.lateral) < floor.gap + 0.16)
    )
      continue;
    if (
      players.some(
        (x) => x.p && distance(x.p, p) < 0.8 && Math.abs(x.p[2] - p[2]) < 0.5,
      )
    )
      continue;
    if (
      others.some(
        (x) =>
          x !== gem &&
          x.active &&
          distance(x.p, p) < 0.65 &&
          Math.abs(x.p[2] - p[2]) < 0.5,
      )
    )
      continue;
    if (gem.p && distance(gem.p, p) < 0.35) continue;
    gem.p = p;
    gem.active = true;
    gem.generation++;
    gem.availableAt = now;
    gem.expiresAt = now + 12000 + random() * 8000;
    gem.value = random() < 0.18 ? 3 : 1;
    gem.color =
      gem.value === 3 ? '#ff76cc' : random() < 0.5 ? '#ffe66d' : '#70f4ff';
    return;
  }
  // A crowded sector waits rather than materializing directly under a racer.
  gem.active = false;
  gem.availableAt = now + 700;
}

export function createGems(map, now, random) {
  const count = Math.max(10, Math.min(22, Math.ceil(getTrack(map).length / 3)));
  const gems = Array.from({ length: count }, (_, slot) => ({
    id: `gem-${slot}`,
    slot,
    generation: 0,
    active: false,
  }));
  for (const gem of gems) {
    renewGem(map, gem, count, now, random, [], gems);
    if (random() < 0.3) gem.availableAt += 800 + random() * 3000;
  }
  return gems;
}
