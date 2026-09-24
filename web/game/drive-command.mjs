import { getTrack, atDistance } from './maps.mjs';
// Full turbo on straights; taper the extra speed as steering increases.
// These are policy commands, never direct changes to physical velocity.
export function boostedForwardSpeed(vx, wz, boosted, terrain = 1) {
  const straight = Math.max(0, Math.min(1, (0.6 - Math.abs(wz)) / 0.5));
  return (
    Math.sign(vx) *
    Math.min(2, Math.abs(vx) * (boosted ? 1 + (straight * terrain) / 3 : 1))
  );
}

export function terrainTurboFactor(map, s) {
  const track = getTrack(map);
  for (const h of [...map.hazards, ...(map.relief ?? [])]) {
    const i = track.frames.findIndex((f) => Math.abs(f.u - h.u) < 1e-7);
    const entry = track.lengths[i] - h.length / 2;
    if (
      (entry - s + track.length) % track.length < 1.2 ||
      (s - entry + track.length) % track.length < h.length + 0.6
    )
      return 0;
  }
  const before = atDistance(map, s - 0.4).p[2],
    here = atDistance(map, s).p[2],
    after = atDistance(map, s + 0.4).p[2];
  const grade = Math.max(Math.abs(here - before), Math.abs(after - here)) / 0.4;
  return Math.max(0, 1 - grade / 0.08);
}
