import { getTrack } from './maps.mjs';

// Driving assistance for tests and recordings only: emits the same jump request
// as the space key. Never changes MuJoCo positions, velocities or forces.
export function jumpApproach(map, s, speed, lookahead) {
  const track = getTrack(map);
  const lead = lookahead ?? Math.max(0.35, Math.abs(speed) * 0.25);
  return map.hazards.find((h) => {
    if (!h.requiresJump) return false;
    const i = track.frames.findIndex((f) => Math.abs(f.u - h.u) < 1e-7);
    const entry = track.lengths[i] - h.length / 2;
    const remaining = (entry - s + track.length) % track.length;
    return remaining > 0 && remaining <= lead;
  });
}
