import { Quaternion } from 'three';
export function sampleReplay(frames, time) {
  let lo = 0,
    hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].now <= time) lo = mid;
    else hi = mid - 1;
  }
  const a = frames[lo],
    b = frames[Math.min(lo + 1, frames.length - 1)];
  const f =
    a === b ? 0 : Math.max(0, Math.min(1, (time - a.now) / (b.now - a.now)));
  return {
    ...a,
    now: time,
    players: a.players.map((p) => {
      const next = b.players.find((x) => x.id === p.id) ?? p;
      const q = new Quaternion(p.q[1], p.q[2], p.q[3], p.q[0]).slerp(
        new Quaternion(next.q[1], next.q[2], next.q[3], next.q[0]),
        f,
      );
      const mix = (key) => p[key].map((v, i) => v + (next[key][i] - v) * f);
      return {
        ...p,
        p: mix('p'),
        j: mix('j'),
        q: [q.w, q.x, q.y, q.z],
        visual: p.visual?.map((v, i) => {
          const w = next.visual[i];
          const rotation = new Quaternion(v.q[1], v.q[2], v.q[3], v.q[0]).slerp(
            new Quaternion(w.q[1], w.q[2], w.q[3], w.q[0]),
            f,
          );
          return {
            id: v.id,
            p: v.p.map((x, k) => x + (w.p[k] - x) * f),
            q: [rotation.w, rotation.x, rotation.y, rotation.z],
          };
        }),
      };
    }),
  };
}
