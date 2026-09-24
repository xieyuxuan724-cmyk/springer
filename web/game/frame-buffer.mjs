// Render slightly behind the physics clock, interpolating instead of stepping
// between Worker messages. Rescues/model changes begin a new timeline.
export class FrameBuffer {
  constructor(delay = 50) {
    this.delay = delay;
    this.frames = [];
  }
  clear() {
    this.frames.length = 0;
  }
  push(frame) {
    const last = this.frames.at(-1);
    if (last && (last.epoch !== frame.epoch || frame.time <= last.time))
      this.clear();
    this.frames.push(frame);
    if (this.frames.length > 16) this.frames.shift();
  }
  sample(now) {
    const target = now - this.delay;
    while (this.frames.length > 2 && this.frames[1].time <= target)
      this.frames.shift();
    const a = this.frames[0];
    if (!a) return null;
    const b = this.frames[1] ?? a;
    const alpha =
      a === b
        ? 0
        : Math.max(0, Math.min(1, (target - a.time) / (b.time - a.time)));
    return { a, b, alpha };
  }
}
