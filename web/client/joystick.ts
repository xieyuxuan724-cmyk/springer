export type StickPosition = { x: number; y: number };
export type DriveAxes = { forward: number; turn: number };

/** Circular travel limit with a radial dead zone; screen up drives forward. */
export function clampStick(x: number, y: number): StickPosition {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return { x: 0, y: 0 };
  const length = Math.max(1, Math.hypot(x, y));
  return { x: x / length, y: y / length };
}
export function joystickAxes(stick: StickPosition): DriveAxes {
  const { x, y } = clampStick(stick.x, stick.y);
  const magnitude = Math.hypot(x, y),
    deadZone = 0.12;
  if (magnitude <= deadZone) return { forward: 0, turn: 0 };
  const gain = (magnitude - deadZone) / (1 - deadZone) / magnitude;
  return { forward: y === 0 ? 0 : -y * gain, turn: x === 0 ? 0 : -x * gain };
}
