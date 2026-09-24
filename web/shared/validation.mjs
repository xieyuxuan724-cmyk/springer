/** Shared wire validation; game rules remain the RaceService's responsibility. */
export function cleanName(value) {
  if (typeof value !== 'string') throw Error('Enter a driver name');
  const name = value.replace(/[\p{Cc}\p{Cf}]/gu, '').trim();
  if (!name || Array.from(name).length > 20)
    throw Error('Driver names must contain 1–20 characters');
  return name;
}
export function validPose(msg) {
  return (
    Array.isArray(msg.p) &&
    msg.p.length === 3 &&
    msg.p.every((v) => Number.isFinite(v) && Math.abs(v) <= 100) &&
    Array.isArray(msg.q) &&
    msg.q.length === 4 &&
    msg.q.every(Number.isFinite) &&
    Math.abs(Math.hypot(...msg.q) - 1) <= 0.1 &&
    Array.isArray(msg.j) &&
    msg.j.length === 7 &&
    msg.j.every((v) => Number.isFinite(v) && Math.abs(v) < 1e7)
  );
}
/** @returns {import('./protocol.ts').ClientMessage} */
export function parseClientMessage(text) {
  const msg = JSON.parse(text);
  if (!msg || typeof msg !== 'object' || Array.isArray(msg))
    throw Error('Invalid message');
  const stringFields = {
    hello: 'name',
    rename: 'name',
    create: 'map',
    join: 'code',
    skin: 'skin',
  };
  if (Object.hasOwn(stringFields, msg.type)) {
    if (typeof msg[stringFields[msg.type]] !== 'string')
      throw Error('Invalid message');
    if (
      msg.type === 'hello' &&
      msg.token != null &&
      typeof msg.token !== 'string'
    )
      throw Error('Invalid message');
  } else if (msg.type === 'ping') {
    if (!Number.isFinite(msg.sent)) throw Error('Invalid message');
  } else if (msg.type === 'ready') {
    if (typeof msg.ready !== 'boolean') throw Error('Invalid message');
  } else if (msg.type === 'state') {
    if (!validPose(msg)) throw Error('Invalid pose');
  } else if (
    !['activity', 'list', 'leave', 'start', 'again', 'rescue', 'use'].includes(
      msg.type,
    )
  ) {
    throw Error('Unknown action');
  }
  return msg;
}
