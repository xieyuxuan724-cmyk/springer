import { getMap, getTrack, routePolylines } from '../game/maps.mjs';
import type { GameState } from './game-session.ts';
export function raceTime(ms: number) {
  return `${Math.floor(Math.max(0, ms) / 60000)
    .toString()
    .padStart(
      2,
      '0',
    )}:${((Math.max(0, ms) % 60000) / 1000).toFixed(2).padStart(5, '0')}`;
}
export function selectGame(state: GameState) {
  const { room, identity, engine } = state;
  const me = room?.players.find((p) => p.id === identity);
  const selected = getMap(state.mapId);
  const serverNow = state.now + state.clockOffset;
  const active = room?.phase === 'racing';
  const countdown =
    room?.phase === 'countdown'
      ? Math.max(1, Math.ceil((room.startAt - serverNow) / 1000))
      : 0;
  const sorted = room
    ? [...room.players].sort((a, b) =>
        a.finished !== null
          ? b.finished !== null
            ? a.finished - b.finished
            : -1
          : b.finished !== null
            ? 1
            : b.lap - a.lap ||
              (b.next || getMap(room.map).points.length) -
                (a.next || getMap(room.map).points.length),
      )
    : [];
  return {
    ...state,
    loaded: engine.phase === 'ready',
    engineStatus: engine.status,
    policyLabel: engine.label,
    jumpSupported: engine.jumpSupported,
    fatal: engine.error,
    me,
    selected,
    serverNow,
    active,
    countdown,
    racing: active || countdown > 0,
    sorted,
    isHost: room?.host === identity,
    skinId: me?.skin ?? state.skin,
    takenSkins: (room?.players ?? [])
      .filter((p) => p.id !== identity)
      .map((p) => p.skin),
  };
}
export function minimapGeometry(mapId: string) {
  const map = getMap(mapId);
  return {
    minimapBounds: getTrack(map).bounds,
    minimapPoints: routePolylines(map).map((points) =>
      points.map((p) => p.slice(0, 2).join(',')).join(' '),
    ),
  };
}
