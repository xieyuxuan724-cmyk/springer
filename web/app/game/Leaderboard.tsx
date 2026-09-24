import { useGame } from './context';
import { getSkin } from '../../game/skins.mjs';
import { raceTime as time } from '../../client/selectors';
export function Leaderboard() {
  const { sorted, identity, room } = useGame();
  return (
    <ol className="leaderboard">
      {sorted.map((p, i) => (
        <li key={p.id} className={p.id === identity ? 'self' : ''}>
          <span className="position">{i + 1}</span>
          <span className="player-name">
            <span style={{ color: getSkin(p.skin).primary }}>●</span>
            {p.name}
            {p.id === identity && <small>You</small>}
            {p.id === room?.host && room?.phase === 'lobby' && (
              <small>Host</small>
            )}
          </span>
          <span className="player-state">
            {!p.connected
              ? 'Offline'
              : p.finished !== null
                ? time(p.finished)
                : room?.phase === 'lobby'
                  ? p.ready
                    ? 'Ready'
                    : 'Waiting'
                  : `${Math.min(p.lap + 1, 2)}/2 laps`}
          </span>
        </li>
      ))}
    </ol>
  );
}
