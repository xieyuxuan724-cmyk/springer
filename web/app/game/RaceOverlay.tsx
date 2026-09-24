import { useGame } from './context';
import { Trophy, Gem, ChevronDown } from 'lucide-react';
import { raceTime as time } from '../../client/selectors';
import { useDisclosure } from './context';
import { Leaderboard } from './Leaderboard';
export function RaceOverlay() {
  const { room, active, countdown, sorted, identity, me, serverNow } =
    useGame();
  const { ref: standings } = useDisclosure();
  if (!room) return null;
  return (
    <>
      {' '}
      {(active || countdown > 0) && (
        <div className="race-hud">
          <details className="race-standings" ref={standings}>
            <summary role="button" aria-label="View standings">
              <strong>
                {Math.max(1, sorted.findIndex((p) => p.id === identity) + 1)}
              </strong>
              <small>/ {sorted.length}</small>
              <ChevronDown size={12} />
            </summary>
            <Leaderboard />
          </details>
          <span className="lap-count" aria-label="Laps">
            <strong>
              {Math.min((me?.lap ?? 0) + 1, 2)}
              <small>/2</small>
            </strong>
            Lap
          </span>
          <time aria-label="Race time">
            {time(
              me?.finished ?? serverNow - room.startAt + (me?.penalty ?? 0),
            )}
          </time>
          <span className="race-score" aria-label={`Score ${me?.gems ?? 0}`}>
            <Gem size={14} />
            <strong>{me?.gems ?? 0}</strong>
          </span>
        </div>
      )}
      {countdown > 0 && (
        <div className="countdown" aria-live="polite">
          <strong>{countdown}</strong>
        </div>
      )}
      {active && me?.finished !== null && (
        <div className="finish-banner">
          <Trophy size={24} />
          <strong>Finished!</strong>
          <span>{time(me?.finished ?? 0)}</span>
        </div>
      )}
    </>
  );
}
