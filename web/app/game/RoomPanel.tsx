import { Button } from '@/components/ui/button';
import { useGame } from './context';
import { Flag, Copy, Check, Trophy, RotateCcw } from 'lucide-react';
import { PolicyCard } from './PolicyCard';
import { SkinControl } from './SkinControl';
import { Leaderboard } from './Leaderboard';
export function RoomPanel() {
  const {
    mode,
    room,
    active,
    countdown,
    send,
    alert,
    loaded,
    fatal,
    engineStatus,
    connected,
    me,
    isHost,
    sorted,
    racing,
  } = useGame();
  if (!room || racing) return null;
  const loading = !loaded && !fatal;
  return (
    <aside
      className={`room-panel panel ${active || countdown > 0 ? 'racing-panel' : ''}`}
    >
      <div className="panel-heading">
        <div>
          <h2>
            {active || countdown > 0
              ? 'Standings'
              : mode === 'single'
                ? 'Single Player'
                : room.code}
            {mode === 'multiplayer' && !active && countdown === 0 && (
              <Button
                title="Copy invite link"
                aria-label="Copy invite link"
                variant="ghost"
                size="icon"
                onClick={async () => {
                  const url = new URL(location.href);
                  url.searchParams.set('room', room.code);
                  try {
                    await navigator.clipboard.writeText(url.href);
                    alert('Invite link copied');
                  } catch {
                    alert(
                      `Room code: ${room.code}. Share this page with a friend and have them enter this code.`,
                    );
                  }
                }}
              >
                <Copy size={16} />
              </Button>
            )}
          </h2>
        </div>
      </div>
      {!active && countdown === 0 && (
        <div className="room-meta">
          <span>
            {mode === 'single'
              ? 'Single Player'
              : `${room.players.length} / 8 drivers`}
          </span>
          <span>2 laps</span>
        </div>
      )}
      {room.phase === 'lobby' && <PolicyCard />}
      <Leaderboard />
      {room.phase === 'lobby' && (
        <>
          <SkinControl />
          <Button
            className="primary"
            disabled={!loaded || !connected}
            aria-busy={loading}
            title={fatal || (loading ? engineStatus : undefined)}
            onClick={() => send({ type: 'ready', ready: !me?.ready })}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true" /> Loading…
              </>
            ) : fatal ? (
              'Loading failed'
            ) : me?.ready ? (
              <>
                <Check size={18} /> Ready · Cancel
              </>
            ) : (
              'Ready up'
            )}
          </Button>
          {isHost ? (
            <Button
              className="start-race"
              disabled={
                !loaded || !room.players.every((p) => p.ready && p.connected)
              }
              onClick={() => send({ type: 'start' })}
            >
              <Flag size={17} /> Start race
            </Button>
          ) : (
            <p className="hint">Ready up, then wait for the host to start.</p>
          )}
        </>
      )}
      {room.phase === 'finished' && (
        <>
          <div className="result">
            <Trophy />
            <strong>
              {sorted[0]?.finished !== null
                ? `${sorted[0]?.name} wins!`
                : 'Time is up. Race over.'}
            </strong>
          </div>
          {isHost ? (
            <Button className="primary" onClick={() => send({ type: 'again' })}>
              Race again <RotateCcw size={16} />
            </Button>
          ) : (
            <p>Waiting for the host to start another race</p>
          )}
        </>
      )}
      {room.phase === 'finished' && (
        <details className="event-feed">
          <summary>Race log</summary>
          {room.events.slice(-3).map((e) => (
            <p key={e.id}>{e.text}</p>
          ))}
        </details>
      )}
    </aside>
  );
}
