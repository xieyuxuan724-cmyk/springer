import { Button } from '@/components/ui/button';
import { useGame } from './context';
import { useState } from 'react';
import { Input } from '@/components/ui/input';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { MAPS, getMap, routePolylines } from '../../game/maps.mjs';
import { Check, ArrowRight, Users } from 'lucide-react';
import { PolicyCard } from './PolicyCard';
import { SkinControl } from './SkinControl';
export function GaragePanel() {
  const {
    mapId,
    changeMap,
    connected,
    loaded,
    send,
    rooms,
    mode,
    multiplayerEnabled,
    session,
    connectionSleeping,
  } = useGame();
  const [joinCode, setJoinCode] = useState(
    () =>
      new URL(location.href).searchParams
        .get('room')
        ?.toUpperCase()
        .slice(0, 5) ?? '',
  );
  return (
    <aside className="garage-panel panel">
      {multiplayerEnabled && (
        <div className="mode-options" role="group" aria-label="Game mode">
          <Button
            variant={mode === 'single' ? 'default' : 'outline'}
            aria-pressed={mode === 'single'}
            onClick={() => session.setMode('single')}
          >
            Single Player
          </Button>
          <Button
            variant={mode === 'multiplayer' ? 'default' : 'outline'}
            aria-pressed={mode === 'multiplayer'}
            onClick={() => session.setMode('multiplayer')}
          >
            Multiplayer
          </Button>
        </div>
      )}
      {mode === 'multiplayer' && connectionSleeping && (
        <Button onClick={session.reconnect}>Reconnect</Button>
      )}
      <PolicyCard />
      <div className="panel-heading">
        <div>
          <h2>Choose a course</h2>
        </div>
      </div>
      <RadioGroup
        value={mapId}
        onValueChange={(v) => changeMap(String(v))}
        className="map-options"
        aria-label="Course"
      >
        {MAPS.map((m) => (
          <label
            key={m.id}
            className={`map-card ${m.id === mapId ? 'selected' : ''}`}
            style={{ '--map-color': m.color } as React.CSSProperties}
          >
            <RadioGroupItem value={m.id} className="sr-only" />
            <svg
              className="course-outline"
              viewBox="0 0 100 60"
              aria-hidden="true"
            >
              {routePolylines(m).map((points, i) => {
                const all = routePolylines(m).flat();
                const xs = all.map((p) => p[0]),
                  ys = all.map((p) => p[1]);
                const minX = Math.min(...xs),
                  minY = Math.min(...ys);
                const scale = Math.min(
                  86 / (Math.max(...xs) - minX),
                  46 / (Math.max(...ys) - minY),
                );
                return (
                  <polyline
                    key={i}
                    points={points
                      .map(
                        (p) =>
                          `${7 + (p[0] - minX) * scale},${53 - (p[1] - minY) * scale}`,
                      )
                      .join(' ')}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="4"
                    strokeLinejoin="round"
                  />
                );
              })}
            </svg>
            <span>
              <strong>{m.name}</strong>
            </span>
            {m.id === mapId && <Check size={17} />}
          </label>
        ))}
      </RadioGroup>
      <SkinControl />
      <Button
        className="primary"
        disabled={!connected || !loaded}
        onClick={() =>
          mode === 'single'
            ? session.startSingle()
            : send({ type: 'create', map: mapId })
        }
      >
        {mode === 'single' ? 'Start game' : 'Create room'}{' '}
        <ArrowRight size={17} />
      </Button>
      {mode === 'multiplayer' && (
        <>
          <form
            className="join-form"
            onSubmit={(e) => {
              e.preventDefault();
              send({ type: 'join', code: joinCode });
            }}
          >
            <Input
              aria-label="Room code"
              placeholder="Enter room code"
              value={joinCode}
              onChange={(e) => setJoinCode(e.target.value.toUpperCase())}
              maxLength={5}
            />
            <Button
              variant="outline"
              type="submit"
              disabled={!connected || joinCode.length !== 5 || !loaded}
            >
              Join
            </Button>
          </form>
          <div className="room-list">
            <span className="section-label">
              <Users size={14} /> Open rooms <span>{rooms.length}</span>
            </span>
            {rooms.length ? (
              rooms.slice(0, 4).map((r) => (
                <Button
                  variant="ghost"
                  className="room-row"
                  key={r.code}
                  disabled={r.phase !== 'lobby' || r.count >= 8 || !loaded}
                  onClick={() => send({ type: 'join', code: r.code })}
                >
                  <span>
                    {r.code} · {getMap(r.map).name}
                  </span>
                  <small>
                    {r.count}/8 {r.phase !== 'lobby' ? 'Racing' : 'Join →'}
                  </small>
                </Button>
              ))
            ) : (
              <small className="hint">No rooms yet</small>
            )}
          </div>
        </>
      )}
    </aside>
  );
}
