import { useMemo } from 'react';
import { useGame } from './context';
import { getSkin } from '../../game/skins.mjs';
import { minimapGeometry } from '../../client/selectors';
export function Minimap() {
  const { mapId, room, showMinimap, racing, selected, identity } = useGame();
  const { minimapBounds, minimapPoints } = useMemo(
    () => minimapGeometry(mapId),
    [mapId],
  );
  if (!room || !showMinimap || !racing) return null;
  return (
    <div className="minimap">
      <svg
        viewBox={`${minimapBounds.minX} ${minimapBounds.minY} ${minimapBounds.maxX - minimapBounds.minX} ${minimapBounds.maxY - minimapBounds.minY}`}
        role="img"
        aria-label="Course and player positions"
      >
        {minimapPoints.map((points, i) => (
          <polyline
            key={'road' + i}
            points={points}
            fill="none"
            stroke="#49626a"
            strokeWidth={selected.width}
            strokeLinejoin="round"
          />
        ))}
        {minimapPoints.map((points, i) => (
          <polyline
            key={'line' + i}
            points={points}
            fill="none"
            stroke={selected.color}
            strokeWidth=".04"
          />
        ))}
        {room.players.map((p) => (
          <circle
            key={p.id}
            cx={p.p?.[0] ?? 0}
            cy={p.p?.[1] ?? 0}
            r={p.id === identity ? 0.28 : 0.22}
            fill={getSkin(p.skin).primary}
            stroke={p.id === identity ? '#fff' : 'none'}
            strokeWidth={0.06}
          />
        ))}
      </svg>
    </div>
  );
}
