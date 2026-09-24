import { Button } from '@/components/ui/button';
import { useGame } from './context';
import { TriangleAlert } from 'lucide-react';
import DebugPanel from '../debug-panel';
export function GameFeedback() {
  const {
    session,
    idleWarning,
    fatal,
    notice,
    setNotice,
    loaded,
    debugOpen,
    mapId,
    getViewer,
    post,
    changeMap,
    room,
    telemetry,
    setDebugOpen,
  } = useGame();
  return (
    <>
      {' '}
      {fatal && (
        <div className="error-panel" role="alert">
          <TriangleAlert />
          <strong>Engine recovery required</strong>
          <p>{fatal}</p>
          <Button onClick={() => {
            const url = new URL(location.href); url.searchParams.delete('assetSource'); location.assign(url.href);
          }}>Retry CDN</Button>
          <Button variant="outline" onClick={() => {
            const url = new URL(location.href); url.searchParams.set('assetSource', 'local'); location.assign(url.href);
          }}>Try backup source (may be slower)</Button>
        </div>
      )}
      {(notice || idleWarning) && (
        <div className="toast" role="status">
          {notice || idleWarning}
          {notice ? (
            <button
              aria-label="Dismiss notification"
              onClick={() => setNotice('')}
            >
              ×
            </button>
          ) : (
            <Button variant="ghost" onClick={session.activity}>
              Stay connected
            </Button>
          )}
        </div>
      )}
      {debugOpen && (
        <DebugPanel
          // Remounting per map is how the panel picks up that map's tuning.
          key={mapId}
          viewer={getViewer}
          post={post}
          mapId={mapId}
          onSelectMap={changeMap}
          inRoom={!!room}
          loaded={loaded}
          telemetry={telemetry}
          onClose={() => setDebugOpen(false)}
        />
      )}
    </>
  );
}
