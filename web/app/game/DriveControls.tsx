import { Button } from '@/components/ui/button';
import { useGame } from './context';
import { ArrowUp, RotateCcw, Zap } from 'lucide-react';
import { VirtualJoystick } from './VirtualJoystick';
export function DriveControls() {
  const { active, telemetry, me, serverNow, jumpSupported, post, send } =
    useGame();
  // Secondary touches do not consistently synthesize click events. Trigger on
  // pointer-down, retaining detail=0 clicks for keyboard/assistive activation.
  const action = (run: () => void) => ({
    onPointerDown: (event: React.PointerEvent<HTMLButtonElement>) => {
      if (event.button !== 0) return;
      event.preventDefault();
      run();
    },
    onClick: (event: React.MouseEvent<HTMLButtonElement>) => {
      if (event.detail === 0) run();
    },
  });
  if (!active) return null;
  return (
    <>
      <div className="drive-hud">
        <div className="drive-stats">
          <div className="speed">
            <strong>{telemetry.speed.toFixed(1)}</strong>
            <span>m/s</span>
          </div>
          {(me?.effects.boost ?? 0) > serverNow && (
            <Zap size={16} aria-label="Boosting" className="boost-icon" />
          )}
        </div>
        <div className="drive-actions">
          <Button
            variant="outline"
            className="jump-action"
            aria-label="Jump"
            aria-keyshortcuts="Space"
            title="Space to jump"
            disabled={!jumpSupported}
            onClick={() => post({ type: 'jump' })}
          >
            <span>Jump</span>
            <kbd>Space</kbd>
          </Button>
        </div>
      </div>
      <div className="touch-controls">
        <VirtualJoystick />
        <div className="touch-actions">
          <Button
            className="touch-reset"
            aria-label="Reset"
            aria-keyshortcuts="R"
            title="Reset to the course (+3 s)"
            {...action(() => send({ type: 'rescue' }))}
          >
            <RotateCcw size={17} />
            <span>Reset</span>
          </Button>
          <Button
            className="touch-jump"
            aria-label="Jump"
            aria-keyshortcuts="Space"
            disabled={!jumpSupported}
            title={
              jumpSupported ? 'Jump' : 'This policy does not support jumping'
            }
            {...action(() => post({ type: 'jump' }))}
          >
            <ArrowUp size={22} />
            <span>Jump</span>
          </Button>
        </div>
      </div>
    </>
  );
}
