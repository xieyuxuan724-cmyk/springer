import { Button } from '@/components/ui/button';
import { useGame, useDisclosure } from './context';
import { PolicyCard } from './PolicyCard';
import { EFFECT_LEVELS, effectsQuality } from '../../game/race-effects';
import { Settings2, Camera, RotateCcw } from 'lucide-react';
export function SettingsMenu() {
  const {
    effectLevel,
    setEffectLevel,
    jumpSupported,
    active,
    cameraOverview,
    session,
    showMinimap,
    setShowMinimap,
    showNames,
    setShowNames,
    send,
    room,
  } = useGame();
  const { ref: settings, close } = useDisclosure();
  return (
    <details className="settings-menu" ref={settings}>
      <summary role="button" aria-label="Settings">
        <Settings2 size={17} />
        <span>Settings</span>
      </summary>
      <div className="settings-panel">
        <label className="effects-control">
          <span>Effects quality</span>
          <select
            aria-label="Effects quality"
            value={effectLevel}
            onChange={(e) => setEffectLevel(effectsQuality(e.target.value))}
          >
            {Object.entries(EFFECT_LEVELS).map(([id, preset]) => (
              <option key={id} value={id}>
                {preset.label}
              </option>
            ))}
          </select>
        </label>
        <div className="control-help">
          <span>
            <kbd>W A S D</kbd>Drive
          </span>
          <span>
            <kbd>Space</kbd>
            {jumpSupported ? 'Jump' : 'Jump unavailable'}
          </span>
          <span>
            <kbd>R</kbd>Rescue (+3 s)
          </span>
        </div>
        <p className="pickup-help">Gems score points and may grant a boost.</p>
        {active && (
          <Button
            variant="ghost"
            className="camera-control"
            onClick={session.toggleCamera}
          >
            <Camera size={16} />
            {cameraOverview
              ? 'Switch to chase view'
              : 'Switch to overhead view'}
            <kbd>C</kbd>
          </Button>
        )}
        <div className="display-options">
          <label>
            <span>Show minimap</span>
            <input
              type="checkbox"
              checked={showMinimap}
              onChange={(e) => setShowMinimap(e.target.checked)}
            />
          </label>
          <label>
            <span>Show driver names</span>
            <input
              type="checkbox"
              checked={showNames}
              onChange={(e) => setShowNames(e.target.checked)}
            />
          </label>
        </div>
        {active && (
          <Button
            variant="ghost"
            className="settings-action"
            onClick={() => {
              send({ type: 'rescue' });
              close();
            }}
          >
            <RotateCcw size={15} />
            Rescue to course<small>+3 s · R</small>
          </Button>
        )}
        {room && room.phase !== 'lobby' && <PolicyCard />}
      </div>
    </details>
  );
}
