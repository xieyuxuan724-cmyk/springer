'use client';
import type { WorkerCommand } from '../shared/protocol';
// Debug tuning updates both the rendered scene and the physics worker.
import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { X, Copy, RotateCcw, Download } from 'lucide-react';
import {
  MAPS,
  getMap,
  getTrack,
  atDistance,
  crossPoint,
  spawn,
} from '@/game/maps.mjs';
import {
  applyTuning,
  defaultTuning,
  readTuning,
  geometrySignature,
  mapSnippet,
} from '@/game/debug-tuning.mjs';
import { SKINS } from '@/game/skins.mjs';
import {
  DEBUG_FLAGS,
  LIGHT_TUNING,
  ROBOT_TUNING,
  TEXTURE_CHANNELS,
  type DebugFlags,
  type LightTuning,
  type RobotTuning,
  type TextureChannel,
  type Viewer,
} from '@/game/viewer';
import { DebugCamera } from '@/game/debug-camera';
import './debug.css';

// Version stored tuning so settings from older course layouts are ignored.
const STORAGE_KEY = 'md_debug_v6';
// Read straight into the initial state: a session's toggles and tuning survive
// a reload, and nothing has to be re-applied from an effect on mount.
function loadSaved() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null') ?? {};
  } catch {
    return {};
  }
}
const CHANNELS: { key: TextureChannel; label: string }[] = [
  { key: 'map', label: 'Color' },
  { key: 'normal', label: 'Normal' },
  { key: 'orm', label: 'ORM / roughness / metalness / AO' },
  { key: 'emissive', label: 'Emissive' },
];
type ColorKey =
  | 'sky'
  | 'ground'
  | 'road'
  | 'railColor'
  | 'railAccent'
  | 'color';
const COLOR_FIELDS: { key: ColorKey; label: string }[] = [
  { key: 'sky', label: 'Sky and fog' },
  { key: 'ground', label: 'Distant ground' },
  { key: 'road', label: 'Road' },
  { key: 'railColor', label: 'Rail color' },
  { key: 'railAccent', label: 'Rail stripes' },
  { key: 'color', label: 'Accent color' },
];
type Hazard = {
  name: string;
  u: number;
  height: number;
  length: number;
  requiresJump?: boolean;
};
type Tuning = Record<ColorKey, string> & {
  width: number;
  railHeight: number;
  friction: number;
  hazards: Hazard[];
};
type Props = {
  viewer: () => Viewer | null;
  post: (msg: WorkerCommand) => void;
  mapId: string;
  onSelectMap: (id: string) => void;
  inRoom: boolean;
  loaded: boolean;
  telemetry: { speed: number; z: number; ratio: number };
  onClose: () => void;
};

function Slider({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="dbg-row">
      <span>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={Number(value.toFixed(4))}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {unit && <em>{unit}</em>}
    </label>
  );
}

function Toggle({
  label,
  value,
  onChange,
  hint,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
}) {
  return (
    <label className="dbg-toggle" title={hint}>
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>{label}</span>
    </label>
  );
}

export default function DebugPanel({
  viewer,
  post,
  mapId,
  onSelectMap,
  inRoom,
  loaded,
  telemetry,
  onClose,
}: Props) {
  const [saved] = useState(loadSaved);
  const [flags, setFlags] = useState<DebugFlags>({
    ...DEBUG_FLAGS,
    ...saved.flags,
  });
  const [lights, setLights] = useState<LightTuning>({
    ...LIGHT_TUNING,
    ...saved.lights,
  });
  const [robot, setRobot] = useState<RobotTuning>({
    ...ROBOT_TUNING,
    ...saved.robot,
  });
  const [channels, setChannels] = useState<Record<TextureChannel, boolean>>({
    ...TEXTURE_CHANNELS,
    ...saved.channels,
  });
  const [tuning, setTuning] = useState<Tuning>(() => {
    const restored: Tuning = {
      ...(readTuning(mapId) as Tuning),
      ...saved.tuning?.[mapId],
    };
    const defaults = defaultTuning(mapId) as Tuning;
    // Keep saved geometry settings, but use the current built-in display names.
    return {
      ...restored,
      hazards: restored.hazards.map((hazard, index) => ({
        ...hazard,
        name: defaults.hazards[index]?.name ?? hazard.name,
      })),
    };
  });
  const [roadRepeat, setRoadRepeat] = useState<{ u: number; v: number }>({
    u: 1,
    v: 1,
    ...saved.roadRepeat,
  });
  const [driving, setDriving] = useState(false);
  const [distance, setDistance] = useState(0);
  const [lateral, setLateral] = useState(0);
  const [stats, setStats] = useState({
    fps: 60,
    triangles: 0,
    calls: 0,
    geometries: 0,
    textures: 0,
  });
  const [message, setMessage] = useState('');
  const camera = useRef<DebugCamera | null>(null);
  const lastGeometry = useRef(geometrySignature(readTuning(mapId) as Tuning));
  const soloRef = useRef(false);
  // Tuning invalidates getTrack's cache, so this read is cheap between edits.
  const trackLength = getTrack(getMap(mapId)).length as number;

  const notify = useCallback((text: string) => {
    setMessage(text);
    setTimeout(() => setMessage(''), 2600);
  }, []);

  // ------------------------------------------------------------ map tuning
  // Colour edits only rebuild the three.js world; width, rail height, friction
  // and hazards also regenerate the MuJoCo prisms, which resets the robot.
  const applyMap = useCallback(
    (next: Tuning) => {
      const v = viewer();
      if (!v) return;
      applyTuning(mapId, next);
      const signature = geometrySignature(next);
      const rebuildPhysics = signature !== lastGeometry.current;
      lastGeometry.current = signature;
      v.setMap(mapId, true);
      if (rebuildPhysics)
        post({
          type: 'map',
          map: mapId,
          tuning: next,
          spawn: spawn(getMap(mapId)),
        });
    },
    [mapId, post, viewer],
  );

  // Debounced so dragging a slider does not rebuild the track every frame. The
  // first pass is what puts a restored session's tuning back onto the map.
  useEffect(() => {
    const t = setTimeout(() => applyMap(tuning), 180);
    return () => clearTimeout(t);
  }, [tuning, applyMap]);

  const setColor = (key: ColorKey, value: string) =>
    setTuning((t) => ({ ...t, [key]: value }));
  const setNumber = (key: 'width' | 'railHeight' | 'friction', value: number) =>
    setTuning((t) => ({ ...t, [key]: value }));
  const setHazard = (index: number, key: keyof Hazard, value: number) =>
    setTuning((t) => ({
      ...t,
      hazards: t.hazards.map((h, i) =>
        i === index ? { ...h, [key]: value } : h,
      ),
    }));

  // ------------------------------------------------------------ view + lights
  useEffect(() => {
    const v = viewer();
    if (!v) return;
    v.debug = { ...flags };
    v.applyDebug();
    if (flags.freeCamera && !camera.current) {
      camera.current = new DebugCamera(
        v.camera,
        v.renderer.domElement,
        v.target,
      );
      v.debugCamera = camera.current;
    }
    if (!flags.freeCamera && camera.current) {
      camera.current.dispose();
      camera.current = null;
      v.debugCamera = null;
    }
    if (camera.current) camera.current.fly = flags.captureKeys;
  }, [flags, viewer]);

  useEffect(() => {
    const v = viewer();
    if (!v) return;
    v.lights = { ...lights };
    v.applyLights();
  }, [lights, viewer]);

  useEffect(() => {
    const v = viewer();
    if (!v) return;
    v.robotTuning = { ...robot };
    v.applyRobotTuning();
  }, [robot, viewer]);

  useEffect(() => {
    const v = viewer();
    if (!v) return;
    for (const key of Object.keys(channels) as TextureChannel[])
      v.setChannel(key, channels[key]);
  }, [channels, viewer]);

  useEffect(() => () => camera.current?.dispose(), []);

  useEffect(() => {
    const t = setInterval(() => {
      const v = viewer();
      if (v) setStats(v.stats());
    }, 500);
    return () => clearInterval(t);
  }, [viewer]);

  // --------------------------------------------------------------- persistence
  // Re-read on every write so the other maps' saved tuning survives.
  useEffect(() => {
    try {
      const stored = loadSaved();
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          flags,
          lights,
          robot,
          channels,
          roadRepeat,
          tuning: { ...stored.tuning, [mapId]: tuning },
        }),
      );
    } catch {}
  }, [flags, lights, robot, channels, roadRepeat, tuning, mapId]);

  // --------------------------------------------------------------- physics
  useEffect(() => {
    if (inRoom) return;
    soloRef.current = driving;
    post({ type: 'running', value: driving && loaded });
  }, [driving, loaded, inRoom, post]);

  // Closing the panel must not leave a solo run stepping in the background.
  useEffect(
    () => () => {
      if (soloRef.current) post({ type: 'running', value: false });
    },
    [post],
  );

  const respawn = () => {
    const map = getMap(mapId),
      f = atDistance(map, distance);
    post({
      type: 'reset',
      spawn: { p: crossPoint(f, lateral, 0.115), yaw: f.yaw },
    });
  };

  const lookAtRobot = () => {
    const p = viewer()?.frame?.p;
    if (!p || !camera.current) return;
    camera.current.lookAt(new THREE.Vector3(p[0], p[1], p[2] + 0.15), 2.5);
  };

  const lookAtTrack = () => {
    const b = getTrack(getMap(mapId)).bounds;
    camera.current?.lookAt(
      new THREE.Vector3((b.minX + b.maxX) / 2, (b.minY + b.maxY) / 2, 0.5),
      Math.max(b.maxX - b.minX, b.maxY - b.minY),
    );
  };

  // --------------------------------------------------------------- export
  const exportSnippet = async () => {
    const text = mapSnippet(mapId);
    try {
      await navigator.clipboard.writeText(text);
      notify('Copied the maps.mjs snippet');
    } catch {
      notify('Clipboard unavailable; downloaded the file instead');
      download(`${mapId}-tuning.txt`, text);
    }
  };
  const exportJSON = () =>
    download(
      `${mapId}-tuning.json`,
      JSON.stringify({ map: mapId, ...tuning }, null, 2),
    );
  function download(name: string, text: string) {
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  }

  const resetAll = () => {
    const fresh = defaultTuning(mapId) as Tuning;
    setTuning(fresh);
    setFlags({ ...DEBUG_FLAGS });
    setLights({ ...LIGHT_TUNING });
    setRobot({ ...ROBOT_TUNING });
    setChannels({ ...TEXTURE_CHANNELS });
    setRoadRepeat({ u: 1, v: 1 });
    const v = viewer();
    if (v) {
      v.setRoadTexture(null);
      for (const c of CHANNELS) void v.setRobotTexture(c.key, null);
    }
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {}
    notify('Restored map and rendering defaults, including custom textures');
  };

  const pickTexture = async (channel: TextureChannel, file: File | null) => {
    const v = viewer();
    if (!v) return;
    try {
      await v.setRobotTexture(channel, file);
      // The extra channels ship off, so an uploaded map switches its own on;
      // otherwise the swap would land on a material that ignores it.
      if (file) setChannels((s) => ({ ...s, [channel]: true }));
      notify(
        file ? `${channel} replaced with ${file.name}` : `${channel} restored`,
      );
    } catch (e) {
      notify((e as Error).message);
    }
  };

  const pickRoadTexture = async (file: File | null) => {
    const v = viewer();
    if (!v) return;
    if (!file) {
      v.setRoadTexture(null);
      notify('Road texture removed');
      return;
    }
    try {
      await v.loadRoadTexture(file, roadRepeat.u, roadRepeat.v);
      notify(`Road texture: ${file.name}`);
    } catch (e) {
      notify((e as Error).message);
    }
  };

  return (
    <aside className="dbg-panel" aria-label="Debug panel">
      <header>
        <strong>Debug panel</strong>
        <span className="dbg-fps">
          {stats.fps.toFixed(0)} fps · {(stats.triangles / 1000).toFixed(0)}k
          tri · {stats.calls} calls
        </span>
        <button type="button" onClick={resetAll} title="Reset all defaults">
          <RotateCcw size={14} />
        </button>
        <button type="button" onClick={onClose} title="Close (F2)">
          <X size={16} />
        </button>
      </header>

      <details open>
        <summary>View</summary>
        <div className="dbg-grid">
          <Toggle
            label="Free camera"
            value={flags.freeCamera}
            hint="Left-drag to orbit, right-drag or use two fingers to pan, scroll to zoom"
            onChange={(v) => setFlags((f) => ({ ...f, freeCamera: v }))}
          />
          <Toggle
            label="Keyboard flight"
            value={flags.captureKeys}
            hint="WASD to move, Q/E up/down, Shift for speed; robot controls are disabled while flying"
            onChange={(v) => setFlags((f) => ({ ...f, captureKeys: v }))}
          />
          <Toggle
            label="Wireframe"
            value={flags.wireframe}
            onChange={(v) => setFlags((f) => ({ ...f, wireframe: v }))}
          />
          <Toggle
            label="Collision prisms"
            value={flags.collision}
            hint="Actual MuJoCo collision geometry; it differs slightly from the rendered surface"
            onChange={(v) => setFlags((f) => ({ ...f, collision: v }))}
          />
          <Toggle
            label="Grid"
            value={flags.grid}
            onChange={(v) => setFlags((f) => ({ ...f, grid: v }))}
          />
          <Toggle
            label="Axes"
            value={flags.axes}
            onChange={(v) => setFlags((f) => ({ ...f, axes: v }))}
          />
          <Toggle
            label="Shadows"
            hint="Also select Medium or High effects in Settings. Low and Off do not render shadows."
            value={flags.shadows}
            onChange={(v) => setFlags((f) => ({ ...f, shadows: v }))}
          />
        </div>
        <div className="dbg-buttons">
          <button
            type="button"
            disabled={!flags.freeCamera}
            onClick={lookAtRobot}
          >
            Look at robot
          </button>
          <button
            type="button"
            disabled={!flags.freeCamera}
            onClick={lookAtTrack}
          >
            Frame course
          </button>
        </div>
      </details>

      <details open>
        <summary>Course and scene</summary>
        <label className="dbg-row">
          <span>Course</span>
          <select value={mapId} onChange={(e) => onSelectMap(e.target.value)}>
            {MAPS.map((m: { id: string; name: string }) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </label>
        <Slider
          label="Road width"
          unit="m"
          min={1.2}
          max={4}
          step={0.05}
          value={tuning.width}
          onChange={(v) => setNumber('width', v)}
        />
        <Slider
          label="Rail height"
          unit="m"
          min={0.1}
          max={1}
          step={0.01}
          value={tuning.railHeight}
          onChange={(v) => setNumber('railHeight', v)}
        />
        <Slider
          label="Road friction"
          min={0.05}
          max={1.5}
          step={0.05}
          value={tuning.friction}
          onChange={(v) => setNumber('friction', v)}
        />
        <div className="dbg-colors">
          {COLOR_FIELDS.map((c) => (
            <label key={c.key}>
              <input
                type="color"
                value={tuning[c.key]}
                onChange={(e) => setColor(c.key, e.target.value)}
              />
              <span>{c.label}</span>
            </label>
          ))}
        </div>
        <Slider
          label="Fog start"
          unit="m"
          min={1}
          max={120}
          step={1}
          value={lights.fogNear}
          onChange={(v) => setLights((l) => ({ ...l, fogNear: v }))}
        />
        <Slider
          label="Fog end"
          unit="m"
          min={5}
          max={400}
          step={5}
          value={lights.fogFar}
          onChange={(v) => setLights((l) => ({ ...l, fogFar: v }))}
        />
        <p className="dbg-note">
          Width, rail height, friction and obstacles rebuild MuJoCo collision
          geometry and reset the robot. Color changes only affect rendering.
        </p>
        <div className="dbg-buttons">
          <button type="button" onClick={exportSnippet}>
            <Copy size={13} /> Copy maps.mjs snippet
          </button>
          <button type="button" onClick={exportJSON}>
            <Download size={13} /> Export JSON
          </button>
        </div>
      </details>

      <details>
        <summary>Bumps and pits</summary>
        {tuning.hazards.map((h, i) => (
          <fieldset key={h.name} className="dbg-hazard">
            <legend>{h.name}</legend>
            <Slider
              label="Position u"
              min={0}
              max={(getMap(mapId).points as number[][]).length}
              step={0.05}
              value={h.u}
              onChange={(v) => setHazard(i, 'u', v)}
            />
            <Slider
              label="Height change"
              unit="m"
              min={-0.1}
              max={0.1}
              step={0.005}
              value={h.height}
              onChange={(v) => setHazard(i, 'height', v)}
            />
            <Slider
              label="Length"
              unit="m"
              min={0.2}
              max={3}
              step={0.05}
              value={h.length}
              onChange={(v) => setHazard(i, 'length', v)}
            />
          </fieldset>
        ))}
      </details>

      <details>
        <summary>Lighting and materials</summary>
        <Slider
          label="Exposure"
          min={0.2}
          max={3}
          step={0.05}
          value={lights.exposure}
          onChange={(v) => setLights((l) => ({ ...l, exposure: v }))}
        />
        <Slider
          label="Sun intensity"
          min={0}
          max={10}
          step={0.1}
          value={lights.sun}
          onChange={(v) => setLights((l) => ({ ...l, sun: v }))}
        />
        <Slider
          label="Sun azimuth"
          unit="°"
          min={-180}
          max={180}
          step={1}
          value={lights.sunAzimuth}
          onChange={(v) => setLights((l) => ({ ...l, sunAzimuth: v }))}
        />
        <Slider
          label="Sun elevation"
          unit="°"
          min={2}
          max={89}
          step={1}
          value={lights.sunElevation}
          onChange={(v) => setLights((l) => ({ ...l, sunElevation: v }))}
        />
        <Slider
          label="Ambient light"
          min={0}
          max={6}
          step={0.1}
          value={lights.ambient}
          onChange={(v) => setLights((l) => ({ ...l, ambient: v }))}
        />
        <Slider
          label="Body roughness"
          min={0}
          max={1}
          step={0.02}
          value={robot.roughness}
          onChange={(v) => setRobot((r) => ({ ...r, roughness: v }))}
        />
        <Slider
          label="Body metalness"
          min={0}
          max={1}
          step={0.02}
          value={robot.metalness}
          onChange={(v) => setRobot((r) => ({ ...r, metalness: v }))}
        />
        <Slider
          label="Emissive intensity"
          min={0}
          max={3}
          step={0.05}
          value={robot.emissive}
          onChange={(v) => setRobot((r) => ({ ...r, emissive: v }))}
        />
      </details>

      <details>
        <summary>Live textures</summary>
        {CHANNELS.map((c) => (
          <div key={c.key} className="dbg-texture">
            <Toggle
              label={c.label}
              value={channels[c.key]}
              onChange={(v) => setChannels((s) => ({ ...s, [c.key]: v }))}
            />
            <input
              type="file"
              accept="image/*"
              aria-label={`Replace ${c.label}`}
              onChange={(e) => {
                void pickTexture(c.key, e.target.files?.[0] ?? null);
                e.target.value = '';
              }}
            />
            <button type="button" onClick={() => void pickTexture(c.key, null)}>
              Restore
            </button>
          </div>
        ))}
        <div className="dbg-texture">
          <span>Road texture</span>
          <input
            type="file"
            accept="image/*"
            aria-label="Road texture"
            onChange={(e) => {
              void pickRoadTexture(e.target.files?.[0] ?? null);
              e.target.value = '';
            }}
          />
          <button type="button" onClick={() => void pickRoadTexture(null)}>
            Remove
          </button>
        </div>
        <Slider
          label="Horizontal tiling"
          unit="/m"
          min={0.05}
          max={8}
          step={0.05}
          value={roadRepeat.u}
          onChange={(u) => {
            setRoadRepeat((r) => ({ ...r, u }));
            const view = viewer();
            if (view) view.setRoadTexture(view.roadTexture, u, roadRepeat.v);
          }}
        />
        <Slider
          label="Vertical tiling"
          unit="/m"
          min={0.05}
          max={8}
          step={0.05}
          value={roadRepeat.v}
          onChange={(v) => {
            setRoadRepeat((r) => ({ ...r, v }));
            const view = viewer();
            if (view) view.setRoadTexture(view.roadTexture, roadRepeat.u, v);
          }}
        />
        <p className="dbg-note">
          Texture overrides last for this session. Refresh to restore files from
          assets/beni_base/textures. Road UVs use meters; tiling sets repeats per
          meter.
        </p>
      </details>

      <details open>
        <summary>Physics</summary>
        <div className="dbg-buttons">
          <button
            type="button"
            disabled={inRoom || !loaded}
            onClick={() => setDriving((d) => !d)}
            title={
              inRoom
                ? 'Controlled by the server while in a room'
                : 'Drive without creating a room'
            }
          >
            {driving ? 'Pause' : 'Solo drive'}
          </button>
          <button
            type="button"
            disabled={inRoom || driving || !loaded}
            onClick={() => post({ type: 'step', count: 1 })}
          >
            Step
          </button>
          <button
            type="button"
            disabled={inRoom || driving || !loaded}
            onClick={() => post({ type: 'step', count: 20 })}
          >
            Step 20
          </button>
          <button type="button" disabled={!loaded} onClick={respawn}>
            Respawn here
          </button>
        </div>
        <Slider
          label="Start distance"
          unit="m"
          min={0}
          max={Math.max(1, Math.round(trackLength))}
          step={0.5}
          value={distance}
          onChange={setDistance}
        />
        <Slider
          label="Lateral offset"
          unit="m"
          min={-(tuning.width / 2)}
          max={tuning.width / 2}
          step={0.05}
          value={lateral}
          onChange={setLateral}
        />
        <label className="dbg-row">
          <span>Skin</span>
          {/* Local only: swaps this robot's colour atlas without telling the
              server, so all eight paint jobs can be checked solo. */}
          <select
            disabled={inRoom}
            defaultValue=""
            onChange={(e) => viewer()?.setSkin(e.target.value)}
          >
            <option value="" disabled>
              Choose a skin
            </option>
            {SKINS.map((s: { id: string; name: string }) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        </label>
        <dl className="dbg-readout">
          <div>
            <dt>Speed</dt>
            <dd>{telemetry.speed.toFixed(2)} m/s</dd>
          </div>
          <div>
            <dt>Body height</dt>
            <dd>{telemetry.z.toFixed(3)} m</dd>
          </div>
          <div>
            <dt>Realtime ratio</dt>
            <dd>{telemetry.ratio.toFixed(2)}×</dd>
          </div>
          <div>
            <dt>Course length</dt>
            <dd>{trackLength.toFixed(1)} m</dd>
          </div>
        </dl>
      </details>

      {message && <p className="dbg-message">{message}</p>}
    </aside>
  );
}
