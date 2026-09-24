import { LocalRace } from './local-race.ts';
import type { Viewer } from '../game/viewer';
import type { EffectsQuality } from '../game/race-effects';
import type {
  ClientMessage,
  ServerMessage,
  WorkerEvent,
  WorkerCommand,
  Room,
  RoomListing,
} from '../shared/protocol.ts';
import { DEFAULT_MAP_ID, getMap, spawn } from '../game/maps.mjs';
import { SimulationClient, type WorkerPort } from './simulation-client.ts';
import { RoomConnection, type SocketPort } from './room-connection.ts';
import { DriveInput, drivingCommand } from './drive-input.ts';
import {
  readName,
  saveName,
  readPreference,
  savePreference,
} from './preferences.ts';
import { cleanName } from '../shared/validation.mjs';

type Engine = {
  phase: 'starting' | 'loading-policy' | 'loading-map' | 'ready' | 'failed';
  status: string;
  label: string;
  jumpSupported: boolean;
  error: string;
};
export type GameMode = 'single' | 'multiplayer';
export type GameState = {
  mode: GameMode;
  multiplayerEnabled: boolean;
  idleWarning: string;
  connectionSleeping: boolean;
  name: string;
  connected: boolean;
  identity: string;
  skin: string;
  mapId: string;
  rooms: RoomListing[];
  room: Room | null;
  engine: Engine;
  notice: string;
  now: number;
  clockOffset: number;
  ping: number;
  telemetry: { speed: number; z: number; ratio: number };
  cameraOverview: boolean;
  debugOpen: boolean;
  effectLevel: EffectsQuality;
  showMinimap: boolean;
  showNames: boolean;
};
type Scene = Pick<
  Viewer,
  | 'setModel'
  | 'setFrame'
  | 'setMap'
  | 'setSkin'
  | 'setPlayerNames'
  | 'setEffectsQuality'
  | 'updateRoom'
  | 'dispose'
  | 'preview'
  | 'debug'
>;
export type SessionDependencies = {
  createWorker: (mode: GameMode) => WorkerPort;
  createSocket: () => SocketPort;
  createViewer: (element: HTMLElement, quality: EffectsQuality) => Scene;
};
/** Application boundary: the only owner of room snapshots and engine readiness.
 * Physics frames go straight to the scene; React only sees throttled HUD data.
 */
export class GameSession {
  private state: GameState;
  private listeners = new Set<() => void>();
  private scene: Scene | null = null;
  private simulation: SimulationClient;
  private connection: RoomConnection;
  readonly input = new DriveInput();
  private dependencies: SessionDependencies;
  private timer: ReturnType<typeof setInterval> | undefined;
  private noticeTimer: ReturnType<typeof setTimeout> | undefined;
  private integration: AbortController | undefined;
  private mounted = false;
  private lastFrame: { wall: number; sim: number; epoch: number } | null = null;
  private lastSent = 0;
  private lastHud = 0;
  private leavingRoom = false;
  private local: LocalRace | null = null;
  private startSoloWhenReady = false;
  private inviteChecked = false;
  private lastActivitySent = -Infinity;
  constructor(
    dependencies: SessionDependencies,
    quality: EffectsQuality = 'high',
  ) {
    this.dependencies = dependencies;
    this.state = {
      mode: 'single',
      multiplayerEnabled: false,
      idleWarning: '',
      connectionSleeping: false,
      name: '',
      connected: false,
      identity: '',
      skin: '',
      mapId: DEFAULT_MAP_ID,
      rooms: [],
      room: null,
      engine: {
        phase: 'starting',
        status: 'Preparing physics…',
        label: 'Loading policy configuration…',
        jumpSupported: false,
        error: '',
      },
      notice: '',
      now: Date.now(),
      clockOffset: 0,
      ping: 0,
      telemetry: { speed: 0, z: 0.11, ratio: 1 },
      cameraOverview: false,
      debugOpen: false,
      effectLevel: quality,
      showMinimap: false,
      showNames: false,
    };
    this.simulation = new SimulationClient(
      () => dependencies.createWorker(this.state.mode),
      this.onWorker,
    );
    this.connection = new RoomConnection({
      createSocket: dependencies.createSocket,
      hello: () => ({
        type: 'hello',
        name: this.state.name,
        token: readPreference('md_player_token', true),
      }),
      inRoom: () => !!this.state.room,
      receive: this.onServer,
      disconnected: () => {
        this.patch({ connected: false });
        this.post({ type: 'running', value: false });
        this.input.clear();
      },
    });
  }
  getSnapshot = () => this.state;
  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };
  private patch(next: Partial<GameState>) {
    this.state = { ...this.state, ...next };
    for (const listener of this.listeners) listener();
  }
  private engine(next: Partial<Engine>) {
    this.patch({ engine: { ...this.state.engine, ...next } });
  }
  get loaded() {
    return this.state.engine.phase === 'ready';
  }
  getViewer = () => this.scene as Viewer | null;
  start(element: HTMLElement) {
    if (this.mounted) return;
    this.mounted = true;
    const name = readName();
    this.patch({
      name,

      skin: readPreference('md_skin'),
      debugOpen: new URL(location.href).searchParams.get('debug') === '1',
    });
    try {
      this.scene = this.dependencies.createViewer(
        element,
        this.state.effectLevel,
      );
      this.scene.setPlayerNames(this.state.showNames);
      this.startSimulation();
    } catch (error) {
      this.engine({
        phase: 'failed',
        error:
          'Unable to create the 3D view. Enable hardware acceleration in your browser.',
      });
      console.error(error);
    }
    if (name) this.openMode();
    this.input.start({
      command: (keys, axes) => {
        if (
          [
            'KeyW',
            'KeyS',
            'KeyA',
            'KeyD',
            'ArrowUp',
            'ArrowDown',
            'ArrowLeft',
            'ArrowRight',
          ].some((key) => keys.has(key)) ||
          axes?.forward ||
          axes?.turn
        )
          this.activity();
        this.post({
          type: 'command',
          command: this.scene?.debug.captureKeys
            ? { vx: 0, wz: 0 }
            : drivingCommand(
                keys,
                this.state.room,
                this.state.identity,
                Date.now() + this.state.clockOffset,
                axes,
              ),
        });
      },
      jump: () => this.post({ type: 'jump' }),
      rescue: () => this.send({ type: 'rescue' }),
      camera: this.toggleCamera,
      debug: () => this.setDebugOpen(!this.state.debugOpen),
    });
    this.timer = setInterval(() => this.patch({ now: Date.now() }), 100);
    this.registerIntegration();
  }
  private registerIntegration() {
    const context = (
      document as Document & {
        modelContext?: {
          registerTool: (
            tool: unknown,
            options: { signal: AbortSignal },
          ) => unknown;
        };
      }
    ).modelContext;
    if (!context?.registerTool) return;
    this.integration = new AbortController();
    try {
      Promise.resolve(
        context.registerTool(
          {
            name: 'read_race_state',
            description:
              'Read the current multiplayer race and local physics readiness.',
            inputSchema: {
              type: 'object',
              properties: {},
              additionalProperties: false,
            },
            annotations: { readOnlyHint: true },
            execute: () => ({ room: this.state.room, ready: this.loaded }),
          },
          { signal: this.integration.signal },
        ),
      ).catch(() => {});
    } catch {}
  }
  private startSimulation() {
    this.lastFrame = null;
    this.engine({ phase: 'starting', error: '' });
    try {
      this.simulation.start();
    } catch (error) {
      this.onWorker({
        type: 'fatal',
        text: `Physics engine failed to start: ${String(error)}`,
      });
    }
  }
  private openMode() {
    if (this.state.mode === 'single')
      this.local = new LocalRace(this.state.name, this.onServer);
    else this.connection.start();
  }
  setMode = (mode: GameMode) => {
    if (
      mode === this.state.mode ||
      (mode === 'multiplayer' && !this.state.multiplayerEnabled)
    )
      return;
    this.send({ type: 'leave' });
    this.connection.dispose();
    this.local?.dispose();
    this.local = null;
    this.clearRoom();
    this.leavingRoom = false;
    this.lastActivitySent = -Infinity;
    this.patch({
      mode,
      connected: false,
      identity: '',
      rooms: [],
      clockOffset: 0,
      idleWarning: '',
      connectionSleeping: false,
    });
    if (this.state.name) this.openMode();
    if (this.mounted) this.startSimulation();
  };
  startSingle = () => {
    if (this.state.mode !== 'single' || !this.loaded) return;
    this.startSoloWhenReady = true;
    this.send({ type: 'create', map: this.state.mapId });
  };
  reconnect = () => {
    if (this.state.mode !== 'multiplayer') return;
    this.patch({ connectionSleeping: false, idleWarning: '' });
    this.connection.start();
  };
  activity = () => {
    if (this.state.mode !== 'multiplayer') return;
    this.patchActivityWarning();
    if (!this.state.connected || Date.now() - this.lastActivitySent < 1000)
      return;
    this.lastActivitySent = Date.now();
    this.connection.send({ type: 'activity' });
  };
  private patchActivityWarning() {
    if (this.state.idleWarning) this.patch({ idleWarning: '' });
  }
  send = (message: ClientMessage) => {
    if (this.state.mode === 'single') this.local?.send(message);
    else this.connection.send(message);
    if (
      [
        'ready',
        'start',
        'again',
        'create',
        'join',
        'leave',
        'rename',
        'skin',
        'rescue',
      ].includes(message.type)
    )
      this.patchActivityWarning();
  };
  leaveRoom = () => {
    if (!this.state.room) return;
    this.leavingRoom = true;
    this.clearRoom();
    this.notify('');
    this.send({ type: 'leave' });
  };
  post = (message: WorkerCommand) => {
    if (message.type === 'jump') this.activity();
    this.simulation.post(message);
  };
  notify = (notice: string) => {
    clearTimeout(this.noticeTimer);
    this.patch({ notice });
    if (notice)
      this.noticeTimer = setTimeout(() => this.patch({ notice: '' }), 3500);
  };
  submitName = (value: string) => {
    const name = cleanName(value);
    if (this.state.name) {
      saveName(name);
      this.patch({ name });
      this.send({ type: 'rename', name });
    } else {
      saveName(name);
      this.patch({ name });
      if (this.mounted) this.openMode();
    }
  };
  setDebugOpen = (debugOpen: boolean) => this.patch({ debugOpen });
  setShowMinimap = (showMinimap: boolean) => this.patch({ showMinimap });
  setShowNames = (showNames: boolean) => {
    this.patch({ showNames });
    this.scene?.setPlayerNames(showNames);
  };
  setEffectLevel = (effectLevel: EffectsQuality) => {
    this.patch({ effectLevel });
    this.scene?.setEffectsQuality(effectLevel);
    savePreference('md_effects_quality', effectLevel);
  };
  toggleCamera = () => {
    const cameraOverview = !(this.scene?.preview ?? this.state.cameraOverview);
    if (this.scene) this.scene.preview = cameraOverview;
    this.patch({ cameraOverview });
  };
  changeMap = (mapId: string) => {
    if (this.state.room) return;
    this.activity();
    this.engine({ phase: 'loading-map', status: 'Switching course…' });
    this.patch({ mapId });
    this.scene?.setMap(mapId);
    this.post({ type: 'map', map: mapId });
  };
  private unready() {
    if (this.state.room?.phase === 'lobby')
      this.send({ type: 'ready', ready: false });
  }
  upload = async (file?: File) => {
    if (!file) return;
    if (this.state.room && this.state.room.phase !== 'lobby') {
      this.notify(
        'A race is in progress. Leave the room after the race to change policy.',
      );
      return;
    }
    if (!this.loaded) return;
    if (
      !file.name.toLowerCase().endsWith('.onnx') ||
      !file.size ||
      file.size > 64 * 1024 * 1024
    ) {
      this.notify('Choose an .onnx file no larger than 64 MB');
      return;
    }
    this.activity();
    this.unready();
    this.engine({ phase: 'loading-policy', status: 'Reading custom policy…' });
    await this.simulation.upload(file);
  };
  restore = () => {
    if (
      this.state.room &&
      this.state.room.phase !== 'lobby' &&
      this.state.engine.phase !== 'failed'
    )
      return;
    this.activity();
    this.unready();
    this.engine({ status: 'Restoring default policy…' });
    this.startSimulation();
  };
  private syncRunning() {
    const { room, identity, connected } = this.state;
    const me = room?.players.find((p) => p.id === identity);
    this.post({
      type: 'running',
      value:
        connected &&
        room?.phase === 'racing' &&
        me?.finished === null &&
        this.loaded,
    });
  }
  private onWorker = (event: WorkerEvent) => {
    switch (event.type) {
      case 'configuration':
        this.patch({ multiplayerEnabled: event.multiplayerEnabled });
        if (!event.multiplayerEnabled && this.state.mode === 'multiplayer')
          this.setMode('single');
        if (!this.inviteChecked) {
          this.inviteChecked = true;
          if (
            event.multiplayerEnabled &&
            new URL(location.href).searchParams.has('room')
          )
            this.setMode('multiplayer');
        }
        break;
      case 'status':
        this.engine({ status: event.text });
        break;
      case 'model':
        this.scene?.setModel(event.geoms);
        break;
      case 'policy':
        this.engine({ label: event.label, jumpSupported: event.jumpSupported });
        break;
      case 'ready': {
        const { room, identity, mapId } = this.state;
        const me = room?.players.find((p) => p.id === identity);
        const map = room?.map ?? mapId;
        this.engine({ phase: 'loading-map' });
        this.post({
          type: 'map',
          map,
          spawn: spawn(getMap(map), me?.slot ?? 0),
        });
        break;
      }
      case 'mapReady':
        if (event.map !== this.state.mapId) break;
        this.engine({ phase: 'ready', status: 'Engine ready', error: '' });
        if (this.state.room) this.syncRunning();
        if (
          this.startSoloWhenReady &&
          this.state.mode === 'single' &&
          this.state.room?.phase === 'lobby'
        ) {
          this.startSoloWhenReady = false;
          this.send({ type: 'ready', ready: true });
          this.send({ type: 'start' });
        }
        break;
      case 'error':
        this.notify(event.text);
        break;
      case 'fatal':
        this.engine({ phase: 'failed', error: event.text });
        this.unready();
        this.post({ type: 'running', value: false });
        break;
      case 'frame': {
        this.scene?.setFrame(event);
        const now = performance.now(),
          previous = this.lastFrame;
        this.lastFrame = { wall: now, sim: event.simTime, epoch: event.epoch };
        if (now - this.lastHud > 120) {
          this.patch({
            telemetry: {
              speed: event.speed,
              z: event.p[2],
              ratio:
                previous && previous.epoch === event.epoch && event.running
                  ? Math.min(
                      1,
                      (event.simTime - previous.sim) /
                        ((now - previous.wall) / 1000),
                    )
                  : 1,
            },
          });
          this.lastHud = now;
        }
        if (
          this.state.room?.phase === 'racing' &&
          now - this.lastSent > 45 &&
          this.loaded
        ) {
          this.send({ type: 'state', p: event.p, q: event.q, j: event.j });
          this.lastSent = now;
        }
        break;
      }
    }
  };
  private clearRoom() {
    this.startSoloWhenReady = false;
    this.patch({ room: null, cameraOverview: true });
    this.input.clear();
    this.post({ type: 'running', value: false });
    this.scene?.updateRoom(null, this.state.identity);
    if (this.scene) this.scene.preview = true;
  }
  private onServer = (event: ServerMessage) => {
    // The local exit takes effect immediately, even while disconnected. Ignore
    // old race snapshots until the server acknowledges it or reconnects.
    if (this.leavingRoom && ['joined', 'room', 'reset'].includes(event.type))
      return;
    switch (event.type) {
      case 'welcome': {
        if (!event.room && this.state.room) this.clearRoom();
        const preferred = this.state.skin;
        this.patch({ connected: true, identity: event.id, skin: event.skin });
        if (this.leavingRoom) {
          if (event.room) this.send({ type: 'leave' });
          else this.leavingRoom = false;
        }
        this.scene?.setSkin(event.skin);
        if (!event.room && preferred && preferred !== event.skin)
          this.send({ type: 'skin', skin: preferred });
        if (this.state.mode === 'multiplayer')
          savePreference('md_player_token', event.token, true);
        break;
      }
      case 'rooms':
        this.patch({ rooms: event.rooms });
        break;
      case 'joined': {
        const mapChanged = event.room.map !== this.state.mapId;
        this.patch({
          room: event.room,
          mapId: event.room.map,
          cameraOverview: false,
          clockOffset: event.room.now - Date.now(),
        });
        if (mapChanged) this.scene?.setMap(event.room.map);
        this.scene?.updateRoom(event.room, this.state.identity);
        this.engine({
          phase: 'loading-map',
          status: mapChanged ? 'Loading course…' : 'Preparing robot…',
        });
        this.post({ type: 'running', value: false });
        this.post({ type: 'map', map: event.room.map, spawn: event.spawn });
        if (this.scene) this.scene.preview = false;
        break;
      }
      case 'room':
        this.patch({
          room: event.room,
          clockOffset: event.room.now - Date.now(),
        });
        this.scene?.updateRoom(event.room, this.state.identity);
        this.syncRunning();
        break;
      case 'left':
        this.leavingRoom = false;
        this.clearRoom();
        this.send({ type: 'list' });
        break;
      case 'reset':
        this.input.clear();
        this.post({ type: 'reset', spawn: event.spawn });
        break;
      case 'error':
        if (event.text.includes('another connection'))
          this.send({ type: 'hello', name: this.state.name });
        else this.notify(event.text);
        break;
      case 'renamed':
        saveName(event.name);
        this.patch({ name: event.name });
        break;
      case 'skinned':
        savePreference('md_skin', event.skin);
        this.patch({ skin: event.skin });
        this.scene?.setSkin(event.skin);
        break;
      case 'idleWarning':
        this.patch({ idleWarning: 'Leaving soon due to inactivity.' });
        break;
      case 'idle':
        this.patch({ idleWarning: '' });
        this.clearRoom();
        if (event.scope === 'connection') {
          this.connection.dispose();
          this.patch({ connected: false, connectionSleeping: true, rooms: [] });
        }
        this.notify(
          event.scope === 'room'
            ? 'Left the room after 3 minutes of inactivity.'
            : 'Disconnected after 3 minutes of inactivity. Reconnect when ready.',
        );
        break;
      case 'pong':
        this.patch({ ping: Math.round(Date.now() - event.sent) });
        break;
    }
  };
  dispose() {
    this.mounted = false;
    clearInterval(this.timer);
    clearTimeout(this.noticeTimer);
    this.integration?.abort();
    this.input.dispose();
    this.connection.dispose();
    this.local?.dispose();
    this.local = null;
    this.simulation.dispose();
    this.scene?.dispose();
    this.scene = null;
  }
}
