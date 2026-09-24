/** Contracts at the network / worker boundaries. No React, DOM or runtime imports. */
export type Spawn = { p: number[]; yaw: number };
export type Command = { vx: number; wz: number };
export type RoomPhase = 'lobby' | 'countdown' | 'racing' | 'finished';
export type Pickup = {
  seq: number;
  p: number[];
  value: number;
  color: string;
  at: number;
  boost: boolean;
  streak: number;
};
export type Racer = {
  id: string;
  name: string;
  skin: string;
  connected: boolean;
  ready: boolean;
  slot: number;
  p: number[];
  q: number[];
  j: number[];
  speed: number;
  lap: number;
  next: number;
  finished: number | null;
  penalty: number;
  gems: number;
  lastPickup: Pickup | null;
  effects: { boost?: number };
};
export type Gem = {
  id: string;
  slot: number;
  generation: number;
  active: boolean;
  p?: number[];
  availableAt?: number;
  expiresAt?: number;
  value?: number;
  color?: string;
};
export type Room = {
  code: string;
  map: string;
  host: string;
  phase: RoomPhase;
  startAt: number;
  laps: number;
  now: number;
  events: { id: string; text: string; at: number }[];
  pickups: Gem[];
  players: Racer[];
};
export type RoomListing = Pick<Room, 'code' | 'map' | 'phase'> & {
  count: number;
  host?: string;
};
export type ClientMessage =
  | { type: 'hello'; name: string; token?: string | null }
  | { type: 'ping'; sent: number }
  | {
      type:
        | 'activity'
        | 'list'
        | 'leave'
        | 'start'
        | 'again'
        | 'rescue'
        | 'use';
    }
  | { type: 'rename'; name: string }
  | { type: 'create'; map: string }
  | { type: 'join'; code: string }
  | { type: 'skin'; skin: string }
  | { type: 'ready'; ready: boolean }
  | { type: 'state'; p: number[]; q: number[]; j: number[] };
export type ServerMessage =
  | {
      type: 'welcome';
      id: string;
      token: string;
      room: string | null;
      skin: string;
    }
  | { type: 'rooms'; rooms: RoomListing[] }
  | { type: 'joined'; room: Room; spawn: Spawn }
  | { type: 'room'; room: Room }
  | { type: 'left' }
  | { type: 'idle'; scope: 'room' | 'connection' }
  | { type: 'idleWarning'; seconds: number }
  | { type: 'reset'; spawn: Spawn }
  | { type: 'error'; text: string }
  | { type: 'renamed'; name: string }
  | { type: 'skinned'; skin: string }
  | { type: 'pong'; sent: number; now: number };
export type MapTuning = Record<string, unknown>;
export type WorkerCommand =
  | { type: 'command'; command: Command }
  | { type: 'running'; value: boolean }
  | { type: 'step'; count?: number }
  | { type: 'stop' | 'jump' | 'default' }
  | { type: 'map'; map: string; spawn?: Spawn; tuning?: MapTuning }
  | { type: 'reset'; spawn: Spawn }
  | { type: 'policy'; bytes: ArrayBuffer; label: string };
export type RobotGeometry = {
  id: number;
  body: number;
  localPos: number[];
  localQuat: number[];
  chain: {
    pos: number[];
    quat: number[];
    joints: { index: number; axis: number[]; pos: number[]; ref: number }[];
  }[];
  vertices: Float32Array;
  normals: Float32Array;
  uv: Float32Array;
  textured: boolean;
  faces: Uint32Array;
  color: number[];
};
export type PhysicsFrame = {
  p: number[];
  q: number[];
  j: number[];
  speed: number;
  simTime: number;
  jump: { ready: boolean; tracking: boolean };
  geomIds: number[] | null;
  geomPos: Float32Array;
  geomMat: Float32Array;
  running: boolean;
  epoch: number;
  wallTime: number;
};
export type WorkerEvent =
  | { type: 'configuration'; multiplayerEnabled: boolean }
  | { type: 'status' | 'error' | 'fatal'; text: string }
  | { type: 'model'; geoms: RobotGeometry[] }
  | {
      type: 'policy';
      label: string;
      jumpSupported: boolean;
      inputs: number;
      actions: number;
      recurrent: boolean;
    }
  | { type: 'ready' }
  | { type: 'mapReady'; map: string; source?: 'reuse' | 'binary' | 'compiled' }
  | { type: 'jumpResult'; accepted: boolean }
  | ({ type: 'frame' } & PhysicsFrame);
