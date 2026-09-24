import {
  clampStick,
  joystickAxes,
  type StickPosition,
  type DriveAxes,
} from './joystick.ts';
import {
  boostedForwardSpeed,
  terrainTurboFactor,
} from '../game/drive-command.mjs';
import { getMap, surfaceAt } from '../game/maps.mjs';
import type { Command, Room } from '../shared/protocol.ts';

export function drivingCommand(
  keys: ReadonlySet<string>,
  room: Room | null,
  identity: string,
  now: number,
  axes?: DriveAxes,
): Command {
  const me = room?.players.find((p) => p.id === identity);
  const boost = (me?.effects.boost ?? 0) > now;
  const forward = keys.has('KeyW') || keys.has('ArrowUp');
  const back = keys.has('KeyS') || keys.has('ArrowDown');
  const turn =
    axes?.turn ??
    (keys.has('KeyA') || keys.has('ArrowLeft') ? 1 : 0) -
      (keys.has('KeyD') || keys.has('ArrowRight') ? 1 : 0);
  const floor =
    boost && me && room ? surfaceAt(getMap(room.map), ...me.p) : null;
  return {
    vx: boostedForwardSpeed(
      (axes?.forward ?? Number(forward) - Number(back)) * 1.5,
      turn * 1.8,
      boost,
      floor && room ? terrainTurboFactor(getMap(room.map), floor.s) : 1,
    ),
    wz: turn * 1.8,
  };
}

/** Keyboard and analog touch share the same command path and focus/reset lifecycle. */
export class DriveInput {
  readonly keys = new Set<string>();
  private stick: StickPosition | null = null;
  private stickListeners = new Set<() => void>();
  private emit: (() => void) | undefined;
  getStick = () => this.stick;
  subscribeStick = (listener: () => void) => {
    this.stickListeners.add(listener);
    return () => {
      this.stickListeners.delete(listener);
    };
  };
  setStick = (x: number, y: number) => {
    this.stick = clampStick(x, y);
    for (const listener of this.stickListeners) listener();
  };
  releaseStick = () => {
    if (this.stick !== null) {
      this.stick = null;
      for (const listener of this.stickListeners) listener();
    }
    this.emit?.();
  };
  clear = () => {
    this.keys.clear();
    this.releaseStick();
  };
  private cleanup: (() => void) | undefined;
  start(actions: {
    command: (keys: ReadonlySet<string>, axes?: DriveAxes) => void;
    jump: () => void;
    rescue: () => void;
    camera: () => void;
    debug: () => void;
  }) {
    this.dispose();
    this.emit = () =>
      actions.command(
        this.keys,
        this.stick ? joystickAxes(this.stick) : undefined,
      );
    const down = (event: KeyboardEvent) => {
      if (event.code === 'F2') {
        event.preventDefault();
        actions.debug();
        return;
      }
      if (
        (event.target as HTMLElement)?.closest(
          'input,textarea,select,[contenteditable=true],.settings-menu',
        )
      )
        return;
      if (
        event.code === 'Space' &&
        (event.target as HTMLElement)?.closest('button,summary')
      )
        return;
      if (
        ['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(
          event.code,
        )
      )
        event.preventDefault();
      this.keys.add(event.code);
      if (event.repeat) return;
      if (event.code === 'Space') actions.jump();
      if (event.code === 'KeyR') actions.rescue();
      if (event.code === 'KeyC') actions.camera();
    };
    const up = (event: KeyboardEvent) => this.keys.delete(event.code);
    const blur = this.clear;
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    window.addEventListener('blur', blur);
    document.addEventListener('visibilitychange', blur);
    const timer = setInterval(this.emit, 20);
    this.cleanup = () => {
      clearInterval(timer);
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
      window.removeEventListener('blur', blur);
      document.removeEventListener('visibilitychange', blur);
      blur();
    };
  }
  press = (code: string) => {
    this.keys.add(code);
  };
  release = (code: string) => {
    this.keys.delete(code);
  };
  dispose() {
    this.cleanup?.();
    this.cleanup = undefined;
    this.clear();
    this.emit = undefined;
  }
}
