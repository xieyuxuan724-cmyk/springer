import { SKINS } from '../game/skins.mjs';
import { MAPS, getMap, spawn, distance, heading } from '../game/maps.mjs';
import { createGems, renewGem } from './gems.mjs';
import { cleanName, validPose } from '../shared/validation.mjs';
export { cleanName } from '../shared/validation.mjs';
export class RaceRules {
  constructor({
    now = Date.now,
    random = Math.random,
    id = () => globalThis.crypto.randomUUID(),
    token = () => globalThis.crypto.randomUUID(),
    roomCode = () => globalThis.crypto.randomUUID().slice(0, 5).toUpperCase(),
    maxRooms = 10,
    idleTimeoutMs = 180000,
  } = {}) {
    this.id = id;
    this.token = token;
    this.roomCode = roomCode;
    this.maxRooms = maxRooms;
    this.idleTimeoutMs = idleTimeoutMs;
    this.now = now;
    this.random = random;
    this.players = new Map();
    this.rooms = new Map();
  }
  send(p, msg) {
    p?.send?.(msg);
  }
  list() {
    return [...this.rooms.values()].map((r) => ({
      code: r.code,
      map: r.map,
      count: r.players.length,
      phase: r.phase,
      host: this.players.get(r.host)?.name,
    }));
  }
  connect(name, token, send) {
    name = cleanName(name);
    let p = [...this.players.values()].find((p) => token && p.token === token);
    if (p?.connected)
      throw Error('This driver is already online on another connection');
    if (!p) {
      if (this.players.size >= 256) throw Error('The server is full');
      p = {
        id: this.id(),
        token: this.token(),
        name,
        skin: SKINS[Math.floor(this.random() * SKINS.length)].id,
        room: null,
      };
      this.players.set(p.id, p);
    }
    p.lastActivityAt ??= this.now();
    p.name = name;
    p.send = send;
    p.connected = true;
    p.lastSeen = this.now();
    this.send(p, {
      type: 'welcome',
      id: p.id,
      token: p.token,
      room: p.room,
      skin: p.skin,
    });
    this.send(p, { type: 'rooms', rooms: this.list() });
    if (p.room) {
      const r = this.rooms.get(p.room);
      if (r) {
        this.send(p, {
          type: 'joined',
          room: this.view(r),
          spawn: this.respawn(r, p),
        });
        this.broadcast(r);
      }
    }
    return p;
  }
  disconnect(p) {
    p.connected = false;
    p.send = null;
    p.lastSeen = this.now();
    if (p.room) this.broadcast(this.rooms.get(p.room));
  }
  view(r) {
    return {
      code: r.code,
      map: r.map,
      host: r.host,
      phase: r.phase,
      startAt: r.startAt ?? 0,
      laps: r.laps,
      now: this.now(),
      events: r.events.slice(-5),
      pickups: r.pickups,
      players: r.players.map((id) => {
        const p = this.players.get(id);
        return {
          id: p.id,
          name: p.name,
          skin: p.skin,
          connected: p.connected,
          ready: !!p.ready,
          slot: p.slot,
          p: p.p,
          q: p.q,
          j: p.j,
          speed: p.speed ?? 0,
          lap: p.lap ?? 0,
          next: p.next ?? 1,
          finished: p.finished ?? null,
          penalty: p.penalty ?? 0,
          gems: p.gems ?? 0,
          lastPickup: p.lastPickup ?? null,
          effects: p.effects ?? {},
        };
      }),
    };
  }
  broadcast(r) {
    if (!r) return;
    const msg = { type: 'room', room: this.view(r) };
    for (const id of r.players) this.send(this.players.get(id), msg);
  }
  event(r, text) {
    r.events.push({ id: this.id(), text, at: this.now() });
    r.events = r.events.slice(-12);
  }
  respawn(r, p) {
    return spawn(getMap(r.map), p.slot, p.lastCheckpoint ?? 0);
  }
  position(p, at) {
    p.p = [...at.p];
    p.q = [Math.cos(at.yaw / 2), 0, 0, Math.sin(at.yaw / 2)];
    p.j = [0.3, 0.3, 0.5, 0.5, 0, 0, 0];
    p.lastState = this.now();
    p.stateCount = 0;
  }
  resetPlayer(r, p) {
    p.ready = false;
    p.lap = 0;
    p.next = 1;
    p.lastCheckpoint = 0;
    p.finished = null;
    p.penalty = 0;
    p.gems = 0;
    p.lastPickup = null;
    p.effects = {};
    p.lastRescue = 0;
    this.position(p, this.respawn(r, p));
  }
  join(p, r) {
    if (p.room) throw Error('Leave your current room first');
    if (r.phase !== 'lobby')
      throw Error('The race has started. Wait for the next race.');
    if (r.players.length >= 8) throw Error('The room is full');
    this.activity(p);
    p.room = r.code;
    p.slot = [0, 1, 2, 3, 4, 5, 6, 7].find(
      (n) => !r.players.some((id) => this.players.get(id).slot === n),
    );
    const usedSkins = new Set(r.players.map((id) => this.players.get(id).skin));
    if (usedSkins.has(p.skin)) {
      const available = SKINS.filter((s) => !usedSkins.has(s.id));
      p.skin = available[Math.floor(this.random() * available.length)].id;
    }
    r.players.push(p.id);
    this.resetPlayer(r, p);
    this.send(p, {
      type: 'joined',
      room: this.view(r),
      spawn: this.respawn(r, p),
    });
    this.event(r, `${p.name} joined the race`);
    this.broadcast(r);
  }
  leave(p) {
    const r = this.rooms.get(p.room);
    p.room = null;
    if (!r) return;
    r.players = r.players.filter((id) => id !== p.id);
    if (!r.players.length) this.rooms.delete(r.code);
    else {
      if (r.host === p.id)
        r.host =
          r.players.find((id) => this.players.get(id).connected) ??
          r.players[0];
      this.broadcast(r);
    }
    this.send(p, { type: 'left' });
  }
  activity(p) {
    p.lastActivityAt = this.now();
    p.idleWarned = false;
  }
  handle(p, msg) {
    if (!this.players.has(p.id)) return;
    this.handleMessage(p, msg);
    if (
      [
        'activity',
        'rename',
        'skin',
        'create',
        'join',
        'leave',
        'ready',
        'start',
        'again',
        'rescue',
      ].includes(msg.type)
    )
      this.activity(p);
  }
  handleMessage(p, msg) {
    const r = this.rooms.get(p.room),
      now = this.now();
    switch (msg.type) {
      case 'activity':
        return;
      case 'ping':
        this.send(p, { type: 'pong', sent: msg.sent, now });
        return;
      case 'list':
        this.send(p, { type: 'rooms', rooms: this.list() });
        return;
      case 'rename':
        p.name = cleanName(msg.name);
        if (r) this.broadcast(r);
        this.send(p, { type: 'renamed', name: p.name });
        return;
      case 'create': {
        if (p.room) throw Error('Leave your current room first');
        if (this.rooms.size >= this.maxRooms)
          throw Error('The room limit has been reached');
        if (!MAPS.some((m) => m.id === msg.map)) throw Error('Invalid course');
        let code;
        do {
          code = this.roomCode();
        } while (this.rooms.has(code));
        const room = {
          code,
          map: msg.map,
          host: p.id,
          players: [],
          phase: 'lobby',
          laps: 2,
          events: [],
          pickups: createGems(getMap(msg.map), now, this.random),
        };
        this.rooms.set(code, room);
        this.join(p, room);
        return;
      }
      case 'join': {
        const target = this.rooms.get(String(msg.code).trim().toUpperCase());
        if (!target) throw Error('Room not found');
        this.join(p, target);
        return;
      }
      case 'skin': {
        const skin = SKINS.find((s) => s.id === msg.skin);
        if (!skin) throw Error('Unknown skin');
        if (r) {
          if (r.phase !== 'lobby')
            throw Error('Cannot change skin during a race');
          // Eight skins for eight slots: keep the room collision-free.
          if (
            r.players.some(
              (id) => id !== p.id && this.players.get(id).skin === skin.id,
            )
          )
            throw Error(
              'This skin is already taken by another driver in the room',
            );
        }
        p.skin = skin.id;
        if (r) this.broadcast(r);
        this.send(p, { type: 'skinned', skin: p.skin });
        return;
      }
      case 'leave':
        this.leave(p);
        return;
    }
    if (!r) throw Error('Join a room first');
    switch (msg.type) {
      case 'ready':
        if (r.phase !== 'lobby')
          throw Error('Cannot change readiness during a race');
        p.ready = !!msg.ready;
        this.broadcast(r);
        break;
      case 'start': {
        if (p.id !== r.host) throw Error('Only the host can start the race');
        if (r.phase !== 'lobby') throw Error('The race has already started');
        if (
          !r.players.every((id) => {
            const x = this.players.get(id);
            return x.ready && x.connected;
          })
        )
          throw Error('Wait until every driver is ready');
        r.phase = 'countdown';
        r.startAt = now + 4000;
        r.pickups = createGems(getMap(r.map), r.startAt - 1500, this.random);
        for (const id of r.players) {
          const x = this.players.get(id);
          this.resetPlayer(r, x);
          x.ready = true;
          this.send(x, { type: 'reset', spawn: this.respawn(r, x) });
        }
        this.event(r, 'Starting countdown');
        this.broadcast(r);
        break;
      }
      case 'again':
        if (p.id !== r.host || r.phase !== 'finished')
          throw Error('The host can start another round after the race ends');
        r.phase = 'lobby';
        r.startAt = 0;
        for (const id of r.players) {
          const x = this.players.get(id);
          this.resetPlayer(r, x);
          this.send(x, { type: 'reset', spawn: this.respawn(r, x) });
        }
        this.broadcast(r);
        break;
      case 'rescue': {
        if (r.phase !== 'racing') return;
        if (now - (p.lastRescue ?? 0) < 3000) return;
        p.lastRescue = now;
        p.penalty += 3000;
        const at = this.respawn(r, p);
        this.position(p, at);
        this.send(p, { type: 'reset', spawn: at });
        this.event(r, `${p.name} rescued +3 s`);
        this.broadcast(r);
        break;
      }
      case 'state':
        this.updateState(r, p, msg, now);
        break;
      case 'use':
        // Ignore the removed manual pickup action from existing clients.
        break;
      default:
        throw Error('Unknown action');
    }
  }
  updateState(r, p, msg, now) {
    if (r.phase !== 'racing' || p.finished !== null) return;
    if (!validPose(msg)) return;
    const dt = Math.min(0.5, Math.max(0, (now - p.lastState) / 1000));
    if (dt < 0.02) return;
    // Sanity checks, not a cheat-proof authority: arbitrary user policies execute locally.
    if (distance(msg.p, p.p) > 12 * dt + 0.12) {
      this.send(p, { type: 'reset', spawn: this.respawn(r, p) });
      this.position(p, this.respawn(r, p));
      return;
    }
    p.speed = distance(msg.p, p.p) / dt;
    const previousPosition = p.p;
    p.lastState = now;
    p.p = msg.p;
    p.q = msg.q;
    p.j = msg.j;
    p.stateCount++;
    const map = getMap(r.map),
      target = map.points[p.next];
    let reached =
      distance(p.p, target) < map.width * 0.65 &&
      p.p[2] > target[2] - 0.1 &&
      p.p[2] < target[2] + 0.8;
    if (reached && p.next === 0) {
      const yaw = heading(map, 0),
        forwardX = Math.cos(yaw),
        forwardY = Math.sin(yaw),
        before =
          (previousPosition[0] - target[0]) * forwardX +
          (previousPosition[1] - target[1]) * forwardY,
        after =
          (p.p[0] - target[0]) * forwardX +
          (p.p[1] - target[1]) * forwardY;
      reached = before < 0 && after >= 0;
    }
    if (reached) {
      p.lastCheckpoint = p.next;
      if (p.next === 0) {
        p.lap++;
        if (p.lap >= r.laps) {
          p.finished = now - r.startAt + p.penalty;
          this.event(
            r,
            `${p.name} finished in ${(p.finished / 1000).toFixed(2)}s`,
          );
        }
      }
      p.next = (p.next + 1) % map.points.length;
    }
    if (p.finished === null) {
      {
        const box = r.pickups.find(
          (x) =>
            x.active &&
            x.availableAt <= now &&
            x.expiresAt > now &&
            distance(x.p, p.p) < 0.42 &&
            Math.abs(x.p[2] - p.p[2]) < 0.5,
        );
        if (box) {
          box.active = false;
          box.availableAt = now + 1800 + this.random() * 2600;
          p.gems += box.value;
          const bonusBoost = this.random() < 0.25;
          if (bonusBoost)
            p.effects.boost = Math.max(p.effects.boost ?? 0, now + 1800);
          p.lastPickup = {
            seq: (p.lastPickup?.seq ?? 0) + 1,
            p: [...box.p],
            value: box.value,
            color: box.color,
            at: now,
            boost: bonusBoost,
            streak:
              p.lastPickup && now - p.lastPickup.at < 4000
                ? p.lastPickup.streak + 1
                : 1,
          };
        }
      }
    }
  }
  tick() {
    const now = this.now();
    for (const p of this.players.values()) {
      if (this.idleTimeoutMs && now - p.lastActivityAt >= this.idleTimeoutMs) {
        if (p.room) {
          this.leave(p);
          this.activity(p); // A fresh idle interval in the lobby, not automatic re-entry.
          this.send(p, { type: 'idle', scope: 'room' });
        } else {
          this.send(p, { type: 'idle', scope: 'connection' });
          p.close?.();
          this.players.delete(p.id);
        }
        continue;
      }
      if (
        this.idleTimeoutMs &&
        p.connected &&
        !p.idleWarned &&
        now - p.lastActivityAt >= this.idleTimeoutMs - 30000
      ) {
        p.idleWarned = true;
        this.send(p, { type: 'idleWarning', seconds: 30 });
      }
      if (!p.connected && now - p.lastSeen >= 20000) {
        this.leave(p);
        this.players.delete(p.id);
      }
    }
    for (const r of this.rooms.values()) {
      const previousPhase = r.phase;
      if (r.phase === 'countdown' && now >= r.startAt) {
        r.phase = 'racing';
        for (const id of r.players) this.players.get(id).lastState = now;
        this.event(r, 'GO!');
      }
      if (
        r.phase === 'racing' &&
        (now - r.startAt > 180000 ||
          r.players.every((id) => this.players.get(id).finished !== null))
      ) {
        r.phase = 'finished';
        this.event(r, 'Race over');
      }
      if (r.phase === 'racing')
        for (const gem of r.pickups)
          if (
            (!gem.active && gem.availableAt <= now) ||
            (gem.active && gem.expiresAt <= now)
          )
            renewGem(
              getMap(r.map),
              gem,
              r.pickups.length,
              now,
              this.random,
              r.players.map((id) => this.players.get(id)),
              r.pickups,
            );
      if (r.phase === 'racing' || r.phase !== previousPhase) this.broadcast(r);
    }
  }
}
