import { RaceRules } from '../shared/race-rules.mjs';
import type { ClientMessage, ServerMessage } from '../shared/protocol.ts';
/** Browser-only adapter. The same rules execute locally without network or idle eviction. */
export class LocalRace {
  private game: RaceRules;
  private player;
  private timer: ReturnType<typeof setInterval>;
  private disposed = false;
  constructor(name: string, receive: (event: ServerMessage) => void) {
    let serial = 0;
    this.game = new RaceRules({
      id: () => `local-${++serial}`,
      token: () => 'local',
      roomCode: () => 'SOLO',
      maxRooms: 1,
      idleTimeoutMs: 0,
    });
    this.player = this.game.connect(name, null, (event: ServerMessage) => {
      // Match the asynchronous network boundary and detach mutable rule state.
      const snapshot = structuredClone(event);
      queueMicrotask(() => {
        if (!this.disposed) receive(snapshot);
      });
    });
    this.timer = setInterval(() => {
      if (this.player.room) this.game.tick();
    }, 1000 / 15);
  }
  send(message: ClientMessage) {
    try {
      this.game.handle(this.player, message);
    } catch (error) {
      this.game.send(this.player, {
        type: 'error',
        text: String((error as Error).message),
      });
    }
  }
  dispose() {
    this.disposed = true;
    clearInterval(this.timer);
    this.game.rooms.clear();
    this.game.players.clear();
  }
}
