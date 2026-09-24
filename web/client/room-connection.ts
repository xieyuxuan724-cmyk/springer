import type { ClientMessage, ServerMessage } from '../shared/protocol.ts';
export type SocketPort = Pick<
  WebSocket,
  | 'readyState'
  | 'send'
  | 'close'
  | 'onopen'
  | 'onmessage'
  | 'onclose'
  | 'onerror'
>;
/** Transport only: reconnect, handshake and polling. Room state belongs to GameSession. */
export class RoomConnection {
  private socket: SocketPort | null = null;
  private generation = 0;
  private retry: ReturnType<typeof setTimeout> | undefined;
  private refresh: ReturnType<typeof setInterval> | undefined;
  private options;
  constructor(options: {
    createSocket: () => SocketPort;
    hello: () => Extract<ClientMessage, { type: 'hello' }>;
    inRoom: () => boolean;
    receive: (message: ServerMessage) => void;
    disconnected: () => void;
  }) {
    this.options = options;
  }
  start() {
    this.dispose();
    const generation = this.generation;
    let outageAt: number | null = null;
    let retries = 0;
    const connect = () => {
      const ws = this.options.createSocket();
      this.socket = ws;
      const current = () =>
        generation === this.generation && this.socket === ws;
      ws.onopen = () => {
        if (current()) this.send(this.options.hello());
      };
      ws.onmessage = ({ data }) => {
        if (!current()) return;
        try {
          const event = JSON.parse(String(data)) as ServerMessage;
          if (event.type === 'welcome') {
            outageAt = null;
            retries = 0;
          }
          this.options.receive(event);
        } catch {
          this.options.receive({
            type: 'error',
            text: 'Unable to read the server message. Please reconnect.',
          });
        }
      };
      ws.onclose = (event) => {
        if (!current()) return;
        this.socket = null;
        this.options.disconnected();
        outageAt ??= Date.now();
        if (event?.code === 4000 || Date.now() - outageAt >= 180000) {
          this.dispose();
          this.options.receive({ type: 'idle', scope: 'connection' });
          return;
        }
        this.retry = setTimeout(
          connect,
          Math.min(30000, 1800 * 2 ** retries++),
        );
      };
      ws.onerror = () => {};
    };
    connect();
    this.refresh = setInterval(() => {
      this.send({ type: 'ping', sent: Date.now() });
      if (!this.options.inRoom()) this.send({ type: 'list' });
    }, 2500);
  }
  send = (message: ClientMessage) => {
    if (this.socket?.readyState === 1)
      this.socket.send(JSON.stringify(message));
  };
  dispose() {
    ++this.generation;
    clearTimeout(this.retry);
    clearInterval(this.refresh);
    if (this.socket) {
      this.socket.onopen =
        this.socket.onmessage =
        this.socket.onclose =
        this.socket.onerror =
          null;
      this.socket.close();
      this.socket = null;
    }
  }
}
