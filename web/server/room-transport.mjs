import { WebSocketServer } from 'ws';
import { parseClientMessage } from '../shared/validation.mjs';
/** Bind a RaceService to a server; return a deterministic cleanup for all timers/sockets. */
export function attachRoomTransport(
  server,
  game,
  { dev = false, maxConnections = 256 } = {},
) {
  const wss = new WebSocketServer({ noServer: true, maxPayload: 8192 });
  const upgrade = (req, socket, head) => {
    if (req.url !== '/ws') {
      if (!dev) socket.destroy();
      return;
    }
    if (game.maxRooms === 0) {
      socket.end('HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n');
      return;
    }
    if (wss.clients.size >= maxConnections) {
      socket.end(
        'HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n',
      );
      return;
    }
    if (req.headers.origin) {
      try {
        if (new URL(req.headers.origin).host !== req.headers.host) {
          socket.destroy();
          return;
        }
      } catch {
        socket.destroy();
        return;
      }
    }
    wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws));
  };
  server.on('upgrade', upgrade);
  wss.on('connection', (ws) => {
    let player = null,
      windowAt = Date.now(),
      count = 0;
    ws.isAlive = true;
    let blockedAt = null;
    const send = (msg) => {
      if (ws.readyState === 1 && ws.bufferedAmount < 1024 * 1024)
        ws.send(JSON.stringify(msg));
    };
    const checkBackpressure = () => {
      if (ws.bufferedAmount < 1024 * 1024) blockedAt = null;
      else if (blockedAt === null) blockedAt = Date.now();
      else if (Date.now() - blockedAt >= 30000) ws.terminate();
    };
    ws.checkBackpressure = checkBackpressure;
    const helloTimeout = setTimeout(() => {
      if (!player) ws.close(1008, 'Name required');
    }, 10000);
    ws.on('pong', () => (ws.isAlive = true));
    ws.on('message', (bytes) => {
      try {
        if (Date.now() - windowAt > 1000) {
          windowAt = Date.now();
          count = 0;
        }
        if (++count > 100) {
          ws.close(1008, 'Rate limit');
          return;
        }
        const msg = parseClientMessage(
          (Array.isArray(bytes)
            ? Buffer.concat(bytes)
            : Buffer.from(bytes)
          ).toString('utf8'),
        );
        if (!player) {
          if (msg.type !== 'hello') throw Error('Enter a username first');
          player = game.connect(msg.name, msg.token, send);
          player.close = () => ws.close(4000, 'Inactive');
          clearTimeout(helloTimeout);
        } else game.handle(player, msg);
      } catch (e) {
        send({ type: 'error', text: e.message });
      }
    });
    ws.on('close', () => {
      clearTimeout(helloTimeout);
      if (player) game.disconnect(player);
    });
    ws.on('error', () => {});
  });
  const timer =
    game.maxRooms === 0 ? undefined : setInterval(() => game.tick(), 1000 / 15);
  const heartbeat =
    game.maxRooms === 0
      ? undefined
      : setInterval(() => {
          for (const ws of wss.clients) {
            ws.checkBackpressure?.();
            if (!ws.isAlive) {
              ws.terminate();
              continue;
            }
            ws.isAlive = false;
            ws.ping();
          }
        }, 15000);

  return () => {
    clearInterval(timer);
    clearInterval(heartbeat);
    server.off('upgrade', upgrade);
    for (const ws of wss.clients) ws.terminate();
    return new Promise((resolve) => wss.close(resolve));
  };
}
