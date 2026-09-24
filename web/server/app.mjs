import http from 'node:http';
import { serverLimits } from './config.mjs';
import { fileURLToPath } from 'node:url';
import { RaceService } from './race.mjs';
import { createHttpHandler } from './http-handler.mjs';
import { attachRoomTransport } from './room-transport.mjs';
/** Importing the server never binds a port. The caller owns listen/close. */
export async function createGameServer({
  root = fileURLToPath(new URL('../', import.meta.url)),
  dev = false,
  freezeSource = process.env.DEMO_FREEZE_SOURCE === '1',
  limits = serverLimits(),
  game = new RaceService({ maxRooms: limits.maxRooms }),
} = {}) {
  let vite;
  const server = http.createServer(
    createHttpHandler({
      root,
      game,
      getVite: () => vite,
    }),
  );
  const detach = attachRoomTransport(server, game, {
    dev,
    maxConnections: limits.maxConnections,
  });
  try {
    if (dev) {
      const { createServer } = await import('vite');
      vite = await createServer({
        root,
        server: {
          middlewareMode: true,
          ...(freezeSource
            ? { watch: { ignored: () => true }, hmr: false }
            : { hmr: { server } }),
        },
        appType: 'spa',
      });
    }
  } catch (error) {
    await detach();
    throw error;
  }
  let closing;
  return {
    server,
    game,
    listen(port = 3000, host = '0.0.0.0') {
      return new Promise((resolve, reject) => {
        const error = (err) => {
          server.off('listening', listening);
          reject(err);
        };
        const listening = () => {
          server.off('error', error);
          resolve(server.address());
        };
        server.once('error', error);
        server.once('listening', listening);
        server.listen(port, host);
      });
    },
    close() {
      return (closing ??= (async () => {
        await detach();
        await vite?.close();
        await new Promise((resolve) => {
          server.close(resolve);
          server.closeAllConnections();
        });
      })());
    },
  };
}
