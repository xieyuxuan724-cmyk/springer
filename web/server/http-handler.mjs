import { frontendPage } from './frontend-page.mjs';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { createReadStream } from 'node:fs';
import { pipeline } from 'node:stream/promises';
export function createHttpHandler({ root, game, getVite = () => null }) {
  const mime = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript',
    '.mjs': 'text/javascript',
    '.css': 'text/css',
    '.wasm': 'application/wasm',
    '.json': 'application/json',
    '.onnx': 'application/octet-stream',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.stl': 'application/octet-stream',
    '.ico': 'image/x-icon',
  };
  return async (req, res) => {
    // Single-threaded WASM works over LAN HTTP without SharedArrayBuffer or HTTPS.
    res.setHeader('X-Content-Type-Options', 'nosniff');
    if (req.url === '/api/config' || req.url === '/game-config.json') {
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Cache-Control', 'no-store');
      res.end(
        JSON.stringify({
          multiplayerEnabled: game.maxRooms > 0,
        }),
      );
      return;
    }
    if (req.url === '/api/health') {
      res.setHeader('Content-Type', 'application/json');
      res.end(
        JSON.stringify({
          ok: true,
          rooms: game.rooms.size,
          players: [...game.players.values()].filter((p) => p.connected).length,
        }),
      );
      return;
    }
    const vite = getVite();
    if (vite) {
      vite.middlewares(req, res);
      return;
    }
    try {
      if (!['GET', 'HEAD'].includes(req.method)) {
        res.writeHead(405);
        res.end();
        return;
      }
      const pathname = decodeURIComponent(
        new URL(req.url, 'http://local').pathname,
      );
      let file = path.resolve(root, 'dist', '.' + pathname);
      const base = path.join(root, 'dist');
      if (file !== base && !file.startsWith(base + path.sep)) {
        res.writeHead(403);
        res.end();
        return;
      }
      if (pathname === '/' || !path.extname(pathname))
        file = path.join(base, 'index.html');
      if (file === path.join(base, 'index.html') && process.env.FRONTEND_ASSET_BASE_URL && new URL(req.url, 'http://local').searchParams.get('assetSource') !== 'local') {
        const html = frontendPage(await readFile(file, 'utf8'), process.env.FRONTEND_ASSET_BASE_URL);
        res.setHeader('Content-Type', 'text/html; charset=utf-8');
        res.setHeader('Cache-Control', 'no-cache');
        res.end(req.method === 'HEAD' ? undefined : html);
        return;
      }
      const s = await stat(file);
      if (!s.isFile()) throw Error('Not a file');
      res.setHeader(
        'Content-Type',
        mime[path.extname(file).toLowerCase()] || 'application/octet-stream',
      );
      res.setHeader(
        'Cache-Control',
        pathname.startsWith('/assets/')
          ? 'public, max-age=31536000, immutable'
          : 'no-cache',
      );
      const etag = `W/"${s.size}-${Math.trunc(s.mtimeMs)}"`;
      res.setHeader('ETag', etag);
      if (req.headers['if-none-match'] === etag) {
        res.writeHead(304);
        res.end();
        return;
      }
      res.setHeader('Content-Length', s.size);
      if (req.method === 'HEAD') res.end();
      else await pipeline(createReadStream(file), res);
    } catch {
      if (res.headersSent || res.destroyed) return;
      res.writeHead(404);
      res.end('Not found');
    }
  };
}
