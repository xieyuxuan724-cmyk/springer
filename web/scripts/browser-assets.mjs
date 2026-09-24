import { readFileSync, createReadStream } from 'node:fs';
import { stat, mkdir, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
const root = fileURLToPath(new URL('../', import.meta.url));
export function browserAssets() {
  const manifest = JSON.parse(
    readFileSync(path.join(root, '.generated/robot/manifest.json')),
  );
  return new Map([
    ...manifest.map((f) => [
      'robot/' + f,
      path.resolve(root, '../assets/beni_base', f),
    ]),
    ...['beni.xml', 'manifest.json'].map((f) => [
      'robot/' + f,
      path.join(root, '.generated/robot', f),
    ]),
    ['policies/beni.onnx', path.resolve(root, '../assets/models/policy.onnx')],
  ]);
}
export function assetPath(name) {
  const result = browserAssets().get(name.replace(/^\//, ''));
  if (!result) throw Error(`Unknown browser asset: ${name}`);
  return result;
}
export function browserAssetsPlugin() {
  let output;
  return {
    name: 'canonical-browser-assets',
    configResolved(config) {
      output = path.resolve(config.root, config.build.outDir);
    },
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        try {
          const name = decodeURIComponent(
            new URL(req.url, 'http://local').pathname,
          ).slice(1);
          const file = browserAssets().get(name);
          if (!file) {
            if (name.startsWith('robot/') || name.startsWith('policies/')) {
              res.statusCode = 404;
              res.end('Not found');
            } else next();
            return;
          }
          if (!['GET', 'HEAD'].includes(req.method)) {
            res.statusCode = 405;
            res.end();
            return;
          }
          const info = await stat(file);
          const types = {
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.svg': 'image/svg+xml',
          };
          res.setHeader(
            'Content-Type',
            types[path.extname(file)] || 'application/octet-stream',
          );
          res.setHeader('Cache-Control', 'no-cache');
          res.setHeader('Content-Length', info.size);
          if (req.method === 'HEAD') res.end();
          else await pipeline(createReadStream(file), res);
        } catch (error) {
          if (res.destroyed || res.headersSent) return;
          next(error);
        }
      });
    },
    async writeBundle() {
      for (const [name, source] of browserAssets()) {
        const target = path.join(output, name);
        await mkdir(path.dirname(target), { recursive: true });
        await copyFile(source, target);
      }
    },
  };
}
