import path from 'node:path';
import { MAPS } from '../../game/maps.mjs';
import { parseArgs } from 'node:util';
export const HELP = `OpenBeni demo-recorder — real MuJoCo + ONNX recording

Usage: npm run demo:record -- [options]
       node /path/to/web/tools/demo-recorder/cli.mjs [options]

  --map ID          harbor, cloud or switchyard (default: harbor)
  --tour            Follow the full course; split routes alternate, marked jumps and automatic pickup boosts
  --players N       Independent physics players, 1–6 (default: 3)
  --seconds N       Simulation duration, 5–180 seconds (default: 45)
  --fps N           Output frame rate, 30 or 60 (default: 60)
  --output DIR      Output directory (absolute paths supported)
  --port N          Dedicated localhost server port (default: choose a free port)
  --render-only     Reuse the output directory's replay.json and verification.json
  --help            Show this help

Requires npm dependencies, Chrome, ffmpeg and ffprobe. CHROME_BIN overrides Chrome.
The tool starts and stops its own server; the normal dev server is left running.
`;
export function parseConfig(args, webRoot) {
  const { values } = parseArgs({
    args,
    options: {
      map: { type: 'string', default: 'harbor' },
      tour: { type: 'boolean' },
      players: { type: 'string', default: '3' },
      seconds: { type: 'string', default: '45' },
      fps: { type: 'string', default: '60' },
      output: { type: 'string', default: 'outputs/physics-demo' },
      port: { type: 'string', default: '0' },
      'render-only': { type: 'boolean' },
      help: { type: 'boolean' },
    },
  });
  if (values.help) return { help: true };
  if (!MAPS.some((m) => m.id === values.map))
    throw Error('--map must name an available course');
  const integer = (name, min, max) => {
    const n = Number(values[name]);
    if (!Number.isInteger(n) || n < min || n > max)
      throw Error(`--${name} must be an integer from ${min} to ${max}`);
    return n;
  };
  const players = integer('players', 1, 6),
    seconds = integer('seconds', 5, 180),
    fps = integer('fps', 30, 60),
    port = integer('port', 0, 65535);
  if (![30, 60].includes(fps)) throw Error('--fps must be 30 or 60');
  const output = path.resolve(webRoot, values.output);
  return {
    map: values.map,
    tour: !!values.tour,
    players,
    seconds,
    fps,
    port,
    output,
    renderOnly: !!values['render-only'],
  };
}
