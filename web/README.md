# OpenBeni Web game

The browser runs the repository's current BENI robot model and policy using MuJoCo WASM and ONNX Runtime Web.

## Development and runtime

Build requirements: Node.js 22.13+ and Python 3.9+ available as `python3` (standard library only; no training dependencies). Build from a full repository checkout: the asset exporter reads `../assets/` and `../python/`, so `web/` alone cannot be installed or built. Commands below use POSIX shell syntax; on Windows, use an environment that provides `python3`, such as WSL.

Run from this `web/` directory:

```bash
npm ci
npm run dev
# Production: npm run build && npm start
```

Open `http://localhost:3000/`. The server listens on `0.0.0.0:3000` by default; use `HOST=127.0.0.1` to bind only to this machine, or `PORT=3003 npm run dev` to run alongside another worktree's server. `predev` / `prebuild` prepare WASM from locked npm dependencies and precompile all three fixed courses into compressed MJB files. Initial generation takes tens of seconds; results are cached in `public/runtime/tracks/` and reused while inputs remain unchanged.

## How to play

1. Enter a **Driver name** and select **Enter garage**. Choose a course and skin.
2. For a solo run, keep **Single Player** selected and choose **Start game**.
3. To race with friends, select **Multiplayer**. Choose **Create room**, join a listed open room, or enter a five-character room code. Share the room's invite link or code. Everyone selects **Ready up**; the host then selects **Start race**. Rooms hold up to eight drivers, and races last two laps.
4. Follow the course to finish both laps. Collect gems for points and occasional boosts. After the race, the host can choose **Race again**.

If the Multiplayer option is absent, this deployment offers Single Player only.

### Controls

| Input | Action |
| --- | --- |
| W/A/S/D or arrow keys | Drive and steer |
| Space | Jump |
| R | Return to the course (+3 seconds) |
| C | Switch camera |
| Touchscreen | Steer with the left joystick; use the Jump and Reset buttons on the right |

Use **Settings** to change effects quality and toggle the minimap or driver names.

## Verification

After `npm ci`, these commands work in a fresh checkout:

```bash
npm run typecheck
npm run build
npm run test:laps
```

`test:laps` runs the bundled policy with MuJoCo WASM and ONNX on all three
courses from three starting slots, including both split routes, for two laps.
It writes reports under `outputs/lap-verification/`; see
`tools/demo-recorder/README.md` for options.

## Optional asset CDN

Set `VITE_ASSET_BASE_URL` at build time to an immutable asset release directory, for example `https://huggingface.co/spaces/OWNER/SPACE/resolve/COMMIT/dist`. Robot assets, policy files, physics binaries and compiled tracks use this base; API requests and WebSocket connections remain same-origin. Leave it unset for a self-contained deployment.

Publish the asset snapshot first, then build the frontend with that snapshot's commit. Keep the ONNX runtime JavaScript module same-origin. Runtime asset downloads are limited to four concurrent transfers per context and retry the selected source up to three times. A transfer aborts only after 60 seconds without incoming data, not after a fixed total duration. Failed CDN transfers never automatically fall back to the Space; the error panel offers a CDN retry or an explicit backup-source reload. The public Space includes the complete source snapshot and assets, but never the original Git history, credentials or local diagnostics.

For HF-hosted frontend bundles, set server variable `FRONTEND_ASSET_BASE_URL` to the immutable release directory containing the exact deployed build. The tiny same-origin HTML loader fetches JS/CSS concurrently and loads typed Blob modules because HF serves JavaScript downloads as plain text. A Blob worker bootstrap queues early messages and fetches the matching worker, using the same idle timeout and selected-source retries as runtime downloads. Source changes require the player to choose the backup explicitly. Do not mix this variable with a different frontend build. Without it, normal local script and worker loading is unchanged.

Startup progress reports downloaded/required MB for the current initial course's physics assets; cached assets are excluded from the download total. Rendering-only skins load separately on demand. Successful runtime downloads are size/SHA-256 verified and stored in Cache Storage by content hash, shared between CDN and explicit backup mode. Refreshes reuse unchanged files; new skins or courses download only missing assets. Cache cleanup runs at most daily per cache and removes entries unused for 30 days; it does not remove another release merely because the current manifest changed. Frontend scripts/styles and worker source use immutable release URL keys. Cache access/quota failures are nonfatal; private browsing, cleared site data or browser eviction can require another download. Initialization still runs after a cache hit.

## Loading checks and source deployment

The canonical upstream is `https://github.com/Mondo-Robotics/open-beni.git`. Run builds and exports from this checkout. Hugging Face is the deployment target, not a separate development repository.

The shared transport lives in `loading/download.mjs`; `loading/cache.mjs` handles optional persistence. The page and worker bootstraps are maintained source modules, bundled by `scripts/build-loaders.mjs`. Course changes cancel superseded downloads and establish a fresh progress plan.

Run `npm run test:loading` for retry, idle timeout, cancellation and cache tests. After building and starting a server, run `GAME_URL=http://localhost:3000/ node tools/loading-checks/browser.mjs` for browser checks. `MOCK_CDN=1` uses local built payloads with HF-like response types; omit it when checking the real Space. Run `node tools/loading-checks/worker-race.mjs` against the local server to verify a map change interrupts an in-flight download. Chrome must be installed; `CHROME_BIN` overrides its path.

Build with the desired immutable `VITE_ASSET_BASE_URL`, then run `python3 web/scripts/export-space.py /path/to/new-export` from the repository root. This exports the complete source snapshot plus `dist/` for CDN delivery, without Git history or ignored local files. It preserves the destination Space's existing `.gitattributes`, which HF needs to resolve binary assets through LFS. The default destination is `mondo-robotics/open-beni`; use `--space OWNER/SPACE` for another existing public Space. Do not overwrite those attributes with the source repository's version. Upload that export to the Space, then set the Space runtime variable `FRONTEND_ASSET_BASE_URL` to `https://huggingface.co/spaces/OWNER/SPACE/resolve/SNAPSHOT/dist`, using the returned upload revision. Before enabling it, run `node web/scripts/check-space.mjs https://YOUR-SPACE.hf.space/` against the completed source build; it must match the local export. Preserve the runtime CDN build argument used for the exported build. The Docker image builds from source; checked-in generated `dist/` is only the immutable CDN payload.

The root Dockerfile builds with Node 24.11.1 and Python 3, then runs the game as a non-root user on `0.0.0.0:7860`. The full training code is published but is not installed or executed by the game server.
