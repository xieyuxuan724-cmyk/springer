# Demo Recorder

OpenBeni's real-physics multiplayer demo recorder defaults to **3 players, 45 seconds, 720p / 60 fps**, producing an MP4 without audio. Left, center and right driving lines reduce overlap on courses without forks. On split courses, drivers alternate left/right branches in both normal recordings and tour mode, following the actual branch paths instead of fixed centerline offsets.

## One command

Install project dependencies with `npm ci` in `web/`. The machine needs Chrome, `ffmpeg` and `ffprobe`. Chrome defaults to `/usr/bin/google-chrome`; override it with `CHROME_BIN`. From the repository root:

```bash
cd web
npm run demo:record
```

You can also invoke the entry point from any directory:

```bash
node /path/to/your/clone/web/tools/demo-recorder/cli.mjs
```

The tool runs physics, starts a separate localhost development server, exports and verifies the video, and closes its own server on completion or Ctrl+C. It leaves the normal game server running.

```bash
npm run demo:record -- --players 3 --seconds 30 --fps 60
# Three-player course tours, including both split routes
npm run demo:record -- --map cloud --tour --seconds 65 --output outputs/cloud-tour
npm run demo:record -- --map switchyard --tour --seconds 45 --output outputs/switchyard-dual-tour
npm run demo:record -- --seconds 8 --fps 30 --output outputs/demo-smoke
npm run demo:record -- --render-only --fps 30
npm run demo:record -- --help
```

`--map` accepts `harbor`, `cloud` or `switchyard`, defaulting to `harbor`. `--tour` enables a full-course tour. Drivers alternate left/right routes by grid slot, use slightly different speeds, send real jump requests before marked trenches, and respond to random pickup boosts. Each driver must complete a full circuit without resets or body contact before video export proceeds.

`--players` accepts 1–6 (default 3), `--seconds` accepts 5–180 (default 45), and `--fps` accepts 30 or 60 (default 60). In tour mode, `--seconds` is an upper limit. Once everyone travels the full circuit distance, capture retains a two-second ending and stops automatically. The header reads “Everyone completed a lap”; a one-lap tour is not presented as a completed two-lap race.

`--output` accepts absolute paths outside the repository, defaulting to `web/outputs/physics-demo`. Prefer an external directory for recordings and diagnostics. Reusing an output directory overwrites its corresponding capture data and video; choose different directories to keep multiple versions. `--port` selects the isolated server port, or an available port is chosen automatically. An occupied explicit port causes an error without terminating the existing service. `--render-only` reuses `replay.json` and `verification.json` from the output directory, taking the player count from the replay.

## Real physics scope

- Each automated driver has an independent MuJoCo WASM simulation and ONNX recurrent state, using the current BENI jump/recovery policy paired with the latest BENI model.
- Drivers send only speed, steering and jump requests. They do not directly modify position or velocity after initial placement.
- Ground, rails, bumps and shallow pits retain real collision geometry. Tours begin with a settling period and request jumps before course obstacles; ordinary clips retain a standing-jump demonstration.
- Real local WebSocket connections use the existing RaceService. During capture, the race clock follows simulated time.
- Original MuJoCo visual-part poses are saved, interpolated at fixed timestamps, and rendered frame by frame to avoid realtime screen-capture drops.
- This is a physical trajectory replay, not a realtime performance benchmark. Players have separate physics worlds; there are no inter-player rigid-body collisions.
- The default 45-second clip includes pickups but does not guarantee a two-lap finish. Capture never teleports or artificially lifts a robot to correct the result.

## Output and verification

- `OpenBeni-physics-3players-60fps.mp4`: the filename reflects player count and frame rate.
- `replay.json`: real room state and physical visual-part poses.
- `verification.json`: physics settings, contacts, jumps and reset records.
- `render-verification.json`: frame count, frame rate and browser errors.
- `frame-0.jpg` and `frame-600.jpg`: extracted frames for visual inspection; short clips may not reach frame 600.

The tool uses ffprobe to verify video frame count and frame rate. Git ignores `outputs/`.

Tours open with a three-second course overview, then use the same low, centered chase camera as the game to follow one driver through a complete circuit. At splits, the camera follows that driver rather than the average position of several drivers, avoiding tunnel roofs. Camera smoothing uses the actual output frame rate.

Recordings use High effects. Random gems, pickup stars, halos and small score labels come from real room state. Collected gems respawn at a new position after a random 1.8–4.4 seconds; uncollected gems relocate after 12–20 seconds. Positions span both sides of the road. Common gems score 1 point and rare pink gems score 3.

Every pickup also has a 25% chance to grant a 1.8-second turbo boost. Straight, flat sections allow a forward command up to 2 m/s; extra speed is withdrawn in turns, on slopes and before obstacles. There is no inventory or manual item action. The recording header shows only a small brand, course name, and the followed driver's position, lap, score and time. The full leaderboard, footer, event banner and overhead nameplates are hidden; the lower-right corner remains clear. Physics reports record each driver's pickups, gem score, jumps, maximum speed command and measured speed.

All three courses contain a mandatory trench: Toy Harbor is 30 cm wide and 8 cm deep, Cloud Climb is 28 cm wide and 9 cm deep, and Candy Junction is 26 cm wide and 8 cm deep. Other small bumps and pits also stay within 10 cm. Local, unpublished Jump regression tests use real MuJoCo + ONNX to check that rolling at 1.5 / 2 m/s cannot cross the trench, while a jump request can. Tour drivers slow before trenches and schedule jumps according to current speed, using policy commands only.

Capture uses the fixed random seed `20260914`; override it with `DEMO_SEED` to vary gem layout and rewards. Normal game rooms remain random. By default, the camera follows the last driver on the grid. Set `DEMO_FOCUS=1` to follow the second driver instead (indices start at zero); in a three-player split-route tour, this driver takes the tunnel.

The recording server disables file watching and hot reload to prevent mid-export page reloads. Headless Chrome uses SwiftShader so machines without a usable GPU retain a WebGL context.

## Files

`cli.mjs` is the default entry point. `capture.mjs` handles physics and multiplayer capture; `render.mjs` renders frames in the browser. `index.html`, `client.ts` and `replay.mjs` display the replay. The production game page does not include this development tool.

`legacy-kinematic.mjs` and `?kinematic=1` retain the earlier UI/network demonstration with scripted obstacle traversal. **They are not physics demonstrations** and are not used by the default recording command.

This worktree always records with the latest paired BENI model and policy. Verification reports record the policy SHA256, model name and actual physics timestep.

## Full-lap capability checks

A recording clip cannot replace full-course verification. After changing a model or course, run:

```bash
npm run test:laps
# Change the reproducible seed or check a single course
npm run test:laps -- --seed 42 --output outputs/lap-seed42
npm run test:laps -- --maps cloud --slots 0 --laps 1
npm run test:laps -- --maps switchyard --branches left,right
```

By default, `verify-laps.mjs` uses the current exported MuJoCo model and ONNX policy to run two consecutive laps from each of three front-row slots on all three courses. Candy Junction tests both branches, giving 12 runs. The cruising command is capped at 1.5 m/s, with pickup boosts reaching 2 m/s on flat straights and reductions before bends and obstacles. The driver sends only velocity, steering and real jump requests; it does not use inventory items, rescue, or modify position, velocity or external forces. Each run has independent physics and recurrent policy state. No server, Chrome or GPU is required.

Passing requires all ordered server checkpoints, at least two laps, and a continuous physical trajectory covering at least two full circuit lengths. The server's finish radius may award a lap slightly early; verification continues integrating to the full length. Every step checks time continuity and displacement bounds. Leaving the road, server resets, 15 seconds without progress, timeouts or missing checkpoints cause a nonzero exit. Use `--help` for options.

Outputs include `outputs/lap-verification/report.json` and a 20 Hz physical pose trace per run, named `{map}-{slot}.json` or `{map}-{slot}-{left|right}.json` for split routes. Reports contain policy/source-weight SHA256 values, robot and course fingerprints, lap times, checkpoint times, body contacts and reset counts. They verify that the current default policy can traverse the courses, not that every driving style or uploaded policy will finish.

The 2026-09-14 course update made Cloud Climb the default and added linked S-bends, hairpins and a cloudtop loop. Candy Junction added two independent routes of about 27.6 m each: an open garden bridge and a candy tunnel, separated by a real void. The split sections measure approximately 5.71 m / 5.67 m; the figure-eight crossing separates the decks vertically. Short bends use smooth continuous slopes, with bumps moved away from downhill corner exits to avoid overturning the current model.

Recorded capability checks from three grid slots on both branches completed two laps in approximately 46.9–48.1 seconds, including two seconds of settling, with no rescue, resets or body contact. Cloud Climb took about 105 seconds for two laps and Toy Harbor about 60 seconds. The corresponding reports are in `outputs/dual-route-laps/report.json` for the revised split routes and `outputs/complex-final/report.json` for Cloud Climb.
