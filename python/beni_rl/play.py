# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Run mjlab play with checkpoint Jump settings and open the Viser browser."""

import math
import webbrowser
from pathlib import Path
from threading import Thread

import yaml
import beni_rl  # noqa: F401
from . import _viser_logging  # noqa: F401
from .jump import BeniJumpCommandCfg
from mjlab.scripts import play as mjlab_play
from mjlab.viewer import ViserPlayViewer


def restore_checkpoint_jump_config(env_cfg, checkpoint_file: str) -> None:
    """Restore saved distance/cooldown settings with the fixed BENI Jump reference.

    Keep play's manual trigger, disabled curriculum/noise and reset settings.
    The training YAML contains Python tags; BaseLoader treats them as inert data.
    """
    jump = env_cfg.commands.get('jump')
    if not isinstance(jump, BeniJumpCommandCfg):
        return
    path = Path(checkpoint_file).expanduser().parent / 'params' / 'env.yaml'
    if not path.is_file():
        print(f'[WARN]: No checkpoint environment config at {path}; '
              f'using task defaults (Jump reference {jump.reference_height:g} m).')
        return
    try:
        saved = yaml.load(path.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)['commands']['jump']
        if not isinstance(saved, dict):
            raise ValueError('commands.jump must be a mapping')
        values = saved.get('distance_range', jump.distance_range)
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError('distance_range must contain two numbers')
        distance_range = tuple(float(x) for x in values)
        if not all(math.isfinite(x) for x in distance_range) or distance_range[0] > distance_range[1]:
            raise ValueError('distance_range must be finite and ordered')
        cooldown = int(saved.get('cool_down_steps', jump.cool_down_steps))
        if cooldown < 0:
            raise ValueError('cool_down_steps must be nonnegative')
    except (yaml.YAMLError, KeyError, TypeError, ValueError, FileNotFoundError) as error:
        raise ValueError(f'Invalid checkpoint Jump configuration in {path}: {error}') from error
    jump.distance_range = distance_range
    jump.cool_down_steps = cooldown
    print(f'[INFO]: Loaded checkpoint Jump config: {path}\n'
          f'[INFO]: Jump reference {jump.reference_height:g} m ({jump.reference_path.name}); '
          'training and play use the fixed reference.')


def _run_play(task_id, cfg, run_play):
    """Apply checkpoint settings after CLI parsing and before mjlab creates the env."""
    original_loader = mjlab_play.load_env_cfg

    def load_env_cfg(name, play=False):
        env_cfg = original_loader(name, play=play)
        if play and name == task_id and cfg.agent == 'trained' and cfg.checkpoint_file is not None:
            restore_checkpoint_jump_config(env_cfg, cfg.checkpoint_file)
        return env_cfg

    mjlab_play.load_env_cfg = load_env_cfg
    try:
        return run_play(task_id, cfg)
    finally:
        mjlab_play.load_env_cfg = original_loader


def _open_browser(url: str) -> None:
    try:
        opened = webbrowser.open_new(url)
    except (webbrowser.Error, OSError):
        opened = False
    if not opened:
        print(f"[INFO]: Could not open a browser automatically. Open {url} to view the simulation.")


class BeniViserPlayViewer(ViserPlayViewer):
    def setup(self) -> None:
        super().setup()
        url = f"http://localhost:{self._server.get_port()}"
        # Some browser launchers block until their window closes.
        Thread(target=_open_browser, args=(url,), daemon=True).start()


def main() -> None:
    original_viewer = mjlab_play.ViserPlayViewer
    original_run_play = mjlab_play.run_play
    mjlab_play.ViserPlayViewer = BeniViserPlayViewer
    mjlab_play.run_play = lambda task_id, cfg: _run_play(task_id, cfg, original_run_play)
    try:
        mjlab_play.main()
    finally:
        mjlab_play.ViserPlayViewer = original_viewer
        mjlab_play.run_play = original_run_play


if __name__ == "__main__":
    main()
