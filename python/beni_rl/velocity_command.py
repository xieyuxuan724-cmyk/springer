# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
    import viser
    from mjlab.envs import ManagerBasedRlEnv


class BeniVelocityCommand(UniformVelocityCommand):
    """Velocity and pose commands with matching Viser controls."""

    cfg: BeniVelocityCommandCfg

    def __init__(self, cfg: BeniVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.base_pose_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_pose_command[:, 2] = sum(cfg.height_range) / 2.0
        self._pose_sliders = []

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if 'jump' in self._env.command_manager.active_terms:
            self._env.command_manager.get_term('jump').release_heading(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        for index, axis in enumerate(("pitch", "roll", "height")):
            self.base_pose_command[env_ids, index] = r.uniform_(*getattr(self.cfg, f"{axis}_range"))

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        jump = (self._env.command_manager.get_term('jump')
                if 'jump' in self._env.command_manager.active_terms else None)
        if jump is not None:
            # Pause command resampling until this jump lands or times out.
            held = jump.command_hold
            if env_ids is None:
                self.time_left += held * dt
            else:
                self.time_left[env_ids] += held[env_ids] * dt
        super().compute(dt, env_ids)
        if jump is not None:
            self.vel_command_b[held] = jump.state['request_twist'][held]
            # An explicit joystick turn ends a completed jump's heading hold.
            # During preparation/flight/landing it cannot override straight Jump.
            if self._joystick_enabled is not None and self._joystick_enabled.value:
                index = self._joystick_get_env_idx()
                if abs(self._joystick_sliders[2].value) > 1e-6:
                    jump.release_heading(jump._env_ids[index:index+1])
            self.vel_command_b[jump.heading_active, 2] = 0.
        if self._joystick_enabled is not None and self._joystick_enabled.value:
            assert self._joystick_get_env_idx is not None
            for index, slider in enumerate(self._pose_sliders):
                self.base_pose_command[self._joystick_get_env_idx(), index] = slider.value

    def create_gui(
        self,
        name: str,
        server: "viser.ViserServer",
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        self._joystick_get_env_idx = get_env_idx
        self._joystick_sliders = []
        self._pose_sliders = []
        with server.gui.add_folder(name.capitalize()):
            self._joystick_enabled = server.gui.add_checkbox("Enable", initial_value=False)
            for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                lower, upper = getattr(self.cfg.ranges, axis)
                self._joystick_sliders.append(
                    server.gui.add_slider(
                        axis,
                        min=lower,
                        max=upper,
                        step=0.05,
                        initial_value=0.0,
                        disabled=lower == upper,
                    )
                )
            for axis in ("pitch", "roll", "height"):
                lower, upper = getattr(self.cfg, f"{axis}_range")
                unit = "m" if axis == "height" else "rad"
                self._pose_sliders.append(
                    server.gui.add_slider(
                        f"base_{axis} ({unit})",
                        min=lower,
                        max=upper,
                        step=0.001 if axis == "height" else 0.01,
                        initial_value=(lower + upper) / 2.0,
                        disabled=lower == upper,
                    )
                )


@dataclass(kw_only=True)
class BeniVelocityCommandCfg(UniformVelocityCommandCfg):
    pitch_range: tuple[float, float] = (-0.1, 0.1)
    roll_range: tuple[float, float] = (-0.1, 0.1)
    height_range: tuple[float, float] = (0.10, 0.12)

    def build(self, env: "ManagerBasedRlEnv") -> BeniVelocityCommand:
        return BeniVelocityCommand(self, env)


class BeniGimbalCommand(CommandTerm):
    """Sample gimbal yaw targets and enable flags independently."""

    cfg: BeniGimbalCommandCfg

    def __init__(self, cfg: BeniGimbalCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self._command[:, 1] = 1.0
        self._override_enabled = None
        self._yaw_slider = None
        self._enable_checkbox = None
        self._get_env_idx = None

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._command[env_ids, 1] = (
            torch.rand(len(env_ids), device=self.device) < self.cfg.enable_probability
        ).float()
        # Sample yaw targets even when the gimbal command is disabled.
        self._command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.yaw_range)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if self._override_enabled is not None and self._override_enabled.value:
            assert self._get_env_idx is not None and self._yaw_slider is not None and self._enable_checkbox is not None
            index = self._get_env_idx()
            self._command[index, 0] = self._yaw_slider.value
            self._command[index, 1] = float(self._enable_checkbox.value)

    def create_gui(
        self,
        name: str,
        server: "viser.ViserServer",
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        self._get_env_idx = get_env_idx
        with server.gui.add_folder(name.capitalize()):
            self._override_enabled = server.gui.add_checkbox("Override", initial_value=False)
            self._yaw_slider = server.gui.add_slider(
                "yaw (rad)", min=self.cfg.yaw_range[0], max=self.cfg.yaw_range[1], step=0.05, initial_value=0.0
            )
            self._enable_checkbox = server.gui.add_checkbox("gimbal_enable", initial_value=True)


@dataclass(kw_only=True)
class BeniGimbalCommandCfg(CommandTermCfg):
    yaw_range: tuple[float, float] = (-5.0, 5.0)
    enable_probability: float = 0.5

    def build(self, env: ManagerBasedRlEnv) -> BeniGimbalCommand:
        return BeniGimbalCommand(self, env)
