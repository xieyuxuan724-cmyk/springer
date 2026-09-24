# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""BENI observations, rewards, resets and termination conditions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.rewards import variable_posture
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.manager_base import ManagerTermBaseCfg


def fallen_reset_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    initial_probability: float,
    final_probability: float,
    warmup_iterations: int,
    ramp_iterations: int,
    steps_per_iteration: int,
    start_iteration: int = 0,
    start_step: int = 0,
) -> float:
    """Increase fallen-reset probability after warmup using the PPO update count.

    The runner supplies the update origin and rollout length for resumed runs.
    """
    del env_ids  # The probability is shared; only resetting robots sample it.
    if not 0.0 <= initial_probability <= final_probability <= 1.0:
        raise ValueError("Reset probabilities must satisfy 0 <= initial <= final <= 1")
    if warmup_iterations < 0 or ramp_iterations <= 0 or steps_per_iteration <= 0:
        raise ValueError("Warmup must be non-negative; ramp and rollout lengths must be positive")
    iteration = start_iteration + (env.common_step_counter - start_step) // steps_per_iteration
    progress = min(1.0, max(0.0, (iteration - warmup_iterations) / ramp_iterations))
    probability = initial_probability + (final_probability - initial_probability) * progress
    env.event_manager.get_term_cfg("reset_base").params["fallen_probability"] = probability
    return probability


def reset_root_state_mixed(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    fallen_probability: float,
) -> None:
    """Mix the usual standing reset with prone, supine and two side-lying poses."""
    if not 0.0 <= fallen_probability <= 1.0:
        raise ValueError("fallen_probability must be between 0 and 1")
    env_ids = envs_mdp.resolve_env_ids(env, env_ids)
    if not hasattr(env, "_recovery_reset_kind"):
        env._recovery_reset_kind = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # Reset categories: normal, supine, prone, left, right, recorded support.
        env._recovery_reset_counts = torch.zeros(6, device=env.device)
    env._recovery_reset_kind[env_ids] = 0
    env._recovery_reset_counts[0] += len(env_ids)
    envs_mdp.reset_root_state_uniform(env, env_ids, pose_range, velocity_range)
    if fallen_probability == 0.0:
        return

    samples = torch.rand(len(env_ids), device=env.device)
    poses = (("pitch", -math.pi / 2), ("pitch", math.pi / 2), ("roll", -math.pi / 2), ("roll", math.pi / 2))
    for index, (axis, angle) in enumerate(poses):
        low = fallen_probability * index / len(poses)
        high = fallen_probability * (index + 1) / len(poses)
        fallen_ids = env_ids[(samples >= low) & (samples < high)]
        if fallen_ids.numel() == 0:
            continue
        env._recovery_reset_kind[fallen_ids] = index + 1
        env._recovery_reset_counts[0] -= len(fallen_ids)
        env._recovery_reset_counts[index + 1] += len(fallen_ids)
        fallen_pose_range = {
            **pose_range,
            # Raise the 0.14 m default root to 0.20 m so rotated collision shapes
            # start above the ground and settle without initial penetration.
            "z": (0.06, 0.06),
            "yaw": (-math.pi, math.pi),
            axis: (angle - 0.15, angle + 0.15),
        }
        envs_mdp.reset_root_state_uniform(env, fallen_ids, fallen_pose_range, velocity_range)


class constant_observation:
    """Cache a fixed observation vector on the environment device."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        self._value = torch.tensor(cfg.params["values"], dtype=torch.float32, device=env.device).expand(
            env.num_envs, -1
        )

    def __call__(self, env: ManagerBasedRlEnv, values: tuple[float, ...]) -> torch.Tensor:
        del env, values
        return self._value


def track_linear_velocity_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward velocity tracking only while the wheels support an upright body."""
    robot: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    velocity_yaw = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), robot.data.root_link_lin_vel_w)
    error = torch.sum(torch.square(command[:, :2] - velocity_yaw[:, :2]), dim=1)
    return torch.exp(-error / std**2) * standing_support(env)


def track_linear_velocity_l1(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    velocity_yaw = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), robot.data.root_link_lin_vel_w)
    return torch.sum(torch.abs(command[:, :2] - velocity_yaw[:, :2]), dim=1)


def track_yaw_velocity_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, require_support: bool = True,
) -> torch.Tensor:
    """Track world-z angular velocity with the ordinary standing-support gate."""
    robot: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = torch.square(command[:, 2] - robot.data.root_link_ang_vel_w[:, 2])
    reward = torch.exp(-error / std**2)
    return reward * standing_support(env) if require_support else reward


class supported_posture(variable_posture):
    """Apply the speed-dependent posture reward while standing support is valid."""

    def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
        return super().__call__(env, **kwargs) * standing_support(env)


def base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Root height above this environment's flat ground, as a critic observation."""
    robot: Entity = env.scene["robot"]
    return (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(1)


def base_pose_command(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Share the sampled pose command between actor and critic observations."""
    return env.command_manager.get_term("twist").base_pose_command


def base_height_below_minimum(env: ManagerBasedRlEnv, minimum: float) -> torch.Tensor:
    """Return the base-height shortfall below the minimum."""
    return (minimum - base_height(env).squeeze(1)).clamp_min(0.0)


def leg_symmetry_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize measured L1/R1 and L2/R2 angle differences (both axes point -Y)."""
    robot: Entity = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]  # L1, R1, L2, R2.
    return (joint_pos[:, 0::2] - joint_pos[:, 1::2]).square().sum(dim=1)


def base_tilt(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return body tilt in radians, from zero upright to pi upside down."""
    robot: Entity = env.scene["robot"]
    gravity = robot.data.projected_gravity_b
    return torch.atan2(torch.linalg.vector_norm(gravity[:, :2], dim=1), -gravity[:, 2])


def large_tilt_l2(
    env: ManagerBasedRlEnv, start_angle_deg: float = 30., full_angle_deg: float = 60.,
) -> torch.Tensor:
    """Bounded excess-tilt cost; the caller selects the applicable Jump window.

    The ordinary base_tilt term still increases beyond full_angle_deg.
    """
    if not 0 <= start_angle_deg < full_angle_deg <= 180:
        raise ValueError('Tilt angles require 0 <= start < full <= 180 degrees')
    excess = (base_tilt(env)-math.radians(start_angle_deg)) / math.radians(full_angle_deg-start_angle_deg)
    return excess.clamp(0, 1).square()


def vertical_velocity_l2(env: ManagerBasedRlEnv, clip: float = 1.0) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    return torch.square(robot.data.root_link_lin_vel_w[:, 2]).clamp(max=clip)


def roll_pitch_velocity_l2(env: ManagerBasedRlEnv, clip: float = 20.0) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    velocity_yaw = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), robot.data.root_link_ang_vel_w)
    return torch.sum(torch.square(velocity_yaw[:, :2]), dim=1).clamp(max=clip)


def masked_action_rate_l2(env: ManagerBasedRlEnv, indices: tuple[int, ...]) -> torch.Tensor:
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta[:, indices]), dim=1)


def masked_action_acc_l2(env: ManagerBasedRlEnv, indices: tuple[int, ...]) -> torch.Tensor:
    delta = env.action_manager.action - 2.0 * env.action_manager.prev_action + env.action_manager.prev_prev_action
    penalty = torch.sum(torch.square(delta[:, indices]), dim=1)
    penalty[env.episode_length_buf < 3] = 0.0
    return penalty


def _contact_mask(
    env: ManagerBasedRlEnv, sensor_name: str, threshold: float, *, use_history: bool = True
) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, ContactSensor)
    if use_history and sensor.data.force_history is not None:
        force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1)
        return (force > threshold).any(dim=-1)
    assert sensor.data.force is not None
    return torch.linalg.vector_norm(sensor.data.force, dim=-1) > threshold


def standing_support(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Gate positive rewards by current wheel support, body clearance and tilt.

    Both wheels must contact the ground (>0.1 N), with no recent contact from
    the base, legs or head. Height fades in from 0.065 to 0.10 m; the upright
    factor fades out between 20 and 60 degrees (linearly in tilt cosine).
    """
    wheels = _contact_mask(env, "wheel_ground_contact", 0.1, use_history=False).all(dim=1)
    body = _contact_mask(env, "body_ground_contact", 0.1).any(dim=1)
    height = ((base_height(env).squeeze(1) - 0.065) / (0.10 - 0.065)).clamp(0.0, 1.0)
    upright_cos = -env.scene["robot"].data.projected_gravity_b[:, 2]
    lower = math.cos(math.radians(60))
    upper = math.cos(math.radians(20))
    upright = ((upright_cos - lower) / (upper - lower)).clamp(0.0, 1.0)
    return (wheels & ~body).float() * height * upright


def contact_count(env: ManagerBasedRlEnv, sensor_name: str, threshold: float) -> torch.Tensor:
    return _contact_mask(env, sensor_name, threshold).sum(dim=1).float()


def any_contact(env: ManagerBasedRlEnv, sensor_name: str, threshold: float) -> torch.Tensor:
    return _contact_mask(env, sensor_name, threshold).any(dim=1).float()


class delayed_illegal_contact:
    """Terminate after a contact violation persists for ``max_steps`` steps."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        del cfg
        self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        threshold: float,
        max_steps: int,
    ) -> torch.Tensor:
        failed = _contact_mask(env, sensor_name, threshold).any(dim=1)
        self._count[failed] += 1
        self._count[~failed] = 0
        return self._count > max_steps

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._count[env_ids] = 0


class delayed_bad_orientation:
    """Allow recovery until excessive tilt persists for ``max_steps`` steps."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        del cfg
        self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, limit_angle: float, max_steps: int) -> torch.Tensor:
        failed = envs_mdp.bad_orientation(env, limit_angle)
        self._count[failed] += 1
        self._count[~failed] = 0
        return self._count > max_steps

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._count[env_ids] = 0
