# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Finite wheel overspeed termination with a short, unmodified state history."""

import math
import os
from pathlib import Path
import time

import numpy as np
import torch


class WheelOverspeed:
    """Check actual wheel speed at the native control-step termination boundary.

    Nonfinite states retain the existing fail-fast observation behavior. This
    term neither repairs observations nor changes optimizer/learning-rate state.
    """

    def __init__(self, cfg, env):
        limit = cfg.params.get('max_speed', 350.)
        history = cfg.params.get('history_s', .5)
        if not math.isfinite(limit) or limit <= 0 or not math.isfinite(history) or history <= 0:
            raise ValueError('Wheel speed limit and history duration must be finite and positive')
        self.wheels = env.scene['robot'].find_joints(('LW', 'RW'), preserve_order=True)[0]
        self.capacity = max(2, math.ceil(history / env.step_dt) + 1)
        self.history = {}
        self.cursor = self.count = 0
        self.limit = limit
        self.step_dt = env.step_dt
        self.rank = int(os.environ.get('RANK', '0'))
        env._wheel_overspeed = self

    def capture(self, env):
        data = env.sim.data
        robot = env.scene['robot'].data
        frame = dict(qpos=data.qpos, qvel=data.qvel, qacc=data.qacc,
                     ctrl=data.ctrl, actuator_force=data.actuator_force,
                     action=env.action_manager.action,
                     joint_velocity_target=robot.joint_vel_target,
                     wheel_speed=robot.joint_vel[:, self.wheels],
                     base_ang_vel=robot.root_link_ang_vel_b,
                     body_contact_force=env.scene['body_ground_contact'].data.force,
                     wheel_contact_force=env.scene['wheel_ground_contact'].data.force,
                     episode_step=env.episode_length_buf,
                     control_step=torch.full_like(env.episode_length_buf, env.common_step_counter))
        for name, attr in (('reset_kind', '_curriculum_kind'), ('reset_state_id', '_recovery_state_id')):
            if hasattr(env, attr):
                frame[name] = getattr(env, attr)
        for name, value in frame.items():
            if name not in self.history:
                self.history[name] = torch.full((self.capacity, *value.shape), -1,
                                                dtype=value.dtype, device=value.device)
            self.history[name][self.cursor].copy_(value)
        self.cursor = (self.cursor + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def dump(self, env, ids, reason):
        if not len(ids) or not self.count:
            return
        directory = Path(getattr(env, '_diagnostics_dir', 'logs/diagnostics/wheel_overspeed')) / f'rank_{self.rank}'
        directory.mkdir(parents=True, exist_ok=True)
        order = (torch.arange(self.count, device=env.device) + self.cursor-self.count) % self.capacity
        record = {name: value[order[:, None], ids[None, :]].cpu().numpy()
                  for name, value in self.history.items()}
        record.update(env_ids=ids.cpu().numpy(), reason=np.array(reason), rank=np.array(self.rank),
                      control_dt_s=np.array(self.step_dt), max_speed_rad_s=np.array(self.limit),
                      wheel_joint_names=np.array(['LW', 'RW']),
                      body_contact_names=np.array(env.scene['body_ground_contact'].primary_names),
                      capture_phase=np.array('pre-reset control boundary; derived/contact state lags one physics substep'))
        path = directory / f'{reason}_step_{env.common_step_counter}_{time.time_ns()}.npz'
        np.savez_compressed(path, **record)
        print(f'[Wheel diagnostics] {len(ids)} environments: {path}', flush=True)

    def __call__(self, env, max_speed=350., history_s=.5):
        self.capture(env)
        speed = env.scene['robot'].data.joint_vel[:, self.wheels]
        # Do not turn an already nonfinite transition into an automatic reset.
        finite = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        for value in (env.sim.data.qpos, env.sim.data.qvel, env.sim.data.qacc, env.sim.data.sensordata):
            finite &= torch.isfinite(value).all(-1)
        terminated = finite & (speed.abs() > max_speed).any(-1)
        log = env.extras.setdefault('log', {})
        log['WheelSafety/max_abs_speed_rad_s'] = speed.abs().max()
        log['WheelSafety/overspeed_count'] = terminated.sum()
        ids = terminated.nonzero().flatten()
        if len(ids):
            self.dump(env, ids, 'overspeed')
        return terminated

    @torch.inference_mode()
    def dump_failure(self, env):
        """Save evidence on an existing exception; the caller re-raises it unchanged."""
        self.capture(env)
        invalid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for value in (env.sim.data.qpos, env.sim.data.qvel, env.sim.data.qacc, env.sim.data.sensordata,
                      env.scene['robot'].data.root_link_ang_vel_b):
            invalid |= ~torch.isfinite(value).all(-1)
        self.dump(env, invalid.nonzero().flatten(), 'physics_failure')
