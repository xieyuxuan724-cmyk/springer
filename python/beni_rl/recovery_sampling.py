# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Adaptive reset sampling after the foundation grounding curriculum completes."""

import numpy as np
import torch
from mjlab.managers.recorder_manager import RecorderTerm

from .recovery_bank import SUBGROUPS, bank_digest, load_recovery_bank, sample_bank_indices

SAMPLING_VERSION = 2
UPDATE_INTERVAL = 500
RECOVERY_HOLD_S = .2
CRITERIA = ('upright', 'height', 'joint_margin', 'wheel_contact', 'wheel_load',
            'body_clear', 'linear_tracking', 'roll_pitch_speed', 'yaw_tracking')
DIAGNOSTIC_CRITERIA = (*CRITERIA, 'strict_posture', 'strict_height', 'strict_angular_speed')
DIAGNOSTICS = ('frames', *DIAGNOSTIC_CRITERIA, 'ready_frames', 'strict_ready_frames',
               'failed_termination', 'failed_deadline')


def unique_recovery_states(bank):
    """Give identical qpos/qvel one ID; retain original row order for foundation draws."""
    from .recovery_curriculum import KINDS
    q = np.concatenate([bank[f'{kind}_qpos'] for kind in KINDS[1:]])
    v = np.concatenate([bank[f'{kind}_qvel'] for kind in KINDS[1:]])
    _, first, inverse = np.unique(np.concatenate((q, v), axis=1), axis=0,
                                  return_index=True, return_inverse=True)
    classes = np.concatenate([np.full(len(bank[f'{kind}_qpos']), i)
                              for i, kind in enumerate(KINDS[1:], 1)])
    labels = np.zeros((len(first), 8 + len(SUBGROUPS)), dtype=bool)
    rows, offset = {}, 0
    for i, kind in enumerate(KINDS[1:]):
        size = len(bank[f'{kind}_qpos'])
        ids = inverse[offset:offset+size]
        rows[kind] = ids.copy()
        labels[ids, i] = True
        if f'{kind}_tags' in bank:
            np.logical_or.at(labels[:, 8:], ids, bank[f'{kind}_tags'])
        offset += size
    return q[first], v[first], classes[first], rows, labels


def sampling_probabilities(failure_rate):
    """Half uniform coverage, half bounded difficulty; never starve easy states."""
    weight = .1 + .9 * failure_rate
    return .5 / len(weight) + .5 * weight / weight.sum()


def recovery_conditions(sample, velocity_b, angular_b, command, target_height):
    """Functional recovery under the current command; final quality uses the strict gate."""
    return dict(
        upright=sample['tilt'] < 30,
        height=(sample['height'] >= .08) & ((sample['height']-target_height).abs() <= .025),
        joint_margin=sample['margin'] > .02,
        wheel_contact=sample['contacts'].all(-1),
        wheel_load=sample['load'] >= .7,
        body_clear=~sample['body'] & ~sample['body_current'],
        linear_tracking=(velocity_b[:, :2]-command[:, :2]).norm(dim=-1) < .5,
        roll_pitch_speed=angular_b[:, :2].norm(dim=-1) < 2,
        yaw_tracking=(angular_b[:, 2]-command[:, 2]).abs() < 1.)


class AdaptiveRecoverySampler(RecorderTerm):
    """Observe initial recovery only; never alter simulation or learner state."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from .jump import JumpMeasurement
        path = env.cfg.events['reset_curriculum'].params['bank_path']
        self.bank_sha256 = bank_digest(path)
        q, v, kinds, rows, labels = unique_recovery_states(load_recovery_bank(path))
        self.qpos = torch.tensor(q, device=env.device)
        self.qvel = torch.tensor(v, device=env.device)
        self.kinds = torch.tensor(kinds, device=env.device)
        self.rows = {kind: torch.tensor(ids, device=env.device) for kind, ids in rows.items()}
        self.labels = torch.tensor(labels, device=env.device)
        self.measure = JumpMeasurement(env)
        gate = getattr(env.cfg.commands.get('jump'), 'recovery_gate', None)
        self.hold = max(1, round(RECOVERY_HOLD_S / env.step_dt))
        self.deadline = round((gate.deadline_s if gate else 6.) / env.step_dt)
        # Rows are failures, completed attempts and reset draws. Pending is local;
        # window and EMA are identical on all ranks after each PPO update.
        self.pending = torch.zeros(3, len(q), device=env.device)
        self.window = torch.zeros_like(self.pending)
        self.ema = torch.zeros(2, len(q), device=env.device)
        self.diagnostic_pending = torch.zeros(len(DIAGNOSTICS), device=env.device, dtype=torch.float64)
        self.diagnostic_window = torch.zeros_like(self.diagnostic_pending)
        self.failure_rate = torch.full((len(q),), .5, device=env.device)
        self.probability = sampling_probabilities(self.failure_rate)
        self.started_at = None
        self.last_update = 0
        self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.streak = torch.zeros_like(self.age)
        env._recovery_state_id = torch.full_like(self.age, -1)
        env._recovery_sampler = self

    def _clear_statistics(self):
        self.pending.zero_(); self.window.zero_(); self.ema.zero_()
        self.diagnostic_pending.zero_(); self.diagnostic_window.zero_()
        self.failure_rate.fill_(.5)
        self.probability = sampling_probabilities(self.failure_rate)
        self.active.zero_(); self.age.zero_(); self.streak.zero_()

    def start(self, completed_updates):
        """Start once, excluding all foundation statistics and episodes already in flight."""
        if self.started_at is not None:
            return
        self._clear_statistics()
        self.started_at = self.last_update = int(completed_updates)

    def draw(self, count, kind=None):
        if kind is None:
            if self.started_at is None:
                raise RuntimeError('Whole-bank sampling requires the completed grounding curriculum')
            return torch.multinomial(self.probability, count, replacement=True)
        ids = self.rows[kind]
        # Keep the foundation source intact during the 500-update source blend.
        groups = getattr(self._env, '_hard_recovery_groups', {}).get(kind, ())
        return ids[sample_bank_indices(len(ids), count, self._env.device, groups)]

    def kind_probabilities(self):
        return torch.bincount(self.kinds, weights=self.probability, minlength=9)[1:]

    def register_draws(self, ids):
        if self.started_at is None:
            return
        states = self._env._recovery_state_id[ids]
        states = states[states >= 0]
        self.pending[2].index_add_(0, states, torch.ones_like(states, dtype=self.pending.dtype))

    def _stable(self, eligible):
        from .mdp import base_pose_command
        from .recovery import H_REF
        sample = self.measure(self._env)
        robot = self._env.scene['robot'].data
        command = self._env.command_manager.get_command('twist')
        conditions = recovery_conditions(sample, robot.root_link_lin_vel_b, robot.root_link_ang_vel_b,
                                         command, base_pose_command(self._env)[:, 2])
        ready = torch.stack(tuple(conditions.values())).all(0)
        # Diagnostic comparisons do not affect functional recovery or the strict gate.
        conditions.update(strict_posture=sample['error'] <= .10,
                          strict_height=(sample['height']-H_REF).abs() <= .01,
                          strict_angular_speed=robot.root_link_ang_vel_b.norm(dim=-1) < 2)
        active = self.active & eligible
        counts = [active.sum(), *((active & ~conditions[k]).sum() for k in DIAGNOSTIC_CRITERIA),
                  (active & ready).sum(), (active & sample['stable'] & ~sample['body_current']).sum()]
        self.diagnostic_pending[:len(counts)] += torch.stack(counts)
        return ready

    def _finish(self, ids, success):
        ids = ids[self.active[ids]]
        states = self._env._recovery_state_id[ids]
        one = torch.ones_like(states, dtype=self.pending.dtype)
        self.pending[1].index_add_(0, states, one)
        if not success:
            self.pending[0].index_add_(0, states, one)
        self.active[ids] = False

    def record_pre_reset(self, env_ids):
        if self.started_at is None:
            return
        env = self._env
        # Native finite failure before confirmation counts once. Early timeouts
        # are incomplete trials; nonfinite states remain the existing fatal path.
        valid = env.reset_terminated[env_ids].clone() & self.active[env_ids]
        for value in (env.sim.data.qpos, env.sim.data.qvel, env.sim.data.qacc):
            valid &= torch.isfinite(value[env_ids]).all(-1)
        self.diagnostic_pending[-2] += valid.sum()
        self._finish(env_ids[valid], success=False)
        self.active[env_ids] = False

    def record_post_reset(self, env_ids):
        if self.started_at is None:
            return
        env = self._env
        self.active[env_ids] = env._recovery_state_id[env_ids] >= 0
        self.age[env_ids] = 0
        eligible = torch.zeros_like(self.active); eligible[env_ids] = True
        self.streak[env_ids] = self._stable(eligible)[env_ids].long()
        self._finish(env_ids[self.streak[env_ids] >= self.hold], success=True)

    def observe(self, stable, eligible):
        """Consume one sensed frame; include a hold starting exactly at the deadline."""
        active = self.active & eligible
        self.age[active] += 1
        self.streak[active] = torch.where(stable[active], self.streak[active]+1, 0)
        success = active & (self.streak >= self.hold)
        self._finish(success.nonzero().flatten(), success=True)
        expired = active & self.active & (self.age >= self.deadline+self.hold-1)
        self.diagnostic_pending[-1] += expired.sum()
        self._finish(expired.nonzero().flatten(), success=False)

    def record_post_step(self):
        if self.started_at is None:
            return
        env = self._env
        if 'jump' in env.cfg.commands:
            # Later landing outcomes must not be attributed to the initial pose.
            self.active &= ~env.command_manager.get_term('jump').jump_flag
        self.observe(self._stable(~env.reset_buf), ~env.reset_buf)

    def update(self, completed_updates):
        """Pool each rollout for exact checkpoints; refresh weights every 500 updates."""
        if self.started_at is not None:
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(self.pending)
                torch.distributed.all_reduce(self.diagnostic_pending)
            self.window += self.pending
            self.diagnostic_window += self.diagnostic_pending
            self.pending.zero_(); self.diagnostic_pending.zero_()
        if completed_updates-self.last_update < UPDATE_INTERVAL:
            return False
        if self.started_at is not None:
            self.ema.mul_(.8).add_(self.window[:2], alpha=.2)
            # Unknown states have a one-success/one-failure prior.
            self.failure_rate = (self.ema[0]+1) / (self.ema[1]+2)
            self.probability = sampling_probabilities(self.failure_rate)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(self.probability, src=0)
        self.last_update = completed_updates
        self.last_metrics = self.metrics()
        self.window.zero_(); self.diagnostic_window.zero_()
        return True

    def metrics(self):
        from .recovery_curriculum import KINDS
        values = dict(enabled=self.started_at is not None, start_updates=self.started_at if self.started_at is not None else -1,
                      unique_states=len(self.qpos), attempted_states=(self.window[1] > 0).sum(),
                      completed_attempts=self.window[1].sum(), failures=self.window[0].sum(),
                      successes=(self.window[1]-self.window[0]).sum(),
                      largest_probability=self.probability.max(), effective_states=1 / self.probability.square().sum())
        frames = self.diagnostic_window[0].clamp_min(1)
        values['observed_frames'] = self.diagnostic_window[0]
        for i, name in enumerate(DIAGNOSTIC_CRITERIA, 1):
            values[f'criteria/{name}_failure_share'] = self.diagnostic_window[i] / frames
        for name in ('ready_frames', 'strict_ready_frames'):
            values[name.removesuffix('_frames')+'_frame_share'] = self.diagnostic_window[DIAGNOSTICS.index(name)] / frames
        for name in ('failed_termination', 'failed_deadline'):
            values[name] = self.diagnostic_window[DIAGNOSTICS.index(name)]
        for i, name in enumerate((*KINDS[1:], *SUBGROUPS)):
            mask = self.labels[:, i]
            values[f'{name}/pool_probability'] = self.probability[mask].sum()
            values[f'{name}/attempts'] = self.window[1, mask].sum()
            values[f'{name}/failures'] = self.window[0, mask].sum()
        return {key: float(value) for key, value in values.items()}

    def state_dict(self):
        return dict(version=SAMPLING_VERSION, bank_sha256=self.bank_sha256,
                    ema=self.ema.cpu().clone(), window=self.window.cpu().clone(),
                    diagnostic_window=self.diagnostic_window.cpu().clone(),
                    started_at=self.started_at, last_update=self.last_update)

    def load_state_dict(self, state):
        if not state or state.get('version') != SAMPLING_VERSION or state.get('bank_sha256') != self.bank_sha256:
            self._clear_statistics()
            self.started_at = None
            self.last_update = 0
            return False
        for name in ('ema', 'window', 'diagnostic_window'):
            value = state[name].to(getattr(self, name).device)
            if (value.shape != getattr(self, name).shape or not torch.isfinite(value).all()
                    or (value < 0).any() or (name != 'diagnostic_window' and (value[0] > value[1]).any())):
                raise ValueError(f'Invalid saved recovery sampling {name}')
            getattr(self, name).copy_(value)
        self.started_at = state['started_at']
        self.last_update = int(state['last_update'])
        if self.last_update < 0 or (self.started_at is not None and
                                   (not isinstance(self.started_at, int) or not 0 <= self.started_at <= self.last_update)):
            raise ValueError('Invalid saved recovery sampling update')
        if self.started_at is None and any(bool(getattr(self, name).any()) for name in ('ema', 'window', 'diagnostic_window')):
            raise ValueError('Inactive recovery sampling checkpoint contains adaptive history')
        self.failure_rate = (self.ema[0]+1) / (self.ema[1]+2)
        self.probability = sampling_probabilities(self.failure_rate)
        self.pending.zero_(); self.diagnostic_pending.zero_()
        self.active.zero_()  # Simulator episodes are intentionally not resumed.
        return True
