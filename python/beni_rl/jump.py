# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT
"""Jump commands, takeoff and landing measurements, and phase-dependent rewards.

Measurements are updated before reward computation. Commands advance afterwards
for the next control step.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, yaw_quat

from . import ASSETS_DIR, mdp
from .recovery import H_REF, Q_REF, position_limits
from .recovery_gate import RecoveryGateCfg
from .jump_evaluation import JumpEvaluationCfg

# t, xyz, quat_wxyz, velocity_xyz, angular_velocity_xyz, target_height, stage.
JUMP_REFERENCE_HEIGHT = .18
JUMP_REFERENCE_PATH = ASSETS_DIR / 'motions' / f'base_jump_h{JUMP_REFERENCE_HEIGHT:g}.npz'
THRESHOLDS = dict(knee_delta_rad=.15, retraction_m=.005, tuck_hold_s=.03,
                  clearance_m=.01, air_hold_s=.05, ascent_vz_m_s=.05,
                  takeoff_deadline_s=.8, protection_s=2., landing_ramp_s=.2,
                  landing_stable_deadline_s=1.5, stable_hold_s=1., followup_s=5.,
                  readiness_s=.2, flight_tilt_deg=30.)
READINESS_THRESHOLDS = dict(tilt_deg=20., height_tolerance_m=.025, joint_error_rad=.30,
                    margin_rad=.02, load_mg=.7, angular_speed_rad_s=2.)
DIRECTION_TOLERANCE_M = 1e-5
# Hip-to-wheel distances at standing joint pose Q_REF, ordered left then right.
STANDING_LEG_LENGTH_M = (.08095823563057503, .08060120486022314)
TUCK_KNEE_RAD = (Q_REF['L2'] + .15, Q_REF['R2'] + .15)
TUCK_LEG_LENGTH_M = tuple(x - .005 for x in STANDING_LEG_LENGTH_M)


THRESHOLDS.update(goal_clearance_m=.08, tuck_knee_rad=TUCK_KNEE_RAD,
                  tuck_length_m=TUCK_LEG_LENGTH_M, tuck_at_apex_required=True)


class JumpMeasurement:
    def __init__(self, env):
        robot = env.scene['robot']
        self.legs = robot.find_joints(list(Q_REF), preserve_order=True)[0]
        self.knees = robot.find_joints(('L2', 'R2'), preserve_order=True)[0]
        self.hips = robot.find_bodies(('L1_Link', 'R1_Link'), preserve_order=True)[0]
        self.wheels = robot.find_bodies(('LW_Link', 'RW_Link'), preserve_order=True)[0]
        self.geoms = robot.find_geoms(('LW_collision', 'RW_collision'), preserve_order=True)[0]
        self.ref = torch.tensor(list(Q_REF.values()), device=env.device)
        self.weight = float(env.sim.mj_model.body_mass.sum() * 9.81)

    def __call__(self, env):
        d = env.scene['robot'].data
        force = env.scene['wheel_ground_contact'].data.force
        body_force = env.scene['body_ground_contact'].data.force
        contacts = force.norm(dim=-1) > .1
        body = (body_force.norm(dim=-1) > .1).any(-1)
        history = env.scene['body_ground_contact'].data.force_history
        # Detect body contacts across every physics substep in the control interval.
        if history is None or history.shape[-2] != env.cfg.decimation:
            raise ValueError('Jump body evidence requires one full control interval of force history')
        peak = history.norm(dim=-1).amax(-1)
        evidence = dict(body_current=body, body_substep_force_max=peak)
        body = (peak > .1).any(-1)
        q = d.joint_pos[:, self.legs]
        limits = d.joint_pos_limits[:, self.legs]
        margin = torch.minimum(q-limits[..., 0], limits[..., 1]-q).amin(-1)
        height = mdp.base_height(env).squeeze(-1)
        tilt = torch.rad2deg(torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1, 1)))
        load = (-force[..., 2]).clamp_min(0).sum(-1) / self.weight
        error = (q-self.ref).abs().amax(-1)
        speed = d.root_link_lin_vel_w[:, :2].norm(dim=-1)
        angular = d.root_link_ang_vel_w.norm(dim=-1)
        stable = ((tilt < 20) & ((height-H_REF).abs() <= .01) & (error <= .10) &
                  (margin > .02) & contacts.all(-1) & (load >= .7) & ~body &
                  (speed < .3) & (angular < 2))
        r = READINESS_THRESHOLDS
        jump_ready = ((tilt < r['tilt_deg']) & ((height-H_REF).abs() <= r['height_tolerance_m']) &
                      (error <= r['joint_error_rad']) & (margin > r['margin_rad']) &
                      contacts.all(-1) & (load >= r['load_mg']) & ~body & (angular < r['angular_speed_rad_s']))
        quat = d.geom_quat_w[:, self.geoms]
        axis = torch.zeros_like(quat[..., :3]); axis[..., 2] = 1
        vertical = quat_apply(quat, axis)[..., 2].abs()
        extent = .038 * (1-vertical.square()).clamp_min(0).sqrt() + .0105 * vertical
        clearance = d.geom_pos_w[:, self.geoms, 2] - env.scene.env_origins[:, 2:3] - extent
        air = ~contacts.any(-1) & ~body & (clearance >= THRESHOLDS['clearance_m']).all(-1)
        return dict(height=height, vz=d.root_link_lin_vel_w[:, 2], tilt=tilt, body=body,
                    position_xy=d.root_link_pos_w[:, :2], velocity_xy=d.root_link_lin_vel_w[:, :2],
                    quat=d.root_link_quat_w,
                    contacts=contacts, load=load, stable=stable, jump_ready=jump_ready, margin=margin, error=error,
                    knee=d.joint_pos[:, self.knees],
                    length=(d.body_link_pos_w[:, self.wheels]-d.body_link_pos_w[:, self.hips]).norm(dim=-1),
                    clearance=clearance, air=air, impact=force.norm(dim=-1).amax(-1), **evidence)


def strong_tuck_progress(knee, length, knee0, length0, margin):
    """Measure bilateral knee flexion and leg shortening toward the tucked pose."""
    required_knee = (knee.new_tensor(TUCK_KNEE_RAD) - knee0).clamp_min(THRESHOLDS['knee_delta_rad'])
    required_length = (length0 - length.new_tensor(TUCK_LEG_LENGTH_M)).clamp_min(THRESHOLDS['retraction_m'])
    progress = torch.minimum((knee-knee0)/required_knee, (length0-length)/required_length)
    return progress.clamp(0, 1).amin(-1) * (margin > .02)


def jump_episode_probability(iteration, start=0, ramp=1000):
    return min(1., max(0., (iteration-start)/ramp)) / 3


class BeniJumpCommand(CommandTerm):
    cfg: BeniJumpCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if (not 0 <= cfg.probability <= 1 or cfg.warmup_iterations < 0 or cfg.cool_down_steps < 0
                or not math.isfinite(cfg.ramp_iterations) or cfg.ramp_iterations <= 0):
            raise ValueError('Invalid jump probability, warmup or cooldown')
        for limits in (cfg.target_height_range, cfg.distance_range):
            if not all(math.isfinite(x) for x in limits) or limits[0] > limits[1]:
                raise ValueError('Jump ranges must be finite and ordered')
        if (not all(math.isfinite(x) and x > 0 for x in
                    (cfg.takeoff_yaw_rate, cfg.yaw_settle_s, cfg.prepare_timeout_s)) or
                cfg.yaw_settle_s < env.step_dt or cfg.prepare_timeout_s <= cfg.yaw_settle_s):
            raise ValueError('Jump yaw preparation requires positive rate and timeout > settle >= dt')
        with np.load(cfg.reference_path, allow_pickle=False) as f:
            trajectory = f['trajectory']
        if trajectory.shape != (41, 16) or not np.allclose(np.diff(trajectory[:, 0]), env.step_dt):
            raise ValueError('Jump reference requires 41 frames at control dt=.01s')
        self._reference = torch.tensor(trajectory, device=self.device, dtype=torch.float32)
        self._env_ids = torch.arange(self.num_envs, device=self.device)
        self.steps_per_iteration = 24
        self._warmup_end_step = 0 if cfg.play else math.inf
        self._jump_start_iteration = None
        self._jump_ramp_iterations = cfg.ramp_iterations
        self.measure = JumpMeasurement(env)
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(self.num_envs, *shape, device=self.device, dtype=dtype)
        self.jump_flag = zeros(dtype=torch.bool)
        self.jump_count = zeros(dtype=torch.long)
        self.jump_cool_down = zeros(dtype=torch.long)
        self.target_height = zeros(); self.distance = zeros()
        self.subset = zeros(dtype=torch.bool)
        self.ready_steps = zeros(dtype=torch.long)
        self.preparing = zeros(dtype=torch.bool)
        self.prepare_age = zeros()
        self.yaw_ready_steps = zeros(dtype=torch.long)
        self.heading_active = zeros(dtype=torch.bool)
        self.heading_reference = zeros()
        self.heading_reward_active = zeros(dtype=torch.bool)
        # Accumulate request statistics across episode resets.
        self.counts = {k: zeros(1).sum() for k in ('requests', 'accepted', 'rejected_not_ready',
                      'rejected_busy', 'rejected_cooldown', 'rejected_disabled', 'terminated')}
        self.events = deque(maxlen=4096)
        self.phase_rewards = None
        self.sample = None
        self._last_observed_step = -1
        self.state = {k: zeros(dtype=torch.bool) for k in ('tracking', 'gate', 'lifted', 'apex_seen',
                     'had_ascent', 'tucked', 'apex_tucked', 'body_hit', 'bad_flight', 'failed', 'post_bad',
                     'attitude_active')}
        self.state.update({k: zeros() for k in ('age', 'start_height', 'apex', 'air_run', 'air_max',
                 'ascent_frames', 'tuck_frames', 'tuck_run', 'tuck_max', 'ready_hold', 'max_tilt', 'impact',
                 'flight_peak_height', 'apex_clearance', 'max_clearance')})
        self.state.update({k: torch.full_like(self.target_height, -1) for k in
                           ('takeoff_s', 'land_s', 'stable_s', 'tuck_s')})
        self.state.update({k: zeros(2) for k in ('knee0', 'length0', 'knee_max', 'retraction_max')})
        self.state.update({k: zeros(2) for k in ('request_xy', 'forward_xy')})
        self.state['request_twist'] = zeros(3)
        self.state.update({k: zeros() for k in ('direction', 'displacement', 'takeoff_displacement',
                                              'flight_displacement', 'reverse_distance')})
        self.knee_delta = zeros(2); self.retraction = zeros(2)
        self.real_ascent = zeros(dtype=torch.bool)
        self.progress = zeros()
        self.body_failure_event = zeros(dtype=torch.bool)
        self.attitude_reward_active = zeros(dtype=torch.bool)
        self.flat_scale = torch.ones_like(self.target_height)
        self.sample_phase = zeros(dtype=torch.long)
        self._requested = zeros(dtype=torch.bool)
        self._snapshot_command = zeros(4)
        self.window_start_step = env.common_step_counter
        # Bound to the native step event at startup, after managers exist.
        self._random_push = None

    def configure_training(self, start_iteration, num_steps_per_env,
                           jump_start_iteration=None, ramp_iterations=None):
        self.steps_per_iteration = num_steps_per_env
        self._jump_start_iteration = jump_start_iteration
        self._jump_ramp_iterations = self.cfg.ramp_iterations if ramp_iterations is None else ramp_iterations
        self._warmup_end_step = (math.inf if jump_start_iteration is None else
                                self._env.common_step_counter+(jump_start_iteration-start_iteration)*num_steps_per_env)
        self.window_start_step = self._env.common_step_counter

    @property
    def jump_phase(self):
        return self._reference[self.jump_count, 0]/self._reference[-1, 0]

    @property
    def jump_stage(self):
        return self._reference[self.jump_count, 15].long()

    @property
    def jump_ref_height(self):
        return self._reference[self.jump_count, 3]

    @property
    def jump_ref_vel(self):
        return self._reference[self.jump_count, 10]

    @property
    def ascent(self):
        # Reference ascent phase, including the grounded push-off.
        mask = self.jump_flag & (self.jump_stage == 1)
        return mask

    @property
    def command(self):
        return torch.stack((self.jump_flag, self.jump_phase, self.target_height, self.distance), -1) * self.jump_flag[:, None]

    def _update_metrics(self):
        pass

    @property
    def direction_active(self):
        """Protect direction through propulsion and flight, including reference cooldown."""
        s = self.state
        return (s['tracking'] & ~self.preparing & (s['land_s'] < 0) & (s['age'] < THRESHOLDS['protection_s']) &
                ((s['age'] <= THRESHOLDS['takeoff_deadline_s']) | s['lifted']))

    @property
    def command_hold(self):
        """Hold a straight command through braking, flight and landing stabilization."""
        s = self.state
        landing = (s['tracking'] & (s['land_s'] >= 0) & (s['stable_s'] < 0) &
                   (s['age'] < s['land_s'] + THRESHOLDS['landing_stable_deadline_s']))
        return self.heading_active & (self.preparing | self.direction_active | landing)

    @property
    def heading_error(self):
        delta = self._env.scene['robot'].data.heading_w - self.heading_reference
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    def release_heading(self, ids):
        """A new locomotion command may release a completed jump's heading target."""
        ids = ids[~self.command_hold[ids]]
        self.heading_active[ids] = False

    def reset(self, env_ids):
        # Reset Jump state without consuming random samples during recovery training.
        if env_ids is None:
            env_ids = self._env_ids
        self._resample_command(env_ids)
        return {}

    def _clear_command(self, ids):
        self.jump_flag[ids] = False; self.jump_count[ids] = 0
        self.target_height[ids] = 0; self.distance[ids] = 0

    def _resample_command(self, env_ids):
        ids = env_ids
        unfinished = ids[self.state['tracking'][ids]]
        if len(unfinished):
            self.state['failed'][unfinished] = True
            self._finish(unfinished, terminated=True)
        for key, value in self.state.items():
            value[ids] = -1 if key in ('takeoff_s', 'land_s', 'stable_s', 'tuck_s') else 0
        self._clear_command(ids)
        self.jump_cool_down[ids] = 0; self.ready_steps[ids] = 0
        self.preparing[ids] = False; self.prepare_age[ids] = 0
        self.yaw_ready_steps[ids] = 0
        self.heading_active[ids] = False; self.heading_reference[ids] = 0
        self.heading_reward_active[ids] = False
        self.state['gate'][ids] = False
        self.flat_scale[ids] = 1
        self.real_ascent[ids] = False; self.progress[ids] = 0
        self.body_failure_event[ids] = False
        self.attitude_reward_active[ids] = False
        self.knee_delta[ids] = 0; self.retraction[ids] = 0
        self.subset[ids] = self.cfg.warmup_iterations == 0 and self.cfg.play
        if self._env.common_step_counter >= self._warmup_end_step and not self.cfg.play:
            elapsed = (self._env.common_step_counter-self._warmup_end_step)/self.steps_per_iteration
            p = jump_episode_probability(elapsed, ramp=self._jump_ramp_iterations)
            # Sample once per episode, including fallen origins. Recovery origins must
            # sustain the full readiness hold below before any automatic Jump request.
            self.subset[ids] = torch.rand(len(ids), device=self.device) < p

    def request(self, ids=None, distance=0., random_fields=False):
        """Accept a straight jump, braking yaw before starting the takeoff reference."""
        ids = self._env_ids if ids is None else ids
        self.counts['requests'] += len(ids)
        self._requested[ids] = True
        disabled = self._env.common_step_counter < self._warmup_end_step
        reasons = torch.zeros(len(ids), dtype=torch.long, device=self.device)
        reasons[self.ready_steps[ids] < round(THRESHOLDS['readiness_s']/self._env.step_dt)] = 1
        reasons[self.jump_cool_down[ids] > 0] = 2
        reasons[self.state['tracking'][ids]] = 3
        if disabled:
            reasons[:] = 4
        for code, key in ((1, 'rejected_not_ready'), (2, 'rejected_cooldown'), (3, 'rejected_busy'), (4, 'rejected_disabled')):
            self.counts[key] += (reasons == code).sum()
        accepted = ids[reasons == 0]
        if len(accepted):
            for key, value in self.state.items():
                value[accepted] = -1 if key in ('takeoff_s', 'land_s', 'stable_s', 'tuck_s') else 0
            self.state['tracking'][accepted] = True
            self.preparing[accepted] = True; self.prepare_age[accepted] = 0
            # Zero yaw in both the policy input and reward target for a straight jump.
            twist = self._env.command_manager.get_term('twist').vel_command_b
            self.yaw_ready_steps[accepted[twist[accepted, 2].abs() >= 1e-6]] = 0
            twist[accepted, 2] = 0.
            self.heading_reference[accepted] = self._env.scene['robot'].data.heading_w[accepted]
            self.heading_active[accepted] = True
            # Capture the base height at request acceptance.
            self.state['start_height'][accepted] = self.sample['height'][accepted]
            self.state['request_twist'][accepted] = twist[accepted]
            self.state['direction'][accepted] = torch.where(twist[accepted, 0] >= 0, 1., -1.)
            self.state['request_xy'][accepted] = self.sample['position_xy'][accepted]
            forward = torch.zeros(len(accepted), 3, device=self.device); forward[:, 0] = 1.
            self.state['forward_xy'][accepted] = quat_apply(yaw_quat(self.sample['quat'][accepted]), forward)[:, :2]
            if random_fields:
                self.target_height[accepted] = torch.empty(len(accepted), device=self.device).uniform_(*self.cfg.target_height_range)
                self.distance[accepted] = torch.empty(len(accepted), device=self.device).uniform_(*self.cfg.distance_range)
            else:
                self.target_height[accepted] = self.cfg.reference_height; self.distance[accepted] = distance
            self.counts['accepted'] += len(accepted)
            # An already straight, settled command needs no additional delay.
            self._begin_jump(accepted[self.yaw_ready_steps[accepted] >=
                                      round(self.cfg.yaw_settle_s / self._env.step_dt)])
            self._suppress_random_push(accepted)
        return reasons

    def _begin_jump(self, ids):
        if not len(ids):
            return
        self.preparing[ids] = False
        self.state['gate'][ids] = True; self.state['attitude_active'][ids] = True
        self.state['start_height'][ids] = self.sample['height'][ids]
        self.state['request_xy'][ids] = self.sample['position_xy'][ids]
        self.jump_flag[ids] = True; self.jump_count[ids] = 0

    def _suppress_random_push(self, ids):
        """Clear active random-push forces and torques without resetting pulse timing."""
        term = self._random_push
        if term is None:
            return
        ids = ids[term._active[ids]]
        if len(ids):
            zero = torch.zeros(len(ids), term._num_bodies, 3, device=self.device)
            term._asset.write_external_wrench_to_sim(
                zero, zero, env_ids=ids, body_ids=term._body_ids)

    def compute(self, dt, env_ids=None):
        if isinstance(dt, torch.Tensor):
            assert env_ids is None
            env_ids = self._env_ids[dt > 0]
        elif dt <= 0:
            return
        self._update_command(env_ids)
        # Apply push suppression after command updates, including newly accepted jumps.
        if self._random_push is not None:
            self._suppress_random_push((self.preparing | self.jump_flag).nonzero().flatten())

    def _update_command(self, env_ids=None):
        ids = self._env_ids if env_ids is None else env_ids
        eligible_cooldown = self.jump_cool_down[ids] == 0
        self.jump_cool_down[ids] = (self.jump_cool_down[ids]-1).clamp_min(0)
        self.jump_count[ids] += self.jump_flag[ids].long()
        stop = ids[self.jump_count[ids] >= len(self._reference)]
        self._clear_command(stop); self.jump_cool_down[stop] = self.cfg.cool_down_steps
        if self.cfg.play:
            self._update_gui(ids)
        if self._env.common_step_counter < self._warmup_end_step:
            return
        settled = self.preparing[ids] & (self.yaw_ready_steps[ids] >=
                                         round(self.cfg.yaw_settle_s / self._env.step_dt))
        self._begin_jump(ids[settled])
        if self.cfg.probability > 0:
            kinds = getattr(self._env, '_curriculum_kind', torch.zeros_like(self.jump_count))
            # jump_ready includes posture, leg shape, wheel load and absence of body
            # contact; it also permits stable locomotion at a nonzero velocity command.
            required = torch.where(kinds[ids] == 0,
                                   round(THRESHOLDS['readiness_s']/self._env.step_dt),
                                   round(THRESHOLDS['stable_hold_s']/self._env.step_dt))
            eligible = ids[self.subset[ids] & ~self.state['tracking'][ids] & eligible_cooldown &
                           (self.ready_steps[ids] >= required)]
            if len(eligible):
                chosen = eligible[torch.rand(len(eligible), device=self.device) < self.cfg.probability]
                if len(chosen):
                    self.request(chosen, random_fields=True)

    def _update_attitude_window(self, measurement):
        """Charge the landing/failure frame, then release grounded recovery.

        An airborne tilt failure must not disable attitude costs. The separate
        window outlasts the reference and gate, but ends at grounded failure,
        stable landing, or the existing two-second protection deadline.
        """
        s = self.state
        self.attitude_reward_active.copy_(s['tracking'] & s['attitude_active'])
        grounded = measurement['body'] | measurement['contacts'].any(-1)
        finished = ((s['failed'] & grounded) | (s['stable_s'] >= 0) |
                    (s['age'] >= THRESHOLDS['protection_s']))
        s['attitude_active'] &= s['tracking'] & ~finished

    def observe(self):
        """Update jump measurements once per control step before reward computation."""
        env = self._env
        if self._last_observed_step == env.common_step_counter:
            return
        self._last_observed_step = env.common_step_counter
        self.body_failure_event.zero_()
        self.attitude_reward_active.zero_()
        self.heading_reward_active.zero_()
        if env.common_step_counter < self._warmup_end_step:
            # Collect base measurements while Jump is disabled.
            self.sample = {'height': mdp.base_height(env).squeeze(-1),
                           'vz': env.scene['robot'].data.root_link_lin_vel_w[:, 2],
                           'contacts': env.scene['wheel_ground_contact'].data.force.norm(dim=-1) > .1}
            return
        m = self.sample = self.measure(env)
        self._snapshot_command.copy_(self.command)
        ready = m['jump_ready']
        self.ready_steps = torch.where(ready, self.ready_steps+1, 0)
        wz = env.scene['robot'].data.root_link_ang_vel_w[:, 2]
        command_wz = env.command_manager.get_command('twist')[:, 2]
        straight = ready & (wz.abs() <= self.cfg.takeoff_yaw_rate) & (command_wz.abs() < 1e-6)
        self.yaw_ready_steps = torch.where(straight, self.yaw_ready_steps+1, 0)
        self.heading_reward_active.copy_(self.heading_active)
        # Keep the constraint in flight and after ordinary landing. Release only
        # genuinely fallen recovery, not a short reference end or a bounced wheel.
        fallen = m['body'] & ((m['tilt'] >= 60.) | (m['height'] < .065))
        self.heading_active &= ~fallen
        self.prepare_age += self.preparing * env.step_dt
        aborted = self.preparing & ((self.prepare_age >= self.cfg.prepare_timeout_s) | m['body'] | fallen)
        if len(ids := aborted.nonzero().flatten()):
            self.state['failed'][ids] = True
            self._finish(ids, closure='preparation_failed')
            self.heading_active[ids] = False
            self._clear_command(ids)
            self.jump_cool_down[ids] = self.cfg.cool_down_steps
        s = self.state; tracking = s['tracking']; dt = env.step_dt
        s['age'] += (tracking & ~self.preparing)*dt
        displacement = ((m['position_xy'] - s['request_xy']) * s['forward_xy']).sum(-1)
        s['displacement'] = torch.where(tracking & (s['land_s'] < 0), displacement, s['displacement'])
        forward_speed = (m['velocity_xy'] * s['forward_xy']).sum(-1)
        s['reverse_distance'] += (-s['direction'] * forward_speed).clamp_min(0) * self.direction_active * dt
        # Clear provisional flight measurements when an unconfirmed airborne interval ends.
        interrupted = tracking & ~s['lifted'] & ~m['air'] & (s['air_run'] > 0)
        for key in ('had_ascent', 'apex_seen', 'tucked', 'ascent_frames', 'tuck_frames',
                    'tuck_run', 'tuck_max', 'knee_max', 'retraction_max', 'apex_tucked',
                    'flight_peak_height', 'apex_clearance', 'max_clearance'):
            s[key][interrupted] = 0
        s['takeoff_s'][interrupted] = -1; s['tuck_s'][interrupted] = -1
        active_air = tracking & ~self.preparing & m['air'] & (s['land_s'] < 0)
        first = active_air & (s['air_run'] == 0) & ~s['lifted'] & ~s['apex_seen']
        s['knee0'][first] = m['knee'][first]; s['length0'][first] = m['length'][first]
        s['takeoff_s'][first] = s['age'][first]
        s['takeoff_displacement'][first] = s['displacement'][first]
        s['flight_displacement'] = torch.where(tracking & (s['land_s'] < 0) & (s['takeoff_s'] >= 0),
                                              s['displacement'] - s['takeoff_displacement'], s['flight_displacement'])
        s['air_run'] = torch.where(active_air, s['air_run']+1, 0)
        s['air_max'] = torch.maximum(s['air_max'], s['air_run'])
        s['lifted'] |= active_air & (s['air_run'] >= round(THRESHOLDS['air_hold_s']/dt))
        self.knee_delta = torch.where((s['takeoff_s'] >= 0)[:, None], m['knee']-s['knee0'], 0.)
        self.retraction = torch.where((s['takeoff_s'] >= 0)[:, None], s['length0']-m['length'], 0.)
        self.real_ascent = active_air & (m['vz'] > THRESHOLDS['ascent_vz_m_s']) & ~s['apex_seen'] & s['gate'] & ~s['failed']
        s['had_ascent'] |= self.real_ascent
        s['apex_seen'] |= tracking & s['had_ascent'] & (m['vz'] <= THRESHOLDS['ascent_vz_m_s'])
        strong = strong_tuck_progress(m['knee'], m['length'], s['knee0'], s['length0'], m['margin'])
        apex_band = (active_air & s['had_ascent'] & (m['vz'].abs() <= THRESHOLDS['ascent_vz_m_s']) &
                     s['gate'] & ~s['failed'])
        self.progress = strong * (self.real_ascent | apex_band)
        peak = active_air & s['had_ascent'] & (m['height'] > s['flight_peak_height'])
        s['flight_peak_height'][peak] = m['height'][peak]
        s['apex_tucked'][peak] = strong[peak] >= 1
        s['apex_clearance'][peak] = m['clearance'][peak].amin(-1)
        s['max_clearance'] = torch.maximum(s['max_clearance'], torch.where(active_air, m['clearance'].amin(-1), 0))
        tuck = self.real_ascent & (self.progress >= 1)
        s['tuck_run'] = torch.where(tuck, s['tuck_run']+1, 0)
        s['tuck_max'] = torch.maximum(s['tuck_max'], s['tuck_run'])
        new_tuck = tuck & (s['tuck_s'] < 0) & (s['tuck_run'] >= round(THRESHOLDS['tuck_hold_s']/dt))
        s['tuck_s'][new_tuck] = s['age'][new_tuck]-(round(THRESHOLDS['tuck_hold_s']/dt)-1)*dt
        s['tucked'] |= new_tuck
        s['ascent_frames'] += self.real_ascent; s['tuck_frames'] += tuck
        for key, value in (('knee_max', self.knee_delta), ('retraction_max', self.retraction)):
            s[key] = torch.maximum(s[key], torch.where(self.real_ascent[:, None], value, 0))
        s['apex'] = torch.maximum(s['apex'], torch.where(tracking, m['height']-s['start_height'], 0))
        # Cache the first body contact per request before updating the contact flag.
        self.body_failure_event.copy_(tracking & m['body'] & ~s['body_hit'])
        m['body_failure_event'] = self.body_failure_event
        s['body_hit'] |= tracking & m['body']
        s['max_tilt'] = torch.maximum(s['max_tilt'], torch.where(tracking & (s['land_s'] < 0), m['tilt'], 0))
        s['impact'] = torch.maximum(s['impact'], torch.where(tracking, m['impact'], 0))
        s['bad_flight'] |= tracking & (s['land_s'] < 0) & (s['takeoff_s'] >= 0) & (m['margin'] <= .02)
        landing = tracking & s['lifted'] & m['contacts'].any(-1) & (s['land_s'] < 0)
        s['land_s'][landing] = s['age'][landing]
        landed = s['land_s'] >= 0
        s['ready_hold'] = torch.where(tracking & landed & ready, s['ready_hold']+1, 0)
        stable = tracking & landed & (s['ready_hold'] >= round(THRESHOLDS['stable_hold_s']/dt)) & (s['stable_s'] < 0)
        s['stable_s'][stable] = s['age'][stable]-(round(THRESHOLDS['stable_hold_s']/dt)-1)*dt
        s['post_bad'] |= tracking & (s['stable_s'] >= 0) & ~ready
        failure = tracking & (m['body'] | (m['tilt'] > 60) |
                  ((s['age'] > THRESHOLDS['takeoff_deadline_s']) & ~s['lifted']) |
                  ((s['age'] >= THRESHOLDS['protection_s']) & (~landed | (self.ready_steps < 20))))
        s['failed'] |= failure
        s['gate'] &= ~failure & (s['age'] < THRESHOLDS['protection_s'])
        self._update_attitude_window(m)
        # Clear measured ascent and tuck progress when the jump protection gate closes.
        self.real_ascent &= s['gate']
        self.progress *= s['gate']
        ramp = ((s['age']-s['land_s'])/THRESHOLDS['landing_ramp_s']).clamp(0, 1)
        self.flat_scale = torch.where(s['gate'], torch.where(landed, ramp, 0.), 1.)
        self.sample_phase = torch.where(s['gate'], torch.where(landed, 5,
                            torch.where(self.jump_flag, self.jump_stage+1, 4)), 0)
        # Finish after stable landing and follow-up, or the nine-second timeout.
        finish = tracking & (((s['stable_s'] >= 0) & (s['age'] >= s['stable_s']+1+THRESHOLDS['followup_s'])) |
                             (s['age'] >= 9.))
        terminal = getattr(env, 'reset_buf', None)
        if terminal is not None:
            finish &= ~terminal  # after_reward records the actual termination reason
        if len(ids := finish.nonzero().flatten()):
            self._finish(ids)
        if (not self.cfg.play):
            # End tracking for a retry once cooldown and grounded readiness are satisfied.
            # Retries are recorded as truncated observations.
            retry = s['tracking'] & ~self.jump_flag & (self.jump_cool_down == 0) & (self.ready_steps >= 20)
            retry &= s['failed'] | (landed & (s['ready_hold'] >= round(1/dt)))
            if terminal is not None:
                retry &= ~terminal
            if len(ids := retry.nonzero().flatten()):
                self._finish(ids, closure='training_retrigger')

    def _finish(self, ids, terminated=False, closure='full_observation'):
        s = self.state
        direction_passed = (s['lifted'] & s['had_ascent'] & (s['land_s'] >= 0) &
                            (s['direction'] * s['displacement'] >= -DIRECTION_TOLERANCE_M) &
                            (s['direction'] * s['flight_displacement'] >= -DIRECTION_TOLERANCE_M))
        height_or_clearance = ((s['apex_clearance'] >= .08) & s['apex_tucked'])
        success = (s['lifted'] & s['had_ascent'] & s['apex_seen'] & s['tucked'] &
                   (s['takeoff_s'] <= THRESHOLDS['takeoff_deadline_s']) &
                   height_or_clearance &
                   (s['max_tilt'] < THRESHOLDS['flight_tilt_deg']) & ~s['body_hit'] & ~s['bad_flight'] &
                   ~s['failed'] & ~s['post_bad'] & direction_passed & (s['land_s'] >= 0) & (s['stable_s'] >= 0) &
                   (s['stable_s']-s['land_s'] <= THRESHOLDS['landing_stable_deadline_s']))
        normal_landing = (~s['body_hit'] & (s['land_s'] >= 0) & (s['stable_s'] >= 0) &
                          (s['stable_s']-s['land_s'] <= THRESHOLDS['landing_stable_deadline_s']))
        short_success = success.clone()
        if terminated or closure == 'training_retrigger':
            success[ids] = False
        if terminated:
            success[ids] = False; self.counts['terminated'] += len(ids)
        fields = ['lifted', 'had_ascent', 'tucked', 'apex', 'takeoff_s', 'air_max', 'tuck_max',
                  'tuck_s', 'ascent_frames', 'tuck_frames', 'land_s', 'stable_s', 'body_hit',
                  'max_tilt', 'impact', 'failed', 'post_bad', 'age']
        fields += ['apex_tucked', 'apex_clearance', 'max_clearance']
        fields += ['direction', 'displacement', 'flight_displacement', 'reverse_distance']
        values = torch.stack([s[k][ids].float() for k in fields]+[success[ids].float()], -1).cpu().numpy()
        heading_errors = torch.rad2deg(self.heading_error[ids].abs()).cpu().tolist()
        preparation_times = self.prepare_age[ids].cpu().tolist()
        request_twists = s['request_twist'][ids].cpu().tolist()
        for i, row in enumerate(values):
            idx = int(ids[i])
            time_limit = terminated and bool(getattr(self._env, 'reset_time_outs', torch.zeros_like(s['tracking']))[idx])
            behavior_end = terminated and bool(getattr(self._env, 'reset_terminated', torch.zeros_like(s['tracking']))[idx])
            end_reason = ('behavior_termination' if behavior_end else 'time_limit' if time_limit else 'reset_unknown') if terminated else closure
            self.events.append(dict(zip(fields+['success'], row.tolist())) | {
                'env_id': idx, 'control_step': self._env.common_step_counter,
                'knee_max': s['knee_max'][ids[i]].cpu().tolist(),
                'retraction_max': s['retraction_max'][ids[i]].cpu().tolist(), 'terminated': terminated,
                'closure': end_reason, 'time_limit': time_limit, 'behavior_termination': behavior_end,
                'observation_truncated': (time_limit and not behavior_end) or closure == 'training_retrigger',
                'normal_landing': bool(normal_landing[idx]),
                'request_vx': request_twists[i][0],
                'request_wz': request_twists[i][2],
                'preparation_s': preparation_times[i],
                'final_heading_error_deg': heading_errors[i],
                'direction_passed': bool(direction_passed[idx]),
                'training_landing_success': bool(short_success[idx]) and not terminated})
        s['tracking'][ids] = False; s['gate'][ids] = False
        s['attitude_active'][ids] = False
        self.preparing[ids] = False
        if terminated:
            self.heading_active[ids] = False

    def after_reward(self, dt):
        env = self._env
        if env.common_step_counter < self._warmup_end_step:
            return
        # Record active jumps before the episode resets.
        done = env.reset_buf & self.state['tracking']
        if len(ids := done.nonzero().flatten()):
            self.state['failed'][ids] |= env.reset_terminated[ids]
            self._finish(ids, terminated=True)
        if self.phase_rewards is None:
            self.phase_rewards = torch.zeros(6, len(env.reward_manager.active_terms)+1, device=self.device)
        for phase in range(6):
            mask = self.sample_phase == phase
            self.phase_rewards[phase, :-1] += (env.reward_manager._step_reward * mask[:, None]).sum(0)*dt
            self.phase_rewards[phase, -1] += mask.sum()
        self._requested[:] = False

    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
        self._gui_requests = deque()
        self._gui_env_idx = get_env_idx
        self._gui_on_change = on_change
        with server.gui.add_folder(f'Jump ({self.cfg.reference_height:g}m fixed reference)'):
            button = server.gui.add_button('Request one jump')
            cooldown_s = self.cfg.cool_down_steps * self._env.step_dt
            status = server.gui.add_text('Last request',
                initial_value=f'Ready required; yaw settles before takeoff, {cooldown_s:g}s cooldown', disabled=True)
            self._gui_state = server.gui.add_text('Measured state', initial_value='Waiting for control sample', disabled=True)
        @button.on_click
        def _(_event):
            # Viser dispatches clicks on a worker thread. Only queue Python data;
            # simulation tensors are updated at the next active control boundary.
            self._gui_requests.append((int(get_env_idx()), status))
            status.value = 'Queued for next active control step'

    def _update_gui(self, ids):
        pending = getattr(self, '_gui_requests', None)
        if pending is None:
            return
        for _ in range(len(pending)):
            idx, status = pending.popleft()
            if idx not in ids:
                pending.append((idx, status))
                continue
            code = int(self.request(self._env_ids[idx:idx+1])[0])
            status.value = ('accepted; braking yaw before takeoff', 'not ready', 'cooldown', 'busy', 'disabled')[code]
            if self._gui_on_change:
                self._gui_on_change()
        if self.sample is not None and self._env.common_step_counter % 10 == 0:
            idx = self._gui_env_idx()
            states = ' '.join(f'{key}={bool(self.state[key][idx])}' for key in ('tracking','lifted','tucked','failed'))
            self._gui_state.value = (f'ready={int(self.ready_steps[idx]) >= 20} {states} '
                                     f'h={float(self.sample["height"][idx]):.3f}m vz={float(self.sample["vz"][idx]):.2f}m/s')


@dataclass(kw_only=True)
class BeniJumpCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1e8, 1e8)
    probability: float = .05
    warmup_iterations: int = 3000
    """Earliest full recovery check for entering Jump; all required classes must pass."""
    ramp_iterations: float = 1000.
    recovery_gate: RecoveryGateCfg = field(default_factory=RecoveryGateCfg)
    best_checkpoint: JumpEvaluationCfg = field(default_factory=JumpEvaluationCfg)
    cool_down_steps: int = 100
    play: bool = False
    distance_range: tuple[float, float] = (-.2, .2)
    takeoff_yaw_rate: float = .3
    """Maximum absolute world-z angular velocity (rad/s) before takeoff."""
    yaw_settle_s: float = .2
    prepare_timeout_s: float = 2.

    @property
    def reference_height(self):
        """Fixed training/play reference, not a trajectory selection."""
        return JUMP_REFERENCE_HEIGHT

    @property
    def target_height_range(self):
        return (JUMP_REFERENCE_HEIGHT, JUMP_REFERENCE_HEIGHT)

    @property
    def reference_path(self):
        return JUMP_REFERENCE_PATH

    def build(self, env):
        return BeniJumpCommand(self, env)


def install_observer(env, env_ids=None):
    """Attach jump measurement and logging to each reward computation."""
    command = env.command_manager.get_term('jump')
    if ('push_robot' in env.cfg.events):
        from mjlab.envs.mdp.events import apply_body_impulse
        term = env.event_manager.get_term_cfg('push_robot').func
        if not isinstance(term, apply_body_impulse):
            raise ValueError('Jump push suppression requires native apply_body_impulse')
        command._random_push = term
    compute = env.reward_manager.compute
    def observed_compute(dt):
        command.observe()
        value = compute(dt)
        command.after_reward(dt)
        return value
    env.reward_manager.compute = observed_compute


class flat_reward:
    def __init__(self, cfg, env):
        self.reward = cfg.params['reward_func']
        if isinstance(self.reward, type):
            self.reward = self.reward(cfg=cfg, env=env)

    def __call__(self, env, reward_func, **params):
        command = env.command_manager.get_term('jump')
        return self.reward(env, **params)*command.flat_scale


def contact_penalty(env, reward_func, jump_scale=5., **params):
    """Increase contact costs through the two-second launch/landing window.

    Failure closes the protection gate, but must not remove the contact cost.
    Preparation, ordinary recovery and expired requests retain the base reward.
    """
    value = reward_func(env, **params)
    command = env.command_manager.get_term('jump')
    s = command.state
    active = s['tracking'] & ~command.preparing & (s['age'] <= THRESHOLDS['protection_s'])
    return torch.where(active, value * jump_scale, value)


def track_yaw_velocity_exp(env, command_name, std):
    """Track zero yaw rate during straight jumps; preserve ordinary recovery rewards."""
    command = env.command_manager.get_term('jump')
    active = command.attitude_reward_active | command.heading_reward_active
    support = torch.where(active, 1., mdp.standing_support(env))
    target = torch.where(active, 0., env.command_manager.get_command(command_name)[:, 2])
    error = (target - env.scene['robot'].data.root_link_ang_vel_w[:, 2]).square()
    return torch.exp(-error / std**2) * support


def heading_error_huber(env, angle_scale_deg=10.):
    """Penalize accumulated heading error without saturating at large offsets."""
    if not math.isfinite(angle_scale_deg) or angle_scale_deg <= 0:
        raise ValueError('Heading error scale must be positive and finite')
    command = env.command_manager.get_term('jump')
    error = command.heading_error.abs() / math.radians(angle_scale_deg)
    loss = torch.where(error <= 1., .5 * error.square(), error - .5)
    return loss * command.heading_reward_active


def large_tilt_l2(env, start_angle_deg=30., full_angle_deg=60.):
    """Apply the additional tilt cost only during an accepted Jump."""
    active = env.command_manager.get_term('jump').attitude_reward_active
    return mdp.large_tilt_l2(env, start_angle_deg, full_angle_deg) * active


def ordinary_roll_pitch_velocity_l2(env, clip=20.):
    """Apply the ordinary roll/pitch rate cost outside accepted jumps."""
    active = env.command_manager.get_term('jump').attitude_reward_active
    return mdp.roll_pitch_velocity_l2(env, clip) * ~active


def attitude_angles(env):
    """Roll and pitch in radians, using the existing projected-gravity signal."""
    gravity = env.scene['robot'].data.projected_gravity_b
    return torch.stack((torch.atan2(-gravity[:, 1], -gravity[:, 2]),
                        torch.asin(gravity[:, 0].clamp(-1, 1))), dim=-1)


def _huber(value):
    value = value.abs()
    return torch.where(value <= 1., .5 * value.square(), value - .5)


def attitude_angle_huber(env, axis, deadband_deg, scale_deg):
    """Zero-target Jump angle cost with a tolerance and no large-angle cap."""
    if axis not in (0, 1) or not math.isfinite(deadband_deg) or deadband_deg < 0:
        raise ValueError('Jump attitude requires axis 0/1 and a nonnegative finite deadband')
    if not math.isfinite(scale_deg) or scale_deg <= 0:
        raise ValueError('Jump attitude angle scale must be positive and finite')
    error = (attitude_angles(env)[:, axis].abs()-math.radians(deadband_deg)).clamp_min(0)
    active = env.command_manager.get_term('jump').attitude_reward_active
    return _huber(error / math.radians(scale_deg)) * active


def attitude_rate_huber(env, axis, scale_rad_s):
    """Each Jump body-axis rate grows independently, even if the other is large."""
    if axis not in (0, 1) or not math.isfinite(scale_rad_s) or scale_rad_s <= 0:
        raise ValueError('Jump attitude requires axis 0/1 and a positive finite rate scale')
    rate = env.scene['robot'].data.root_link_ang_vel_b[:, axis]
    active = env.command_manager.get_term('jump').attitude_reward_active
    return _huber(rate / scale_rad_s) * active


def jump_height_tracking_l1(env, nominal_height=H_REF):
    j = env.command_manager.get_term('jump')
    # Subtract the reference error of a stationary robot at nominal height.
    baseline = j.jump_ref_height.abs()
    value = (j.jump_ref_height+nominal_height-j.sample['height']).abs()-baseline
    value = torch.where(j.state['failed'], value.clamp_min(0), value)
    return value*j.ascent


def jump_vel_tracking_l2(env):
    """Squared reference-vz error relative to the stationary baseline."""
    j = env.command_manager.get_term('jump')
    value = (j.sample['vz']-j.jump_ref_vel).square()-j.jump_ref_vel.square()
    value = torch.where(j.state['failed'], value.clamp_min(0), value)
    return value*j.ascent


def jump_position_limits(env, asset_cfg, target_margin=1., jump_scale=100.):
    """Scale joint-position penalties during jump protection and landing.

    The penalty returns to its ordinary weight as flat_scale reaches one.
    """
    value = position_limits(env, asset_cfg, target_margin=target_margin)
    ordinary_scale = env.command_manager.get_term('jump').flat_scale
    return value * (1 + (jump_scale-1)*(1-ordinary_scale))


def jump_knee_regulation(env):
    return env.command_manager.get_term('jump').progress


def jump_direction_progress(env, distance_scale=.05):
    """Bounded progress in the requested direction, earned only during confirmed flight."""
    if not math.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError('Jump direction distance_scale must be finite and positive')
    j = env.command_manager.get_term('jump')
    if env.common_step_counter < j._warmup_end_step:
        return torch.zeros_like(j.target_height)
    s = j.state
    active = (j.direction_active & s['lifted'] & s['had_ascent'] & j.sample['air'] &
              ~s['failed'] & ~s['body_hit'])
    # Ground travel cannot compensate for flying in the wrong direction.
    progress = torch.minimum(s['direction'] * s['displacement'],
                             s['direction'] * s['flight_displacement'])
    return (progress / distance_scale).clamp(-1, 1) * active


def jump_reverse_velocity_l1(env):
    """Penalize reverse horizontal speed during propulsion and flight, even without wheel support."""
    j = env.command_manager.get_term('jump')
    if env.common_step_counter < j._warmup_end_step:
        return torch.zeros_like(j.target_height)
    forward_speed = (j.sample['velocity_xy'] * j.state['forward_xy']).sum(-1)
    return (-j.state['direction'] * forward_speed).clamp_min(0) * j.direction_active


def jump_takeoff_progress(env):
    """Reward wheel unloading, ground clearance and upward speed during reference ascent."""
    j = env.command_manager.get_term('jump')
    if env.common_step_counter < j._warmup_end_step:
        return torch.zeros_like(j.target_height)
    m = j.sample
    # Use the larger wheel force so progress requires unloading both wheels.
    unloaded = (1-m['impact']/(.5*j.measure.weight)).clamp(0, 1)
    clearance = (m['clearance'].amin(-1)/THRESHOLDS['clearance_m']).clamp(0, 1)
    speed = (m['vz']/j._reference[:, 10].amax()).clamp(0, 1)
    valid = (j.ascent & j.state['gate'] & ~j.state['failed'] &
             (j.state['land_s'] < 0) & ~m['body'] &
             (m['tilt'] < THRESHOLDS['flight_tilt_deg']) & (m['margin'] > .02) &
             (m['vz'] > THRESHOLDS['ascent_vz_m_s']))
    return (.25*unloaded+.75*clearance)*speed*valid


def single_foot_contact_penalty(env):
    j = env.command_manager.get_term('jump')
    return -((j.sample['contacts'].sum(-1) == 1) & j.state['gate'] &
             j.state['lifted'] & (j.state['land_s'] >= 0)).float()


def jump_body_failure(env):
    """Mark the first body-ground contact of each tracked jump request."""
    return env.command_manager.get_term('jump').body_failure_event.float()
