# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Evaluate jump checkpoints with deterministic standing and moving trials."""

from copy import deepcopy
from dataclasses import dataclass
import math
import random

import numpy as np
import torch

# Stored in checkpoints to detect incompatible evaluation criteria on resume.
JUMP_EVALUATION_PROTOCOL = 'controlled_directional_landing_with_heading_and_native_termination_and_recovery'


@dataclass
class JumpEvaluationCfg:
    enabled: bool = True
    start_updates: int = 8000
    """Evaluate saved checkpoints after this many completed PPO updates."""
    trials_per_speed: int = 8
    speeds: tuple[float, ...] = (-2., -1., -.5, 0., .5, 1., 1.5, 2.)
    seed: int = 49
    heading_tolerance_deg: float = 10.
    """Maximum heading error at first wheel contact and throughout the final one-second hold."""
    peak_heading_tolerance_deg: float = 20.
    """Maximum heading error from the request through the end of the trial."""
    min_median_clearance_m: float = .15
    """Best checkpoints must exceed this median wheel clearance across all trials."""

    def validate(self):
        if self.start_updates < 0 or self.trials_per_speed < 1 or self.seed < 0:
            raise ValueError('Invalid Jump checkpoint evaluation budget or seed')
        if not self.speeds or len(set(self.speeds)) != len(self.speeds) or not all(map(math.isfinite, self.speeds)):
            raise ValueError('Jump evaluation speeds must be distinct finite values')
        if not (math.isfinite(self.heading_tolerance_deg) and math.isfinite(self.peak_heading_tolerance_deg)
                and 0 < self.heading_tolerance_deg <= self.peak_heading_tolerance_deg < 180):
            raise ValueError('Jump heading tolerances require 0 < landing/final <= peak < 180 degrees')
        if not math.isfinite(self.min_median_clearance_m) or self.min_median_clearance_m < 0:
            raise ValueError('Minimum median wheel clearance must be finite and nonnegative')


def heading_error_deg(heading, reference):
    """Shortest absolute heading error, including the +/-pi boundary."""
    difference = heading-reference
    return torch.rad2deg(torch.atan2(torch.sin(difference), torch.cos(difference)).abs())


def jump_heading_passed(landing, peak, final, cfg):
    """Require accurate heading at landing and finish, with bounded error throughout."""
    return (landing is not None and all(math.isfinite(value) for value in (landing, peak, final))
            and 0 <= landing <= cfg.heading_tolerance_deg
            and 0 <= final <= cfg.heading_tolerance_deg
            and max(landing, final) <= peak <= cfg.peak_heading_tolerance_deg)


def summarize_jump_trials(trials, cfg=None):
    """Qualify by median height, then rank controlled landings that preserve heading."""
    cfg = cfg if cfg is not None else JumpEvaluationCfg()
    cfg.validate()
    if not trials:
        raise ValueError('Jump evaluation requires at least one trial')
    clearances = [row['clearance_m'] for row in trials]
    if not all(map(math.isfinite, clearances)):
        raise ValueError('Nonfinite Jump wheel clearance')
    # Include failed/unaccepted trials too; do not condition height on successful landings.
    median_clearance = float(np.median(clearances))
    count = len(trials)
    mean = lambda key: sum(row[key] for row in trials) / count
    metrics = {key: mean(key) for key in ('landed', 'took_off', 'strict_success',
                                        'joint_limit_violation_rad', 'peak_tilt_deg')}
    # Among height-qualified checkpoints, capped clearance only breaks landing ties.
    clearance = sum(min(.08, max(0., row['clearance_m'])) * row['landed'] for row in trials) / count
    directional_landing = sum(row['landed'] and row['direction_passed'] for row in trials) / count
    heading_landing = sum(row['landed'] and row['direction_passed'] and row['heading_passed']
                          for row in trials) / count
    peak_heading_error = mean('peak_heading_error_deg')
    final_heading_error = mean('final_heading_error_deg')
    score = [heading_landing, directional_landing, metrics['landed'], metrics['took_off'],
             metrics['strict_success'], -metrics['joint_limit_violation_rad'],
             -peak_heading_error, clearance, -metrics['peak_tilt_deg']]
    if not all(map(math.isfinite, [*score, final_heading_error])):
        raise ValueError('Nonfinite Jump checkpoint evaluation')
    return dict(protocol=JUMP_EVALUATION_PROTOCOL, score=score,
                median_clearance_m=median_clearance,
                best_eligible=median_clearance > cfg.min_median_clearance_m,
                heading_directional_landing_rate=heading_landing,
                heading_pass_rate=mean('heading_passed'),
                mean_peak_heading_error_deg=peak_heading_error,
                mean_final_heading_error_deg=final_heading_error,
                directional_landing_rate=directional_landing,
                landing_rate=metrics['landed'], takeoff_rate=metrics['took_off'],
                strict_success_rate=metrics['strict_success'],
                mean_joint_limit_violation_rad=metrics['joint_limit_violation_rad'],
                mean_capped_clearance_m=clearance, mean_peak_tilt_deg=metrics['peak_tilt_deg'],
                trials=trials)


def summarize_landing_recovery(body, ready, lifted, terminated, dt, request_step):
    """Summarize recovery after post-jump body contact."""
    arrays = [value.cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
              for value in (body,ready,lifted,terminated)]
    body,ready,lifted,terminated = arrays
    terminated = np.maximum.accumulate(terminated,axis=0)
    hold_steps = max(1,round(1./dt)); recovery_window_steps = round(6./dt)
    rows = []
    for trial in range(body.shape[1]):
        contacts = np.flatnonzero(body[request_step:,trial] & lifted[request_step:,trial])
        if not len(contacts): continue
        start = request_step+int(contacts[0])
        window_end = start + recovery_window_steps + hold_steps
        stable = ready[start:window_end,trial] & ~terminated[start:window_end,trial]
        matches = (np.flatnonzero(np.convolve(stable.astype(int),np.ones(hold_steps,dtype=int),'valid') == hold_steps)
                   if len(stable)>=hold_steps else np.array([],dtype=int))
        onset = float(matches[0]*dt) if len(matches) else None
        rows.append(dict(trial=trial,body_landing_s=(start+1)*dt,recovered=onset is not None,
                         recovery_s=onset,window_complete=len(stable)==recovery_window_steps+hold_steps,
                         native_failure=bool(terminated[-1,trial]),
                         ready_at_end=bool(ready[-hold_steps:,trial].all())))
    count = len(rows)
    result = dict(body_landings=count,recovered=sum(r['recovered'] for r in rows),trials=rows)
    for seconds in (1,3,6):
        result[f'recovered_by_{seconds}s_rate'] = (sum(r['recovery_s'] is not None and r['recovery_s']<=seconds+1e-6
                                                     for r in rows)/count if count else None)
    times = [r['recovery_s'] for r in rows if r['recovered']]
    result['recovery_time_p90_s'] = float(np.quantile(times,.9)) if times else None
    return result


@torch.inference_mode()
def evaluate_jump_checkpoint(env_cfg, actor, cfg, device):
    """Run one jump per trial in a separate simulator, preserving training RNG and actor state."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import EventTermCfg, TerminationManager
    from .jump import attitude_angles

    cfg.validate()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    env = None
    with torch.random.fork_rng(devices=[torch.device(device)] if torch.device(device).type == 'cuda' else []):
        try:
            eval_env_cfg = deepcopy(env_cfg)
            eval_env_cfg.recorders.pop('recovery_sampling', None)
            eval_env_cfg.seed = cfg.seed
            eval_env_cfg.scene.num_envs = len(cfg.speeds) * cfg.trials_per_speed
            eval_env_cfg.auto_reset = False
            eval_env_cfg.terminations = {}; eval_env_cfg.curriculum = {}; eval_env_cfg.metrics = {}
            # Keep reward terms for diagnostics; checkpoint ranking uses measured motion.
            eval_env_cfg.events = {key: term for key, term in eval_env_cfg.events.items()
                                   if key in ('reset_base', 'reset_joints', 'jump_observer')}
            eval_env_cfg.events['reset_base'].params.update(fallen_probability=0., pose_range={})
            eval_env_cfg.events['reset_joints'].params.update(position_range=(-.025, .025), velocity_range=(0., 0.))
            for group in eval_env_cfg.observations.values():
                group.enable_corruption = False
            eval_env_cfg.commands['jump'].play = True
            eval_env_cfg.commands['jump'].warmup_iterations = 0
            eval_env_cfg.commands['jump'].probability = 0.
            twist = eval_env_cfg.commands['twist']
            twist.resampling_time_range = (1e9, 1e9)
            twist.rel_standing_envs = 0.
            twist.heading_command = False
            twist.ranges.heading = None
            twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0., 0.)
            twist.pitch_range = twist.roll_range = (0., 0.)
            twist.height_range = (.11, .11)
            eval_env_cfg.commands['gimbal'].yaw_range = (0., 0.)
            eval_env_cfg.commands['gimbal'].enable_probability = 1.

            def record_native_termination(env,env_ids):
                native_failure.logical_or_(termination.compute())
            eval_env_cfg.events['record_native_termination'] = EventTermCfg(func=record_native_termination,mode='step')
            env = ManagerBasedRlEnv(eval_env_cfg, device=device)
            termination = TerminationManager(deepcopy(env_cfg.terminations),env)
            native_failure = torch.zeros(env.num_envs,dtype=torch.bool,device=device)
            env.reset(seed=cfg.seed)
            robot = env.scene['robot']
            command = env.command_manager.get_term('jump')
            speeds = torch.tensor(cfg.speeds, device=device).repeat_interleave(cfg.trials_per_speed)
            target = torch.stack((speeds, torch.zeros_like(speeds)), dim=-1)
            env.command_manager.get_term('twist').vel_command_b[:, :2] = target
            obs = env.get_observations()
            policy = actor.as_onnx().to(device).eval()
            hidden = torch.zeros(actor.rnn.rnn.num_layers, env.num_envs, actor.rnn.rnn.hidden_size, device=device)
            request_step, total_steps = round(6. / env.step_dt), round(16. / env.step_dt)
            hold_steps = round(1. / env.step_dt)
            ready_at_end = torch.ones(env.num_envs, dtype=torch.bool, device=device)
            limit_violation = torch.zeros(env.num_envs, device=device)
            final_velocity_error = torch.zeros_like(limit_violation)
            peak_heading_error = torch.zeros_like(limit_violation)
            landing_heading_error = torch.zeros_like(limit_violation)
            final_heading_error = torch.zeros_like(limit_violation)
            landing_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            recovery_trace = {key:[] for key in ('body','ready','lifted','terminated')}
            attitude_trace = {key:[] for key in ('angles', 'rates', 'phase', 'rewards')}
            for step in range(total_steps):
                if step == request_step:
                    request_velocity_error = (robot.data.root_link_lin_vel_b[:, :2] - target).norm(dim=-1)
                    # All trials command wz=0, so the requested heading remains fixed.
                    request_heading = robot.data.heading_w.clone()
                    request_reasons = command.request()
                    obs = env.get_observations()
                action, hidden, _ = policy(obs['actor'], hidden)
                obs, *_ = env.step(action)
                for key,value in dict(body=command.sample['body_current'],ready=command.sample['jump_ready'],
                                      lifted=command.state['lifted'],terminated=native_failure).items():
                    recovery_trace[key].append(value.clone())
                if step >= request_step:
                    active = command.attitude_reward_active
                    airborne = command.sample['air'] & (command.state['land_s'] < 0)
                    phase = torch.where(command.state['land_s'] >= 0, 4,
                        torch.where(airborne, torch.where(command.sample['vz'] > 0, 2, 3), 1))
                    for key, value in dict(angles=torch.rad2deg(attitude_angles(env)),
                            rates=robot.data.root_link_ang_vel_b[:, :2],
                            phase=phase * active,
                            rewards=env.reward_manager._step_reward).items():
                        attitude_trace[key].append(value.clone())
                    violation = (-command.sample['margin']).clamp_min(0) * (request_reasons == 0)
                    limit_violation = torch.maximum(limit_violation, violation)
                    heading_error = heading_error_deg(robot.data.heading_w, request_heading)
                    peak_heading_error = torch.maximum(peak_heading_error, heading_error)
                    first_landing = command.state['lifted'] & (command.state['land_s'] >= 0) & ~landing_seen
                    landing_heading_error[first_landing] = heading_error[first_landing]
                    landing_seen |= first_landing
                if step >= total_steps - hold_steps:
                    ready_at_end &= command.sample['jump_ready']
                    final_velocity_error += (robot.data.root_link_lin_vel_b[:, :2] - target).norm(dim=-1) / hold_steps
                    final_heading_error = torch.maximum(final_heading_error, heading_error)

            events = {row['env_id']: row for row in command.events}
            rows = []
            for index, reason in enumerate(request_reasons.cpu().tolist()):
                event = events.get(index, {})
                took_off = reason == 0 and bool(event.get('lifted')) and bool(event.get('had_ascent'))
                controlled = bool(request_velocity_error[index] <= .25 and final_velocity_error[index] <= .25)
                landed = (took_off and bool(event.get('normal_landing')) and not event.get('failed', True)
                          and not event.get('post_bad', True) and bool(ready_at_end[index]) and controlled
                          and not bool(native_failure[index]))
                landing_heading = float(landing_heading_error[index]) if bool(landing_seen[index]) else None
                peak_heading = float(peak_heading_error[index])
                final_heading = float(final_heading_error[index])
                heading_passed = took_off and jump_heading_passed(landing_heading, peak_heading, final_heading, cfg)
                rows.append(dict(speed_m_s=float(speeds[index]), trial=index % cfg.trials_per_speed,
                                 accepted=reason == 0, took_off=took_off, landed=landed,
                                 direction_passed=took_off and bool(event.get('direction_passed')),
                                 heading_passed=heading_passed,
                                 landing_heading_error_deg=landing_heading,
                                 peak_heading_error_deg=peak_heading,
                                 final_heading_error_deg=final_heading,
                                 displacement_m=event.get('displacement') if event.get('land_s', -1) >= 0 else None,
                                 flight_displacement_m=event.get('flight_displacement') if event.get('land_s', -1) >= 0 else None,
                                 reverse_distance_m=event.get('reverse_distance', 0.),
                                 strict_success=landed and heading_passed and bool(event.get('success')),
                                 clearance_m=float(event.get('apex_clearance', 0.)),
                                 joint_limit_violation_rad=float(limit_violation[index]),
                                 peak_tilt_deg=float(event.get('max_tilt', 0.)),
                                 request_velocity_error_m_s=float(request_velocity_error[index]),
                                 final_velocity_error_m_s=float(final_velocity_error[index])))
            result = summarize_jump_trials(rows, cfg)
            result['landing_recovery'] = summarize_landing_recovery(
                **{key:torch.stack(values) for key,values in recovery_trace.items()},
                dt=env.step_dt,request_step=request_step)
            result['attitude'] = summarize_jump_attitude(
                {key: torch.stack(values).cpu().numpy() for key, values in attitude_trace.items()},
                env.reward_manager.active_terms, env.step_dt)
            return result
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)


def summarize_jump_attitude(trace, reward_names, dt):
    """Summarize roll/pitch motion and reward integrals across jump phases.

    Phase 0 is outside the window; 1=ground propulsion, 2=air with vz>0,
    3=air with vz<=0, 4=landing. Reward integrals are averaged across all
    requested trials, including failures and rejected requests.
    """
    result = {}
    for name, codes in (('push', (1,)), ('air_ascent', (2,)), ('air_descent', (3,)),
                        ('flight', (2, 3)), ('landing', (4,)), ('window', (1, 2, 3, 4))):
        mask = np.isin(trace['phase'], codes)
        angles = np.abs(trace['angles'])
        peak = np.where(mask[..., None], angles, 0.).max(axis=0)
        peak_rate = np.where(mask[..., None], np.abs(trace['rates']), 0.).max(axis=0)
        reward = (trace['rewards'] * mask[..., None]).sum(axis=0).mean(axis=0) * dt
        result[name] = dict(
            trials_with_samples=int(mask.any(axis=0).sum()),
            mean_duration_s=float(mask.sum(axis=0).mean()*dt),
            mean_peak_angle_deg=dict(zip(('roll', 'pitch'), peak.mean(axis=0).tolist())),
            p90_peak_angle_deg=dict(zip(('roll', 'pitch'), np.quantile(peak, .9, axis=0).tolist())),
            mean_peak_rate_rad_s=dict(zip(('roll', 'pitch'), peak_rate.mean(axis=0).tolist())),
            roll_over20_count=int((peak[:, 0] > 20).sum()),
            mean_reward_integrals=dict(zip(reward_names, reward.tolist())))
    return result
