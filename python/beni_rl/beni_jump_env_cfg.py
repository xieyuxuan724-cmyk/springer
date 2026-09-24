# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT
"""Jump training built on velocity tracking and recovery."""
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg

from . import jump
from .beni_velocity_flat_env_cfg import make_beni_velocity_flat_env_cfg
from .recovery import configure_recovery_rewards, H_REF


def make_beni_jump_env_cfg(play=False):
    cfg = make_beni_velocity_flat_env_cfg(play=play)
    configure_recovery_rewards(cfg)
    cfg.commands['jump'] = jump.BeniJumpCommandCfg(
        warmup_iterations=0 if play else 3000, play=play, probability=0. if play else .05)
    cfg.rewards['track_yaw_velocity'].func = jump.track_yaw_velocity_exp
    cfg.rewards['roll_pitch_velocity'].func = jump.ordinary_roll_pitch_velocity_l2
    for group in cfg.observations.values():
        group.terms['jump_command'] = ObservationTermCfg(
            func=envs_mdp.generated_commands, params={'command_name': 'jump'})
    # Fade standing rewards back in after landing; velocity rewards use wheel support.
    for name in ('base_height', 'pose', 'vertical_velocity'):
        term = cfg.rewards[name]
        term.params = {**term.params, 'reward_func': term.func}
        term.func = jump.flat_reward
    for name in ('body_contact', 'undesired_contacts'):
        term = cfg.rewards[name]
        term.params = {**term.params, 'reward_func': term.func, 'jump_scale': 5.}
        term.func = jump.contact_penalty
    limits = cfg.rewards['joint_pos_limits']
    limits.func = jump.jump_position_limits
    limits.params['jump_scale'] = 100.
    cfg.rewards.update({
        'jump_heading': RewardTermCfg(func=jump.heading_error_huber, weight=-2.5,
                                     params={'angle_scale_deg': 10.}),
        'large_tilt': RewardTermCfg(func=jump.large_tilt_l2, weight=-20.,
                                   params={'start_angle_deg': 30., 'full_angle_deg': 60.}),
        'jump_roll_angle': RewardTermCfg(func=jump.attitude_angle_huber, weight=-80.,
                                        params={'axis': 0, 'deadband_deg': 3., 'scale_deg': 10.}),
        'jump_pitch_angle': RewardTermCfg(func=jump.attitude_angle_huber, weight=-30.,
                                         params={'axis': 1, 'deadband_deg': 10., 'scale_deg': 20.}),
        'jump_roll_rate': RewardTermCfg(func=jump.attitude_rate_huber, weight=-4.,
                                       params={'axis': 0, 'scale_rad_s': 1.}),
        'jump_pitch_rate': RewardTermCfg(func=jump.attitude_rate_huber, weight=-2.,
                                        params={'axis': 1, 'scale_rad_s': 2.}),
        'jump_height_tracking_l1': RewardTermCfg(
            func=jump.jump_height_tracking_l1, weight=-1000., params={'nominal_height': H_REF}),
        'jump_vel_tracking_l2': RewardTermCfg(func=jump.jump_vel_tracking_l2, weight=-132.5826935437),
        'jump_knee_regulation': RewardTermCfg(func=jump.jump_knee_regulation, weight=200.),
        'jump_direction_progress': RewardTermCfg(func=jump.jump_direction_progress, weight=20.,
                                                  params={'distance_scale': .05}),
        'jump_reverse_velocity_l1': RewardTermCfg(func=jump.jump_reverse_velocity_l1, weight=-20.),
        'single_foot_penalty': RewardTermCfg(func=jump.single_foot_contact_penalty, weight=5.),
        'jump_takeoff_progress': RewardTermCfg(func=jump.jump_takeoff_progress, weight=200.),
        # RewardManager multiplies by dt=0.01: the first contact costs 50.
        'jump_body_failure': RewardTermCfg(func=jump.jump_body_failure, weight=-5000.),
    })
    cfg.events['jump_observer'] = EventTermCfg(func=jump.install_observer, mode='startup')
    return cfg
