# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Recovery rewards, recorded-state restoration and physical occupancy metrics."""

import math

import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from . import mdp

# Joint angles and base height used by standing rewards.
Q_REF = {"L1": .2848744094, "R1": .2899141610, "L2": .4957503676, "R2": .5020063519}
H_REF = .1101198718
HEAD_REF = -.0133943176


def restore_recorded_state(env, qpos, qvel, env_ids=None):
    """Restore recorded state, converting root angular velocity from body to world coordinates."""
    robot = env.scene["robot"]
    velocity_w = torch.cat((qvel[:, :3], quat_apply(qpos[:, 3:7], qvel[:, 3:6])), dim=-1)
    robot.write_root_link_pose_to_sim(qpos[:, :7], env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(velocity_w, env_ids=env_ids)
    robot.write_joint_state_to_sim(qpos[:, 7:], qvel[:, 6:], env_ids=env_ids)


def near_upright(env):
    """Fade posture constraints out between 30 and 80 degrees, independent of contact."""
    cosine = -env.scene["robot"].data.projected_gravity_b[:, 2]
    low, high = math.cos(math.radians(80)), math.cos(math.radians(30))
    return ((cosine-low)/(high-low)).clamp(0, 1)


def height_error(env):
    height = mdp.base_height(env).squeeze(1)
    target = mdp.base_pose_command(env)[:, 2]
    active = near_upright(env)
    # Track commanded height near upright; penalize low base height during falls.
    return active * (height-target).square() + (1-active) * (H_REF-height).clamp_min(0) / 15


class posture_error:
    def __init__(self, cfg, env):
        self.ids, _ = env.scene["robot"].find_joints(list(Q_REF), preserve_order=True)
        self.reference = torch.tensor(list(Q_REF.values()), device=env.device)
        self.head_id = env.scene["robot"].find_joints("head_yaw_joint")[0][0]

    def __call__(self, env, head_weight=0.0):
        command = env.command_manager.get_command("twist")
        moving = command.norm(dim=-1) > .05
        std = torch.where(moving, .35, .20)
        error = (env.scene["robot"].data.joint_pos[:, self.ids]-self.reference)/std[:, None]
        # Bound each joint's posture error to [0, 1).
        penalty = (1-torch.rsqrt(1+error.square())).mean(-1)
        if head_weight:
            # Apply the head posture cost independently of the leg errors.
            head_error = (env.scene["robot"].data.joint_pos[:, self.head_id]-HEAD_REF)/.5
            penalty += head_weight * (1-torch.rsqrt(1+head_error.square()))
        return penalty*near_upright(env)


def position_limits(env, asset_cfg, target_margin=1.0):
    robot = env.scene[asset_cfg.name]
    actual = envs_mdp.joint_pos_limits(env, asset_cfg)
    limits = robot.data.joint_pos_limits[:, asset_cfg.joint_ids]
    target = robot.data.joint_pos_target[:, asset_cfg.joint_ids]
    # Penalize PD targets beyond the joint limits plus target_margin.
    excess = (limits[..., 0]-target_margin-target).clamp_min(0)
    excess += (target-limits[..., 1]-target_margin).clamp_min(0)
    return actual + .1 * excess.sum(-1)


def joint_velocity(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    velocity = robot.data.joint_vel[:, asset_cfg.joint_ids]
    wheels = mdp._contact_mask(env, "wheel_ground_contact", .1, use_history=False)
    # Penalize unloaded wheels above the speed threshold.
    wheel_excess = (velocity[:, 4:].abs()-83.7758).clamp_min(0) * (~wheels)
    return velocity[:, :4].square().sum(-1) + wheel_excess.square().sum(-1)


def configure_recovery_rewards(cfg):
    """Configure recovery posture, height, joint-limit and termination rewards."""
    before = set(cfg.rewards)
    # Reward weights are scaled by control dt; time-limit truncations are excluded.
    cfg.rewards["termination_penalty"].weight = -2500.
    term = cfg.rewards["base_height"]
    term.func, term.weight, term.params = height_error, -1500., {}
    term = cfg.rewards["pose"]
    term.func, term.weight, term.params = posture_error, -1.5, {}
    term.params["head_weight"] = .2
    term = cfg.rewards["joint_pos_limits"]
    term.func = position_limits
    term = cfg.rewards["leg_joint_vel"]
    term.func = joint_velocity
    term.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=(*Q_REF, "LW", "RW"), preserve_order=True)}
    assert set(cfg.rewards) == before


def sampling_metrics(env, env_ids=None):
    del env_ids
    if not hasattr(env, "_recovery_reset_kind"):
        return
    if not hasattr(env, "_curriculum_kind"):
        frames = torch.bincount(env._recovery_reset_kind, minlength=6)
        if not hasattr(env, "_recovery_frame_counts"):
            env._recovery_frame_counts = torch.zeros(6, device=env.device)
        env._recovery_frame_counts += frames
        log = env.extras.setdefault("log", {})
        for i, name in enumerate(("normal", "supine", "prone", "left_side", "right_side", "support")):
            log[f"Recovery/{name}_reset_share"] = env._recovery_reset_counts[i]/env._recovery_reset_counts.sum().clamp_min(1)
            log[f"Recovery/{name}_frame_share"] = env._recovery_frame_counts[i]/env._recovery_frame_counts.sum().clamp_min(1)
        log["Recovery/fallen_reset_share"] = env._recovery_reset_counts[1:5].sum()/env._recovery_reset_counts.sum().clamp_min(1)
        log["Recovery/fallen_frame_share"] = env._recovery_frame_counts[1:5].sum()/env._recovery_frame_counts.sum().clamp_min(1)
    log = env.extras.setdefault("log", {})
    # Measure current physical states separately from episode reset categories.
    robot = env.scene["robot"]
    height = mdp.base_height(env).squeeze(1)
    tilt = mdp.base_tilt(env)
    wheels = mdp._contact_mask(env, "wheel_ground_contact", .1, use_history=False)
    body = mdp._contact_mask(env, "body_ground_contact", .1).any(-1)
    supported = wheels.all(-1) & ~body & (tilt < math.radians(20))
    physical = torch.stack((body & ((tilt >= math.radians(70)) | (height < .065)),
                            body & ~wheels.any(-1) & (tilt < math.radians(20)), supported), dim=-1)
    if not hasattr(env, "_recovery_physical_counts"):
        env._recovery_physical_counts = torch.zeros(3, device=env.device)
        env._recovery_physical_frames = 0
        env._recovery_leg_ids = robot.find_joints(list(Q_REF), preserve_order=True)[0]
        env._recovery_q_ref = torch.tensor(list(Q_REF.values()), device=env.device)
    env._recovery_physical_counts += physical.sum(0)
    env._recovery_physical_frames += env.num_envs
    for i, name in enumerate(("fallen", "upright_no_wheels", "supported")):
        log[f"Recovery/physical_{name}_frame_share"] = env._recovery_physical_counts[i]/env._recovery_physical_frames
    for i, name in enumerate(("fallen", "upright_no_wheels", "supported")):
        log[f"Recovery/physical_{name}_current_share"] = physical[:, i].float().mean()
    ids = env._recovery_leg_ids
    limits = robot.data.joint_pos_limits[:, ids]
    q = robot.data.joint_pos[:, ids]
    margin = torch.minimum(q-limits[..., 0], limits[..., 1]-q).amin(-1)
    log["Recovery/near_leg_limit_current_share"] = (margin <= .02).float().mean()
    count = supported.sum().clamp_min(1)
    error = (robot.data.joint_pos[:, env._recovery_leg_ids]-env._recovery_q_ref).abs().amax(-1)
    log["Recovery/supported_height_mean_m"] = (height*supported).sum()/count
    log["Recovery/supported_leg_error_mean_rad"] = (error*supported).sum()/count
