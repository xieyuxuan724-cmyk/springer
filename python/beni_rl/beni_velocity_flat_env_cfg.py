# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""MJLab configuration for BENI flat-ground velocity training."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from . import mdp as beni_mdp
from .beni_robot import (
    BENI_ACTION_ORDER,
    BENI_ACTION_SCALE,
    BENI_LEG_JOINTS,
    BENI_POSITION_OBS_JOINTS,
    BENI_VELOCITY_OBS_JOINTS,
    get_beni_robot_cfg,
)
from .velocity_command import BeniGimbalCommandCfg, BeniVelocityCommandCfg
from .wheel_safety import WheelOverspeed

POLICY_OBSERVATION_DIM = 40
CRITIC_OBSERVATION_DIM = 50
ACTION_DIM = 7

POLICY_OBSERVATION_LAYOUT = {
    "base_ang_vel": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "velocity_commands": slice(6, 9),
    "joint_pos": slice(9, 14),
    "joint_vel": slice(14, 21),
    "last_action": slice(21, 28),
    "base_pose": slice(28, 31),
    "gimbal_command": slice(31, 33),
    "head_ang_vel": slice(33, 36),
    "jump_command": slice(36, 40),
}

_ILLEGAL_BODY_PATTERNS = ("base_link", r"[LR][12]_Link")
_BODY_CONTACT_BODY_PATTERNS = (*_ILLEGAL_BODY_PATTERNS, "head")


def _observation_terms(*, corrupted: bool) -> dict[str, ObservationTermCfg]:
    def noise(limit: float) -> UniformNoiseCfg | None:
        return UniformNoiseCfg(n_min=-limit, n_max=limit) if corrupted else None

    return {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_gyro"},
            noise=noise(0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=noise(0.05),
        ),
        "velocity_commands": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={
                "biased": corrupted,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=BENI_POSITION_OBS_JOINTS,
                    preserve_order=True,
                ),
            },
            noise=noise(0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=BENI_VELOCITY_OBS_JOINTS,
                    preserve_order=True,
                )
            },
            noise=noise(1.5),
        ),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action),
        "base_pose": ObservationTermCfg(func=beni_mdp.base_pose_command),
        "gimbal_command": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "gimbal"},
        ),
        "head_ang_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/head_imu_gyro"},
            noise=noise(0.2),
        ),
        "jump_command": ObservationTermCfg(
            func=beni_mdp.constant_observation,
            params={"values": (0.0, 0.0, 0.0, 0.0)},
        ),
    }


def make_beni_velocity_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the BENI flat-ground velocity environment."""
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": get_beni_robot_cfg()}
    cfg.scene.num_envs = 1 if play else 4096
    cfg.scene.env_spacing = 2.5
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None

    cfg.scene.sensors = (
        ContactSensorCfg(
            name="illegal_ground_contact",
            primary=ContactMatch(mode="body", pattern=_ILLEGAL_BODY_PATTERNS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            history_length=5,
        ),
        ContactSensorCfg(
            name="body_ground_contact",
            primary=ContactMatch(mode="body", pattern=_BODY_CONTACT_BODY_PATTERNS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            history_length=5,
        ),
        ContactSensorCfg(
            name="wheel_ground_contact",
            primary=ContactMatch(mode="body", pattern=("LW_Link", "RW_Link"), entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
        ),
    )

    actor_terms = _observation_terms(corrupted=not play)
    critic_terms = _observation_terms(corrupted=False)
    critic_terms["base_lin_vel"] = ObservationTermCfg(func=envs_mdp.base_lin_vel)
    critic_terms["base_height"] = ObservationTermCfg(func=beni_mdp.base_height)
    critic_terms["wheel_contact_forces"] = ObservationTermCfg(
        func=velocity_mdp.foot_contact_forces,
        params={"sensor_name": "wheel_ground_contact"},
    )
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
            nan_policy="error",
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            nan_policy="error",
        ),
    }

    # Separate terms preserve the deployment order despite BENI's tree-ordered MJCF.
    cfg.actions = {
        "hips": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("L1", "R1"),
            scale=BENI_ACTION_SCALE["L1"],
            use_default_offset=True,
        ),
        "knees": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("L2", "R2"),
            scale=BENI_ACTION_SCALE["L2"],
            use_default_offset=True,
        ),
        "wheels": JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=("LW", "RW"),
            scale=BENI_ACTION_SCALE["LW"],
            use_default_offset=True,
        ),
        "head_yaw": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("head_yaw_joint",),
            scale=BENI_ACTION_SCALE["head_yaw_joint"],
            use_default_offset=True,
        ),
    }

    cfg.commands = {
        "twist": BeniVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.25,
            rel_heading_envs=0.5,
            rel_world_envs=0.0,
            rel_forward_envs=0.2,
            heading_command=True,
            heading_control_stiffness=2.0,
            debug_vis=False,
            ranges=BeniVelocityCommandCfg.Ranges(
                lin_vel_x=(-2.0, 2.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-5.0, 5.0),
                heading=(-math.pi, math.pi),
            ),
        ),
        "gimbal": BeniGimbalCommandCfg(resampling_time_range=(0.0, 5.0)),
    }

    cfg.events = {
        "reset_base": EventTermCfg(
            func=beni_mdp.reset_root_state_mixed,
            mode="reset",
            params={
                "fallen_probability": 0.0 if play else 0.2,
                "pose_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                    "yaw": (-0.2, 0.2),
                    "roll": (0.0, 0.0) if play else (-0.05, 0.05),
                    "pitch": (0.0, 0.0) if play else (-0.05, 0.05),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        ),
        "reset_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0) if play else (-0.05, 0.05),
                "velocity_range": (0.0, 0.0) if play else (-0.1, 0.1),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    if not play:
        cfg.events.update(
            {
                "wheel_friction": EventTermCfg(
                    func=dr.geom_friction,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", geom_names=("LW_collision", "RW_collision")),
                        "ranges": (0.85, 1.15),
                        "operation": "abs",
                        "shared_random": True,
                    },
                ),
                "encoder_bias": EventTermCfg(
                    func=dr.encoder_bias,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", joint_names=BENI_POSITION_OBS_JOINTS),
                        "bias_range": (-0.015, 0.015),
                    },
                ),
                "base_com": EventTermCfg(
                    func=dr.body_com_offset,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                        "operation": "add",
                        "ranges": {0: (-0.005, 0.005), 1: (-0.005, 0.005), 2: (-0.005, 0.005)},
                    },
                ),
                "push_robot": EventTermCfg(
                    func=envs_mdp.apply_body_impulse,
                    mode="step",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                        "force_range": (-20.0, 20.0),
                        "torque_range": (-1.0, 1.0),
                        "duration_s": (0.01, 0.01),
                        "cooldown_s": (2.0, 4.0),
                    },
                ),
            }
        )

    cfg.rewards = {
        "termination_penalty": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
        "track_linear_velocity": RewardTermCfg(
            func=beni_mdp.track_linear_velocity_exp,
            weight=2.0,
            params={"command_name": "twist", "std": 0.5},
        ),
        "track_linear_velocity_sharp": RewardTermCfg(
            func=beni_mdp.track_linear_velocity_exp,
            weight=1.0,
            params={"command_name": "twist", "std": math.sqrt(0.02)},
        ),
        "track_linear_velocity_l1": RewardTermCfg(
            func=beni_mdp.track_linear_velocity_l1,
            weight=-2.0,
            params={"command_name": "twist"},
        ),
        "track_yaw_velocity": RewardTermCfg(
            func=beni_mdp.track_yaw_velocity_exp,
            weight=2.5,
            params={"command_name": "twist", "std": 0.5},
        ),
        "base_height": RewardTermCfg(
            func=beni_mdp.base_height_below_minimum,
            weight=-100.0,
            # Apply a lower height bound near the default wheel-supported pose.
            params={"minimum": 0.11},
        ),
        "pose": RewardTermCfg(
            func=beni_mdp.supported_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=BENI_POSITION_OBS_JOINTS, preserve_order=True),
                "command_name": "twist",
                "std_standing": {".*": 0.05},
                "std_walking": {".*": 0.2},
                "std_running": {".*": 0.3},
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        ),
        "leg_symmetry": RewardTermCfg(
            func=beni_mdp.leg_symmetry_l2,
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=BENI_LEG_JOINTS, preserve_order=True),
            },
        ),
        "joint_pos_limits": RewardTermCfg(
            func=envs_mdp.joint_pos_limits,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=BENI_POSITION_OBS_JOINTS),
            },
        ),
        "joint_torques": RewardTermCfg(
            func=envs_mdp.joint_torques_l2,
            weight=-2e-5,
            params={"asset_cfg": SceneEntityCfg("robot", actuator_names=BENI_ACTION_ORDER)},
        ),
        "joint_acc": RewardTermCfg(
            func=envs_mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BENI_ACTION_ORDER)},
        ),
        "leg_joint_vel": RewardTermCfg(
            func=envs_mdp.joint_vel_l2,
            weight=-2e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BENI_LEG_JOINTS)},
        ),
        # Penalize tilt continuously from upright to fully inverted.
        "base_tilt": RewardTermCfg(func=beni_mdp.base_tilt, weight=-4.0),
        "vertical_velocity": RewardTermCfg(
            func=beni_mdp.vertical_velocity_l2,
            weight=-5.0,
            params={"clip": 1.0},
        ),
        "roll_pitch_velocity": RewardTermCfg(
            func=beni_mdp.roll_pitch_velocity_l2,
            weight=-0.1,
            params={"clip": 20.0},
        ),
        "leg_action_rate": RewardTermCfg(
            func=beni_mdp.masked_action_rate_l2,
            weight=-0.05,
            params={"indices": (0, 1, 2, 3)},
        ),
        "leg_action_acc": RewardTermCfg(
            func=beni_mdp.masked_action_acc_l2,
            weight=-0.05,
            params={"indices": (0, 1, 2, 3)},
        ),
        "wheel_action_rate": RewardTermCfg(
            func=beni_mdp.masked_action_rate_l2,
            weight=-0.5,
            params={"indices": (4, 5)},
        ),
        "wheel_action_acc": RewardTermCfg(
            func=beni_mdp.masked_action_acc_l2,
            weight=-0.5,
            params={"indices": (4, 5)},
        ),
        "undesired_contacts": RewardTermCfg(
            func=beni_mdp.contact_count,
            weight=-0.5,
            params={"sensor_name": "illegal_ground_contact", "threshold": 0.1},
        ),
        "body_contact": RewardTermCfg(
            func=beni_mdp.any_contact,
            weight=-3.0,
            params={"sensor_name": "body_ground_contact", "threshold": 0.1},
        ),
    }

    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "wheel_overspeed": TerminationTermCfg(
            func=WheelOverspeed, params={"max_speed": 350., "history_s": .5}),
        "illegal_contact": TerminationTermCfg(
            func=beni_mdp.delayed_illegal_contact,
            params={
                "sensor_name": "illegal_ground_contact",
                "threshold": 0.1,
                "max_steps": 300,
            },
        ),
        "bad_orientation": TerminationTermCfg(
            func=beni_mdp.delayed_bad_orientation,
            params={"limit_angle": math.pi / 2, "max_steps": 300},
        ),
    }

    cfg.curriculum = {}
    if not play:
        cfg.curriculum["fallen_reset_probability"] = CurriculumTermCfg(
            func=beni_mdp.fallen_reset_curriculum,
            params={
                "initial_probability": 0.2,
                "final_probability": 0.5,
                "warmup_iterations": 1000,
                "ramp_iterations": 1000,
                # The runner replaces this with the actual PPO rollout length.
                "steps_per_iteration": 24,
            },
        )
    cfg.metrics = {}

    cfg.sim.mujoco.timestep = 0.002
    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.nconmax = None
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.decimation = 5
    cfg.episode_length_s = 1.0e9 if play else 20.0
    cfg.seed = 42
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 1.0
    cfg.viewer.elevation = -10.0

    return cfg


assert len(BENI_ACTION_ORDER) == ACTION_DIM
