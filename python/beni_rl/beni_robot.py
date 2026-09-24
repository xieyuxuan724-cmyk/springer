# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""BENI robot definition for mjlab."""

from pathlib import Path

import mujoco
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from . import BENI_MJCF
from .actuator import BeniMotorCfg

BENI_LEG_JOINTS = ("L1", "R1", "L2", "R2")
BENI_ACTION_ORDER = (*BENI_LEG_JOINTS, "LW", "RW", "head_yaw_joint")
BENI_POSITION_OBS_JOINTS = ("L1", "R1", "head_yaw_joint", "L2", "R2")
BENI_VELOCITY_OBS_JOINTS = (
    "L1",
    "R1",
    "head_yaw_joint",
    "L2",
    "R2",
    "LW",
    "RW",
)
BENI_ACTION_SCALE = {
    "L1": 0.25,
    "R1": 0.25,
    "L2": 0.25,
    "R2": 0.25,
    "LW": 20.0,
    "RW": 20.0,
    "head_yaw_joint": 0.5,
}
BENI_DEFAULT_JOINT_POS = {
    "L1": 0.3,
    "R1": 0.3,
    "L2": 0.5,
    "R2": 0.5,
    "head_yaw_joint": 0.0,
    "head_roll_joint": 0.0,
}


def get_beni_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(BENI_MJCF))
    mesh_dir = BENI_MJCF.parent.parent / "meshes"
    for mesh in spec.meshes:
        mesh.file = str((mesh_dir / Path(mesh.file).name).resolve())
    for actuator in tuple(spec.actuators):
        spec.delete(actuator)
    for name in ("LW", "RW"):
        geom = next(geom for geom in spec.body(f"{name}_Link").geoms if geom.contype or geom.conaffinity)
        geom.name = f"{name}_collision"
        # Let wheel friction below 1.0 override the ground's fixed friction.
        geom.priority = 1
    for name in (*BENI_ACTION_ORDER, "head_roll_joint"):
        joint = spec.joint(name)
        # Set encoder zero independently of the configured initial pose.
        joint.ref = 0.0
        # The motor model limits total torque, including the passive knee spring.
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec


_COMMAND_DELAY = {"delay_min_lag": 2, "delay_max_lag": 5}

BENI_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        BeniMotorCfg(
            target_names_expr=("L1", "R1"),
            stiffness=0.5,
            damping=0.02,
            effort_limit=1.375,
            armature=0.0001625,
            frictionloss=0.02,
            **_COMMAND_DELAY,
        ),
        BeniMotorCfg(
            target_names_expr=("L2", "R2"),
            stiffness=0.5,
            damping=0.02,
            effort_limit=3.6,
            simulation_effort_limit=4.6,
            spring_scale=1.0,
            armature=0.0005375,
            frictionloss=0.02,
            **_COMMAND_DELAY,
        ),
        BeniMotorCfg(
            target_names_expr=("LW", "RW"),
            stiffness=0.0,
            damping=0.004,
            effort_limit=0.6,
            armature=0.0,
            frictionloss=0.005,
            **_COMMAND_DELAY,
        ),
        IdealPdActuatorCfg(
            target_names_expr=("head_yaw_joint",),
            stiffness=3.0,
            damping=0.05,
            effort_limit=0.1,
            armature=0.0,
            frictionloss=0.0,
            **_COMMAND_DELAY,
        ),
        # The real gimbal follows roll mechanically.
        IdealPdActuatorCfg(
            target_names_expr=("head_roll_joint",),
            stiffness=2.0,
            damping=0.0,
            effort_limit=1.0,
            armature=0.0,
            frictionloss=0.0,
        ),
    ),
    soft_joint_pos_limit_factor=0.8,
)


def get_beni_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_beni_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.14),
            joint_pos=BENI_DEFAULT_JOINT_POS.copy(),
            joint_vel={".*": 0.0},
        ),
        articulation=BENI_ARTICULATION,
    )
