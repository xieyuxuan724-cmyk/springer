# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""BENI PD motors and passive knee springs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.actuator import ActuatorCmd, IdealPdActuator, IdealPdActuatorCfg
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
    from mjlab.entity import Entity


def knee_spring_torque(
    theta: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Return passive knee spring torque in N*m for joint angles in radians.

    ``scale`` multiplies the spring torque; zero disables the spring.
    """
    angle = theta - 1.0
    return scale * (0.22 * torch.sin(angle) + 0.15 * torch.sin(2.0 * angle))


@dataclass(kw_only=True)
class BeniMotorCfg(IdealPdActuatorCfg):
    """PD motor with constant torque limits and an optional knee spring."""

    simulation_effort_limit: float | None = None
    spring_scale: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.simulation_effort_limit is not None and self.simulation_effort_limit < self.effort_limit:
            raise ValueError("simulation_effort_limit cannot be below effort_limit")
        if self.spring_scale < 0.0:
            raise ValueError("spring_scale cannot be negative")

    def build(
        self,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> BeniMotor:
        return BeniMotor(self, entity, target_ids, target_names)


class BeniMotor(IdealPdActuator[BeniMotorCfg]):
    """Apply PD motor torque and passive knee spring torque each physics substep."""

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        sim_limit = (
            self.cfg.effort_limit if self.cfg.simulation_effort_limit is None else self.cfg.simulation_effort_limit
        )
        for target_name in target_names:
            self._mjs_actuators.append(
                create_motor_actuator(
                    spec,
                    target_name,
                    effort_limit=sim_limit,
                    armature=self.cfg.armature,
                    frictionloss=self.cfg.frictionloss,
                    viscous_damping=self.cfg.viscous_damping,
                    transmission_type=self.cfg.transmission_type,
                )
            )

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        motor_torque = super().compute(cmd)
        if self.cfg.spring_scale:
            motor_torque = motor_torque + knee_spring_torque(cmd.pos, self.cfg.spring_scale)
        sim_limit = (
            self.cfg.effort_limit if self.cfg.simulation_effort_limit is None else self.cfg.simulation_effort_limit
        )
        return motor_torque.clamp(-sim_limit, sim_limit)
