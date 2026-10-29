"""Additional MDP terms for the Microban tracking task"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.tracking.mdp import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def motion_global_anchor_velocity_z_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward tracking of the anchor vertical linear velocity of the motion."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.square(command.anchor_lin_vel_w[:, 2] - command.robot_anchor_lin_vel_w[:, 2])
    return torch.exp(-error / std**2)
