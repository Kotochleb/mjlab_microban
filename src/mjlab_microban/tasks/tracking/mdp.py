"""Additional MDP terms for the Microban tracking task"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, TerrainHeightSensor
from mjlab.tasks.tracking.mdp import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def motion_global_anchor_velocity_z_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward tracking of the anchor vertical linear velocity of the motion."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.square(command.anchor_lin_vel_w[:, 2] - command.robot_anchor_lin_vel_w[:, 2])
    return torch.exp(-error / std**2)


def _is_upright(asset: Entity, max_tilt: float) -> torch.Tensor:
    """1.0 where the root is tilted less than max_tilt (rad) from vertical, else 0.0."""
    # The z component of the gravity projected in the root frame is -cos(tilt).
    return (-asset.data.projected_gravity_b[:, 2] > math.cos(max_tilt)).float()


def feet_slip_upright(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_tilt: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact), only when upright."""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
    vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
    cost = torch.sum(torch.square(vel_xy_norm) * in_contact, dim=1)
    env.extras["log"]["Metrics/slip_velocity_mean"] = torch.sum(
        vel_xy_norm * in_contact
    ) / torch.clamp(torch.sum(in_contact), min=1)
    return cost * _is_upright(asset, max_tilt)


class feet_swing_height_upright:
    """Penalize deviation from target swing height, evaluated at landing, only when upright."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        height_sensor = env.scene[cfg.params["height_sensor_name"]]
        assert isinstance(height_sensor, TerrainHeightSensor), (
            f"feet_swing_height_upright requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
        )
        self.peak_heights = torch.zeros(
            (env.num_envs, height_sensor.num_frames), device=env.device, dtype=torch.float32
        )
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        height_sensor_name: str,
        target_height: float,
        max_tilt: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene[sensor_name]
        height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
        foot_heights = height_sensor.data.heights
        in_air = contact_sensor.data.found == 0
        self.peak_heights = torch.where(
            in_air, torch.maximum(self.peak_heights, foot_heights), self.peak_heights
        )
        first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
        error = self.peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1)
        env.extras["log"]["Metrics/peak_height_mean"] = torch.sum(
            self.peak_heights * first_contact.float()
        ) / torch.clamp(torch.sum(first_contact.float()), min=1)
        self.peak_heights = torch.where(
            first_contact, torch.zeros_like(self.peak_heights), self.peak_heights
        )
        return cost * _is_upright(asset, max_tilt)
