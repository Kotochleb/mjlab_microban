"""Additional MDP terms for the Microban climbing task (copied from tasks/tracking/mdp.py)"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, TerrainHeightSensor
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_inv, quat_mul

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


def _ball_site_offsets(
    env: ManagerBasedRlEnv, command: MotionCommand, site_names: tuple[str, ...], asset_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index (in the motion command's bodies) of the body carrying each ball site, and the site offset."""
    mj_model = env.sim.mj_model
    cmd_body_idx, offsets = [], []
    for name in site_names:
        site = mj_model.site(f"{asset_name}/{name}")
        body = mj_model.body(int(mj_model.site_bodyid[site.id])).name.split("/", 1)[-1]
        cmd_body_idx.append(command.cfg.body_names.index(body))
        offsets.append(mj_model.site_pos[site.id].copy())
    return (
        torch.tensor(cmd_body_idx, device=env.device),
        torch.tensor(offsets, dtype=torch.float32, device=env.device),  # [N, 3]
    )


class ball_position_error_exp:
    """Reward tracking of the climbing balls' positions (world frame) of the motion.

    The motion only stores body poses, so the reference ball positions are those of the bodies carrying
    the balls (which must be in the motion command's body_names) moved by the balls' fixed offsets.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        command = cast(MotionCommand, env.command_manager.get_term(cfg.params["command_name"]))
        asset_name = cfg.params.get("asset_name", "robot")
        asset: Entity = env.scene[asset_name]
        site_names = tuple(cfg.params["site_names"])
        self.cmd_body_idx, self.offsets = _ball_site_offsets(env, command, site_names, asset_name)
        self.site_ids = asset.find_sites(site_names, preserve_order=True)[0]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        site_names: tuple[str, ...],
        asset_name: str = "robot",
    ) -> torch.Tensor:
        command = cast(MotionCommand, env.command_manager.get_term(command_name))
        asset: Entity = env.scene[asset_name]
        body_pos = command.body_pos_w[:, self.cmd_body_idx]  # [B, N, 3]
        body_quat = command.body_quat_w[:, self.cmd_body_idx]  # [B, N, 4]
        offsets = self.offsets.expand(body_pos.shape[0], -1, -1)
        ref_pos = body_pos + quat_apply(body_quat, offsets)
        error = torch.sum(torch.square(ref_pos - asset.data.site_pos_w[:, self.site_ids]), dim=-1)  # [B, N]
        env.extras["log"]["Metrics/ball_pos_error_mean"] = torch.sqrt(error).mean()
        return torch.exp(-error.mean(dim=-1) / std**2)


def motion_joint_pos_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward tracking of the motion's joint configuration."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    asset: Entity = env.scene[asset_cfg.name]
    error = torch.sum(torch.square(command.joint_pos - asset.data.joint_pos), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_position_z_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward tracking of the anchor (trunk) height of the motion, in the world frame."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
    return torch.exp(-error / std**2)


class motion_future_window:
    """Rolling window over the next ``num_steps`` frames of the reference motion.

    At every policy step the window covers frames t+stride, t+2*stride, ..., t+num_steps*stride of the
    motion (t being the current motion time step), so it slides along the trajectory as the episode
    runs. Frames past the end of the motion are clamped to its last frame.

    Every frame is expressed in the *current reference* anchor (trunk) frame, so the term only depends
    on the motion (no robot state, hence no state estimation needed); the policy relates it to the
    robot through motion_anchor_pos_b / motion_anchor_ori_b. Per frame:
      - anchor position (3) and orientation (first two rotation matrix columns, 6),
      - ball positions (3 per ball site),
      - joint positions, relative to the current reference joint positions.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        command = cast(MotionCommand, env.command_manager.get_term(cfg.params["command_name"]))
        num_steps = cfg.params.get("num_steps", 20)
        stride = cfg.params.get("stride", 1)
        self.future = torch.arange(1, num_steps + 1, device=env.device) * stride  # [K]
        self.cmd_body_idx, self.ball_offsets = _ball_site_offsets(
            env, command, tuple(cfg.params["site_names"]), cfg.params.get("asset_name", "robot")
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        site_names: tuple[str, ...],
        num_steps: int = 20,
        stride: int = 1,
        asset_name: str = "robot",
    ) -> torch.Tensor:
        command = cast(MotionCommand, env.command_manager.get_term(command_name))
        motion = command.motion
        anchor = command.motion_anchor_body_index
        num_envs, num_future = command.time_steps.shape[0], self.future.shape[0]

        # Env origins cancel out in the relative poses, so the raw motion frames are used.
        t = command.time_steps
        idx = torch.clamp(t[:, None] + self.future, max=motion.time_step_total - 1)  # [B, K]

        ref_pos = motion.body_pos_w[t, anchor][:, None]  # [B, 1, 3]
        ref_quat_inv = quat_inv(motion.body_quat_w[t, anchor])[:, None]  # [B, 1, 4]

        # Future anchor poses in the current reference anchor frame.
        fut_anchor_pos = motion.body_pos_w[idx, anchor]  # [B, K, 3]
        fut_anchor_quat = motion.body_quat_w[idx, anchor]  # [B, K, 4]
        q_inv_k = ref_quat_inv.expand(-1, num_future, -1)
        anchor_pos_b = quat_apply(q_inv_k, fut_anchor_pos - ref_pos)
        anchor_ori_b = matrix_from_quat(quat_mul(q_inv_k, fut_anchor_quat))[..., :2]
        anchor_ori_b = anchor_ori_b.reshape(num_envs, num_future, 6)

        # Future ball positions in the current reference anchor frame.
        body_pos = motion.body_pos_w[idx[..., None], self.cmd_body_idx]  # [B, K, N, 3]
        body_quat = motion.body_quat_w[idx[..., None], self.cmd_body_idx]  # [B, K, N, 4]
        ball_pos_w = body_pos + quat_apply(body_quat, self.ball_offsets.expand_as(body_pos))
        q_inv_kn = ref_quat_inv[:, :, None].expand(-1, num_future, ball_pos_w.shape[2], -1)
        ball_pos_b = quat_apply(q_inv_kn, ball_pos_w - ref_pos[:, :, None])
        ball_pos_b = ball_pos_b.reshape(num_envs, num_future, -1)

        # Future joint positions relative to the current reference ones (the current ones are in the
        # "command" observation), i.e. in the same space as the residual actions.
        joint_pos_delta = motion.joint_pos[idx] - motion.joint_pos[t][:, None]  # [B, K, J]

        return torch.cat([anchor_pos_b, anchor_ori_b, ball_pos_b, joint_pos_delta], dim=-1).reshape(
            num_envs, -1
        )


@dataclass(kw_only=True)
class ResidualJointPositionActionCfg(JointPositionActionCfg):
    """Joint position targets as residuals on the reference motion: target = ref_joint_pos + scale * a.

    The reference frame used is the one the step ends on (motion time step t + 1): the targets are
    applied during the physics substeps of the step, after which the motion command advances to t + 1
    and the rewards compare the robot to that frame. Using frame t would make the feedforward lag by one
    policy step.
    """

    command_name: str = "motion"
    use_default_offset: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.use_default_offset or self.offset != 0.0:
            raise ValueError("ResidualJointPositionActionCfg adds the reference: no offset allowed.")

    def build(self, env: ManagerBasedRlEnv) -> ResidualJointPositionAction:
        return ResidualJointPositionAction(self, env)


class ResidualJointPositionAction(JointPositionAction):
    cfg: ResidualJointPositionActionCfg

    def __init__(self, cfg: ResidualJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._env = env
        self._command: MotionCommand | None = None  # the command manager is built after the actions

    def apply_actions(self) -> None:
        if self._command is None:
            self._command = cast(MotionCommand, self._env.command_manager.get_term(self.cfg.command_name))
        motion = self._command.motion
        next_step = torch.clamp(self._command.time_steps + 1, max=motion.time_step_total - 1)
        ref_joint_pos = motion.joint_pos[next_step][:, self._target_ids]
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = ref_joint_pos + self._processed_actions - encoder_bias
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)



def _pose_w(pos_w: torch.Tensor, quat_w: torch.Tensor, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Position relative to the env origin (3) and orientation as the first two rotation matrix columns (6)."""
    rot = matrix_from_quat(quat_w)[..., :2].reshape(pos_w.shape[0], 6)
    return torch.cat([pos_w - env.scene.env_origins, rot], dim=-1)


def robot_anchor_pose_w(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """World placement of the robot's anchor (trunk): position relative to the env origin and orientation."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return _pose_w(command.robot_anchor_pos_w, command.robot_anchor_quat_w, env)


def motion_anchor_pose_w(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """World placement of the motion's reference anchor (trunk): position relative to the env origin and
    orientation."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return _pose_w(command.anchor_pos_w, command.anchor_quat_w, env)
