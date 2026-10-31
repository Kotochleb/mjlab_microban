"""Additional MDP terms for the Microban climbing task (copied from tasks/tracking/mdp.py)"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import warp as wp

from bam.mjlab import BamActuator
from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, TerrainHeightSensor
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_error_magnitude, quat_inv, quat_mul

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
        torch.tensor(np.array(offsets), dtype=torch.float32, device=env.device),  # [N, 3]
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
        reduction: str = "mean",
    ) -> torch.Tensor:
        """exp(-e / std^2), e being the mean (or, with reduction="sum", the sum as in DeepMimic's end-effector
        reward) over the balls of the squared position errors."""
        command = cast(MotionCommand, env.command_manager.get_term(command_name))
        asset: Entity = env.scene[asset_name]
        body_pos = command.body_pos_w[:, self.cmd_body_idx]  # [B, N, 3]
        body_quat = command.body_quat_w[:, self.cmd_body_idx]  # [B, N, 4]
        offsets = self.offsets.expand(body_pos.shape[0], -1, -1)
        ref_pos = body_pos + quat_apply(body_quat, offsets)
        error = torch.sum(torch.square(ref_pos - asset.data.site_pos_w[:, self.site_ids]), dim=-1)  # [B, N]
        # Envs whose physics just blew up (NaN state, reset by nan_detection right after the rewards) would
        # make the batch mean NaN: average over the finite ones and log how many were not.
        ball_err = torch.sqrt(error).mean(dim=-1)  # [B]
        finite = torch.isfinite(ball_err)
        env.extras["log"]["Metrics/ball_pos_error_mean"] = ball_err[finite].mean()
        env.extras["log"]["Metrics/nonfinite_envs"] = (~finite).sum().float()
        error = error.sum(dim=-1) if reduction == "sum" else error.mean(dim=-1)
        return torch.exp(-error / std**2)


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


# ---------------------------------------------------------------------------------------------------------
# DeepMimic imitation reward terms (Peng et al., "DeepMimic", SIGGRAPH 2018): exp(-k * squared error)
# ---------------------------------------------------------------------------------------------------------


def deepmimic_pose_exp(
    env: ManagerBasedRlEnv, command_name: str, k: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Pose reward: joint angle errors plus the root orientation error (angle), squared and summed."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    asset: Entity = env.scene[asset_cfg.name]
    joint_err = torch.sum(torch.square(command.joint_pos - asset.data.joint_pos), dim=-1)
    root_err = torch.square(quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w))
    return torch.exp(-k * (joint_err + root_err))


def deepmimic_velocity_exp(
    env: ManagerBasedRlEnv, command_name: str, k: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Velocity reward: joint velocity errors plus the root angular velocity error, squared and summed."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    asset: Entity = env.scene[asset_cfg.name]
    joint_err = torch.sum(torch.square(command.joint_vel - asset.data.joint_vel), dim=-1)
    root_err = torch.sum(torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w), dim=-1)
    return torch.exp(-k * (joint_err + root_err))


def deepmimic_root_position_exp(env: ManagerBasedRlEnv, command_name: str, k: float) -> torch.Tensor:
    """Root (trunk) position reward, in the world frame: DeepMimic's centre-of-mass term on the root."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    err = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-k * err)


def residual_action_l2(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Squared norm of the policy's residual (raw action): keeps the policy close to the reference."""
    return torch.sum(torch.square(env.action_manager.get_term(action_name).raw_action), dim=-1)


def motion_global_anchor_position_z_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward tracking of the anchor (trunk) height of the motion, in the world frame."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
    return torch.exp(-error / std**2)


class motion_future_window:
    """Rolling window over future frames of the reference motion.

    At every policy step the window covers frames t+o for o in ``offsets`` (t being the current motion
    time step), so it slides along the trajectory as the episode runs; the offsets can get sparser further
    ahead. Frames past the end of the motion are clamped to its last frame.

    Every frame is expressed in the *current reference* anchor (trunk) frame, so the term only depends
    on the motion (no robot state); the critic relates it to the robot through its anchor pose terms. Per frame:
      - anchor position (3) and orientation (first two rotation matrix columns, 6),
      - ball positions (3 per ball site),
      - joint positions, relative to the current reference joint positions.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        command = cast(MotionCommand, env.command_manager.get_term(cfg.params["command_name"]))
        self.future = torch.tensor(cfg.params["offsets"], device=env.device)  # [K]
        self.cmd_body_idx, self.ball_offsets = _ball_site_offsets(
            env, command, tuple(cfg.params["site_names"]), cfg.params.get("asset_name", "robot")
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        site_names: tuple[str, ...],
        offsets: tuple[int, ...],
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


# ---------------------------------------------------------------------------------------------------------
# PD gain and stiffness-dependent action scale (BAM actuators)
# ---------------------------------------------------------------------------------------------------------


def _bam_actuators(asset: Entity) -> list[BamActuator]:
    acts = [act for act in asset.actuators if isinstance(act, BamActuator)]
    assert acts, "the PD gain terms need BAM actuators"
    return acts


def bam_kp(asset: Entity) -> torch.Tensor:
    """Current firmware P gain (kp_fw units) of each env, shape [B]."""
    act = _bam_actuators(asset)[0]
    return act.kp_scale[:, 0] * float(act._bam_model.actuator.kp)


def set_bam_kp(asset: Entity, kp: float | torch.Tensor) -> None:
    """Set the firmware P gain (kp_fw units) of all BAM actuators, for all envs."""
    for act in _bam_actuators(asset):
        act.kp_scale[:] = kp / float(act._bam_model.actuator.kp)


def bam_stiffness_and_effort_limit(act: BamActuator) -> tuple[float, float]:
    """(stiffness per unit kp_fw [Nm/rad], effort limit [Nm]) of a BAM actuator, at its nominal voltage.

    BAM firmware law: torque = kt/R * vin * clip(kp_fw * error_gain * err, +-max_pwm) (- back-EMF), clipped
    to +-max_current * kt. So the stiffness is K = kt/R * vin * error_gain * kp_fw. vin is the middle of
    the randomised range.
    """
    bam, cfg = act._bam_model, act.cfg
    kt, R = bam.kt.value, bam.R.value
    vin = sum(cfg.vin_range) / 2 if cfg.vin_range is not None else bam.actuator.vin
    effort_limit = kt / R * vin * bam.actuator.max_pwm
    if cfg.max_current is not None:
        effort_limit = min(effort_limit, cfg.max_current * kt)
    return kt / R * vin * bam.actuator.error_gain, effort_limit


def set_pd_gain(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    kp: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Set the firmware P gain (kp_fw units) of all BAM actuators, for all envs (startup event)."""
    set_bam_kp(env.scene[asset_cfg.name], kp)


class bam_power_curriculum:
    """Curriculum on the BAM motor power: starts with a boosted supply voltage and current limit and goes
    linearly down to the realistic ones (the actuator config's) over ``num_steps`` env steps.

    Every env interpolates from ``vin_start`` to its own startup-sampled battery voltage (``vin_range``), and the
    firmware current limit from ``max_current_start`` to the config's ``max_current``. The load voltage drop and
    ``vin_min`` still apply on top. MuJoCo's actuator force range (set by BAM from the realistic max voltage) is
    raised once so it does not clip the boosted torques; it stays a ceiling above the realistic limits after.

    Only the physics change: the residual action scale and the torque utilisation barrier keep using
    the realistic effort limit / the current limit, respectively (the latter follows the curriculum).
    Curriculum terms run on env resets, so the values are updated whenever some env resets.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
        vin_start = float(cfg.params["vin_start"])
        max_current_start = float(cfg.params["max_current_start"])
        self._acts = _bam_actuators(asset)
        self._vin_final = [act.vin_tensor.clone() for act in self._acts]  # [B, 1], realistic per-env voltage
        self._max_current_final = [act.cfg.max_current for act in self._acts]
        # Each actuator gets its own copy of the config: it may be shared with other robots / tasks.
        for act in self._acts:
            act.cfg = replace(act.cfg)

        # The raw array is [1 or B, nu, 2] (1 when shared by all worlds); the bridge's view broadcasts it.
        forcerange = wp.to_torch(env.sim.model.actuator_forcerange.wp_array)
        for act in self._acts:
            bam = act._bam_model
            kt, R = bam.kt.value, bam.R.value
            limit = kt / R * vin_start * bam.actuator.max_pwm
            limit = min(limit, max_current_start * kt)
            # Add a margin for the friction torque BAM adds on top of the motor torque.
            limit *= 1.1
            ids = act.global_ctrl_ids
            new = torch.clamp(forcerange[..., ids, 1], min=limit)
            forcerange[..., ids, 0] = -new
            forcerange[..., ids, 1] = new

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice,
        vin_start: float,
        max_current_start: float,
        num_steps: int,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> dict[str, float]:
        del env_ids, asset_cfg
        alpha = min(max(env.common_step_counter / num_steps, 0.0), 1.0)
        for act, vin_final, max_current_final in zip(
            self._acts, self._vin_final, self._max_current_final, strict=True
        ):
            act.vin_tensor[:] = vin_start + alpha * (vin_final - vin_start)
            if max_current_final is not None:
                act.cfg.max_current = max_current_start + alpha * (max_current_final - max_current_start)
        act = self._acts[0]
        return {
            "progress": alpha,
            "vin_mean": float(act.vin_tensor.mean()),
            "max_current": float(act.cfg.max_current) if act.cfg.max_current is not None else float("nan"),
        }


def bam_torque_utilisation(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Fraction of the currently available motor torque used by each BAM actuator, in [0, 1], shape [B, J].

    It follows BAM's firmware law (BamActuator.compute): torque = kt/R * (vin * duty - kt * kd_scale * vel), with the
    duty clipped to +-max_pwm, then clipped to +-max_current * kt. So the torque available in the direction the motor
    pushes shrinks with the back-EMF when it pushes along the joint velocity and grows against it:
        available = min(max_current * kt, kt/R * (vin * max_pwm - sign(torque) * kt * kd_scale * vel)).
    Torques are the motor torques (before joint friction) of the last physics substep, and vin the per-env battery
    voltage minus its load drop.
    """
    asset: Entity = env.scene[asset_cfg.name]
    out = []
    for act in _bam_actuators(asset):
        bam, cfg = act._bam_model, act.cfg
        kt, R, max_pwm = bam.kt.value, bam.R.value, bam.actuator.max_pwm
        torque = act._prev_motor_torque  # [B, J]
        vin = act.vin_tensor  # [B, 1]
        if act.vin_drop_gain is not None:
            vin = vin - act.vin_drop_gain * torque.abs().sum(dim=-1, keepdim=True)
            if cfg.vin_min is not None:
                vin = torch.clamp(vin, min=cfg.vin_min)
        vel = env.sim.data.qvel[:, act._dof_ids]
        available = kt / R * (vin * max_pwm - torch.sign(torque) * kt * act.kd_scale * vel)
        if cfg.max_current is not None:
            available = torch.clamp(available, max=cfg.max_current * kt)
        # Back-EMF can leave (almost) nothing available: then any torque in that direction is at the limit.
        out.append(torch.clamp(torque.abs() / available.clamp(min=1e-6), max=1.0))
    return torch.cat(out, dim=-1)


def motor_torque_exp_barrier(
    env: ManagerBasedRlEnv,
    threshold: float = 0.95,
    sharpness: float = 0.0125,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Exponential barrier on the motor torque utilisation u (bam_torque_utilisation), summed over the joints:
        exp(max(u - threshold, 0) / sharpness) - 1.
    0 below the threshold (95 % of the available torque), continuous there, and growing exponentially up to the
    limit: with sharpness 0.0125, e^4 - 1 = 53.6 per joint at saturation."""
    u = bam_torque_utilisation(env, asset_cfg)
    cost = torch.exp(torch.clamp(u - threshold, min=0.0) / sharpness) - 1.0
    cost = torch.nan_to_num(cost, nan=0.0)
    env.extras["log"]["Metrics/torque_utilisation_max"] = torch.nan_to_num(u).amax(dim=-1).mean()
    env.extras["log"]["Metrics/torque_over_threshold_frac"] = (u > threshold).float().mean()
    return cost.sum(dim=-1)


# ---------------------------------------------------------------------------------------------------------
# Privileged (critic) terms: contacts, contact schedule of the reference, tracking errors, torques
# ---------------------------------------------------------------------------------------------------------


class ball_contacts:
    """Ball-wall contact flags (N) and net contact forces in the world frame (N * 3), in ``geom_names`` order.

    The contact sensor lists its geoms in model order; they are put back in ``geom_names`` order (the order
    of the ball sites used by the other terms).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_name = cfg.params.get("asset_name", "robot")
        names = tuple(cfg.params["geom_names"])
        ids = [env.sim.mj_model.geom(f"{asset_name}/{n}").id for n in names]
        model_order = sorted(range(len(names)), key=lambda i: ids[i])  # sensor slot -> geom_names index
        self.perm = torch.tensor([model_order.index(i) for i in range(len(names))], device=env.device)  # name -> slot

    def __call__(
        self, env: ManagerBasedRlEnv, sensor_name: str, geom_names: tuple[str, ...], asset_name: str = "robot"
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        assert sensor.data.found is not None and sensor.data.force is not None
        found = (sensor.data.found[:, self.perm] > 0).float()
        force = sensor.data.force[:, self.perm].reshape(found.shape[0], -1)
        return torch.cat([found, force], dim=-1)


class motion_contact_schedule:
    """Where and when the reference expects the balls to be on holds.

    The planned motion has no contact labels: a ball is expected in contact on the frames where its
    reference position is within ``threshold`` of a hold socket (``hold_positions``, env frame: the centre
    of a ball hooked in the hold, wall.hold_positions).
    Per ball:
      - expected contact now (1) and at the frames t+o for o in ``offsets`` (K),
      - position of the next expected contact (the hold, from frame t on), in the robot's trunk frame (3),
      - time until that contact [s], clipped to ``max_time`` (1); 0 while it is expected now.
    After the last contact of the motion, the next contact is the last one.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        command = cast(MotionCommand, env.command_manager.get_term(cfg.params["command_name"]))
        motion = command.motion
        idx, offs = _ball_site_offsets(env, command, tuple(cfg.params["site_names"]), cfg.params.get("asset_name", "robot"))
        body_pos, body_quat = motion.body_pos_w[:, idx], motion.body_quat_w[:, idx]  # [T, N, ...], env frame
        ball_ref = body_pos + quat_apply(body_quat, offs.expand_as(body_pos))
        holds = torch.tensor(cfg.params["hold_positions"], dtype=torch.float32, device=env.device)  # [H, 3], env frame
        dist, hold_id = torch.cdist(ball_ref, holds[None].expand(ball_ref.shape[0], -1, -1)).min(dim=-1)  # [T, N]
        self.contact = dist < cfg.params.get("threshold", 0.003)  # [T, N]

        # Next expected contact frame from each frame on (backward scan); after the last one, the last one.
        contact = self.contact.cpu().numpy()
        T, N = contact.shape
        nxt = np.empty((T, N), dtype=np.int64)
        last = np.full(N, -1)
        for t in range(T - 1, -1, -1):
            last = np.where(contact[t], t, last)
            nxt[t] = last
        for n in range(N):  # no contact left: the last contact of the motion (or the last frame)
            c = np.nonzero(contact[:, n])[0]
            nxt[nxt[:, n] < 0, n] = c[-1] if len(c) else T - 1
        self.next_frame = torch.tensor(nxt, device=env.device)  # [T, N]
        self.next_pos = holds[hold_id.gather(0, self.next_frame)]  # [T, N, 3], env frame
        self.future = torch.tensor(cfg.params["offsets"], device=env.device)  # [K]
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        site_names: tuple[str, ...],
        offsets: tuple[int, ...],
        hold_positions: tuple[tuple[float, float, float], ...],
        threshold: float = 0.003,
        max_time: float = 1.0,
        asset_name: str = "robot",
    ) -> torch.Tensor:
        command = cast(MotionCommand, env.command_manager.get_term(command_name))
        t = command.time_steps
        B = t.shape[0]
        idx = torch.clamp(t[:, None] + self.future, max=self.contact.shape[0] - 1)  # [B, K]
        now = self.contact[t].float()  # [B, N]
        future = self.contact[idx].float().reshape(B, -1)  # [B, K * N]
        next_pos_w = self.next_pos[t] + env.scene.env_origins[:, None]  # [B, N, 3]
        q_inv = quat_inv(command.robot_anchor_quat_w)[:, None].expand(-1, next_pos_w.shape[1], -1)
        next_pos_b = quat_apply(q_inv, next_pos_w - command.robot_anchor_pos_w[:, None]).reshape(B, -1)
        time_to = torch.clamp((self.next_frame[t] - t[:, None]).clamp(min=0) * self.step_dt, max=max_time)
        return torch.cat([now, future, next_pos_b, time_to], dim=-1)


class motion_tracking_errors_b:
    """Reference minus robot, in the robot's trunk frame: ball positions (N * 3) and trunk position (3)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        command = cast(MotionCommand, env.command_manager.get_term(cfg.params["command_name"]))
        asset_name = cfg.params.get("asset_name", "robot")
        site_names = tuple(cfg.params["site_names"])
        self.cmd_body_idx, self.offsets = _ball_site_offsets(env, command, site_names, asset_name)
        self.site_ids = env.scene[asset_name].find_sites(site_names, preserve_order=True)[0]

    def __call__(
        self, env: ManagerBasedRlEnv, command_name: str, site_names: tuple[str, ...], asset_name: str = "robot"
    ) -> torch.Tensor:
        command = cast(MotionCommand, env.command_manager.get_term(command_name))
        asset: Entity = env.scene[asset_name]
        body_pos = command.body_pos_w[:, self.cmd_body_idx]
        body_quat = command.body_quat_w[:, self.cmd_body_idx]
        ref_ball = body_pos + quat_apply(body_quat, self.offsets.expand(body_pos.shape[0], -1, -1))  # [B, N, 3]
        ball_err_w = ref_ball - asset.data.site_pos_w[:, self.site_ids]
        root_err_w = (command.anchor_pos_w - command.robot_anchor_pos_w)[:, None]  # [B, 1, 3]
        err_w = torch.cat([ball_err_w, root_err_w], dim=1)
        q_inv = quat_inv(command.robot_anchor_quat_w)[:, None].expand(-1, err_w.shape[1], -1)
        return quat_apply(q_inv, err_w).reshape(err_w.shape[0], -1)


def contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Number of the sensor's primaries (e.g. balls) in contact with its secondary."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return (sensor.data.found > 0).float().sum(dim=-1)


def applied_joint_torques(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Torques applied by the actuators [Nm]."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force


def motion_future_joint_pos(
    env: ManagerBasedRlEnv,
    command_name: str,
    num_steps: int = 20,
    stride: int = 1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Future reference joint configurations: frames t+stride, ..., t+num_steps*stride of the motion.

    Joint positions are relative to the default joint positions (as the joint_pos observation), and
    frames past the end of the motion are clamped to its last frame. Shape [B, num_steps * num_joints].
    """
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    asset: Entity = env.scene[asset_cfg.name]
    motion = command.motion
    future = torch.arange(1, num_steps + 1, device=env.device) * stride  # [K]
    idx = torch.clamp(command.time_steps[:, None] + future, max=motion.time_step_total - 1)  # [B, K]
    joint_pos = motion.joint_pos[idx] - asset.data.default_joint_pos[:, None]  # [B, K, J]
    return joint_pos.reshape(joint_pos.shape[0], -1)


@dataclass(kw_only=True)
class ResidualJointPositionActionCfg(JointPositionActionCfg):
    """Joint position targets as residuals on the reference motion: target = ref_joint_pos + scale * a.

    The reference frame used is the one the step ends on (motion time step t + 1): the targets are
    applied during the physics substeps of the step, after which the motion command advances to t + 1
    and the rewards compare the robot to that frame. Using frame t would make the feedforward lag by one
    policy step.

    With ``stiffness_scale`` set, ``scale`` is ignored and the action scale is BeyondMimic's
    ``stiffness_scale * effort_limit / stiffness``, recomputed every step from each env's current BAM P gain
    (so it follows any change of the gain).
    """

    command_name: str = "motion"
    use_default_offset: bool = False
    stiffness_scale: float | None = None

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
        if cfg.stiffness_scale is not None:
            self._stiffness_per_kp, self._effort_limit = bam_stiffness_and_effort_limit(_bam_actuators(self._entity)[0])

    def process_actions(self, actions: torch.Tensor) -> None:
        if self.cfg.stiffness_scale is not None:
            stiffness = self._stiffness_per_kp * bam_kp(self._entity)  # [B] Nm/rad
            self._scale = (self.cfg.stiffness_scale * self._effort_limit / stiffness).unsqueeze(-1)
        super().process_actions(actions)

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


# ---------------------------------------------------------------------------------------------------------
# Heading (rotation about z) observations
# ---------------------------------------------------------------------------------------------------------
# An angle is fed to the networks as (cos, sin) rather than as the raw angle or its sine alone: the raw angle
# jumps at +-pi, and sin alone is ambiguous (sin(a) = sin(pi - a)) and loses resolution around +-pi/2. The 2D
# (cos, sin) point on the unit circle is the continuous representation of SO(2) (Zhou et al., "On the Continuity
# of Rotation Representations in Neural Networks", CVPR 2019 -- whose 6D representation, used above for full
# orientations, is the SO(3) counterpart), and is the usual heading encoding in legged / humanoid RL.


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Rotation about z (rad, in (-pi, pi]) of wxyz quaternions."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _cos_sin(angle: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


class base_yaw_windup_cos_sin:
    """Actor heading: the trunk's rotation about z since the start of the episode, plus a random offset, as
    (cos, sin).

    The angle is wound up: every step adds the (wrapped) yaw increment of the trunk to an unbounded accumulator,
    so it is continuous across +-pi and counts full turns (as an IMU-integrated heading would). On reset the
    accumulator starts at a random offset in [offset_range), by default a full turn, so the policy cannot rely on
    the absolute heading -- only on how it changes, as with a real IMU whose yaw reference is arbitrary. The
    reference yaw is taken lazily on the first observation after a reset, i.e. after the motion command has
    written the episode's initial state.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        lo, hi = cfg.params.get("offset_range", (0.0, 2.0 * math.pi))
        self.offset_range = (float(lo), float(hi))
        self.windup = torch.zeros(env.num_envs, device=env.device)
        self.prev_yaw = torch.zeros(env.num_envs, device=env.device)
        self.needs_init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.needs_init[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        offset_range: tuple[float, float] = (0.0, 2.0 * math.pi),
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        yaw = _yaw_from_quat(asset.data.root_link_quat_w)
        init = self.needs_init
        if init.any():
            lo, hi = self.offset_range
            offset = torch.rand(int(init.sum()), device=yaw.device) * (hi - lo) + lo
            self.windup[init] = offset
            self.prev_yaw[init] = yaw[init]
            self.needs_init[:] = False
        # Wrapped increment (|delta| < pi per 20 ms step), accumulated without wrapping. Non-finite yaws (blown-up
        # envs, reset right after) do not poison the accumulator.
        delta = torch.atan2(torch.sin(yaw - self.prev_yaw), torch.cos(yaw - self.prev_yaw))
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        self.windup += delta
        self.prev_yaw = torch.where(torch.isfinite(yaw), yaw, self.prev_yaw)
        return _cos_sin(self.windup)


def base_yaw_rel_board_cos_sin(
    env: ManagerBasedRlEnv,
    board_name: str = "wall",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Privileged heading: the trunk's rotation about the board's z axis, relative to the board, as (cos, sin).

    Exact and drift-free (no episode offset): the trunk orientation is expressed in the board frame and its yaw
    taken there, so it is 0 when the trunk faces the board as in the start pose.
    """
    asset: Entity = env.scene[asset_cfg.name]
    board: Entity = env.scene[board_name]
    q_rel = quat_mul(quat_inv(board.data.root_link_quat_w), asset.data.root_link_quat_w)
    return _cos_sin(_yaw_from_quat(q_rel))
