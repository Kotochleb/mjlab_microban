# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Adaptive phase dynamics for the climbing task, after Sleiman, Mittal & Hutter, "Guided Reinforcement Learning
for Robust Multi-Contact Loco-Manipulation" (CoRL 2024, arXiv:2410.13817).

The reference motion is no longer played back at a fixed rate: its phase advances at a rate that drops when the
robot falls behind the demonstration, so the reference waits for the robot (e.g. a ball that slipped off a hold)
instead of running away from it, and the policy can learn recoveries absent from the planned climb.

Paper (phase phi in [0, 1] over the demonstration of duration T):
    phi_{t+1}  = phi_t + v_t / T * dt
    v_t        = v_ref_t + sigma_1 * dv_t                         (dv_t: an extra policy action, sigma_1 = 0.01)
    v_ref_t    = prod_i min(k * exp(-lambda * r_i), 1)            (r_i <= 0: the task-level tracking rewards, k = 10)
    r_phi      = kappa_1 * v_ref * exp(-kappa_2 * (phi - phi_nom)^2)   (phi_nom: the nominal, fixed-rate phase)
plus a curriculum level l in [0, 1] scaling the regularisation penalties and the initial state randomisation,
moved by +-0.25 from the phase progress p = (phi - phi_init) / (phi_nom - phi_init) (up above 0.95, down below 0.75).

Here, in motion frames (the motion command's fractional ``phase``, see mdp._TimeScaledMotionCommand):
  - the phase advances by ``nominal_speed * v`` frames per policy step (``nominal_speed``: the motion_speed
    curriculum's per-env speed, 1 without it), and never goes before the first frame of the env's motion;
  - the paper's task-level tracking terms (base position, base orientation, object) become the climbing ones: the
    balls' positions (mean over the four balls of the squared error), the trunk position and the trunk orientation.
    Each factor is written with a tolerance: min(k ** (1 - e / tol^2), 1), i.e. the paper's form with
    lambda * w = ln(k) / tol^2: 1 while the error is within ``tol``, 0.1 at 1.41 tol, 0.01 at 1.73 tol;
  - the climbs are ~100 s long and the episodes 10 s, so the phase lag of the progress reward is measured in
    seconds (exp(-(lag / lag_std)^2)) rather than as a fraction of the whole demonstration.

Deployment note: v_ref needs the trunk pose and ball positions relative to the reference (state estimation), which
the rest of the climbing task avoids. ``rate_terms`` selects the factors; ("joint_pos", "trunk_ori") only uses what
the robot measures itself (encoders and IMU).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude

from mjlab_microban.tasks.climbing.mdp import (
    _PHASE_EPS,
    _TimeScaledMotionCommand,
    _ball_site_offsets,
    _motion_first_frame as motion_first_frame,
    _motion_last_frame,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

RATE_TERMS = ("balls", "trunk_pos", "trunk_ori", "joint_pos")


def motion_num_frames(command: MotionCommand) -> torch.Tensor:
    """Frames of each env's motion, shape [B]."""
    return _motion_last_frame(command) - motion_first_frame(command) + 1


# ---------------------------------------------------------------------------------------------------------
# Adaptive phase motion command
# ---------------------------------------------------------------------------------------------------------


class _AdaptivePhaseMotionCommand(_TimeScaledMotionCommand):
    """Mixin put in front of a motion command's class by enable_adaptive_phase.

    ``speed`` (frames per env step) is now the phase's actual rate ``nominal_speed * v`` of the last step: it scales
    the reference velocities and the look-ahead of the observations / residual actions as before.
    ``nominal_phase`` advances at ``nominal_speed`` (the paper's phi_nom), ``phase_init`` is the phase the env was
    (re)started at, ``phase_rate_ref`` is v_ref and ``phase_rate`` is v, both of the last step.
    """

    nominal_speed: torch.Tensor
    nominal_phase: torch.Tensor
    phase_init: torch.Tensor
    phase_rate_ref: torch.Tensor
    phase_rate: torch.Tensor
    phase_cfg: PhaseRateActionCfg
    _phase_action: PhaseRateAction | None

    # -- the reference phase rate v_ref -------------------------------------------------------------------

    def _setup_phase_rate_ref(self) -> None:
        cfg = self.phase_cfg
        unknown = set(cfg.rate_terms) - set(RATE_TERMS)
        if unknown:
            raise ValueError(f"unknown adaptive phase rate terms {sorted(unknown)}, expected some of {RATE_TERMS}")
        self._robot: Entity = self._env.scene[cfg.entity_name]
        if "balls" in cfg.rate_terms:
            self._ball_body_idx, self._ball_offsets = _ball_site_offsets(
                self._env, self, tuple(cfg.ball_site_names), cfg.entity_name
            )
            self._ball_site_ids = self._robot.find_sites(tuple(cfg.ball_site_names), preserve_order=True)[0]

    def rate_errors(self) -> dict[str, torch.Tensor]:
        """Squared tracking errors of the rate terms, against the current reference frame, each [B]."""
        errors = {}
        terms = self.phase_cfg.rate_terms
        if "balls" in terms:
            body_pos = self.body_pos_w[:, self._ball_body_idx]
            body_quat = self.body_quat_w[:, self._ball_body_idx]
            ref = body_pos + quat_apply(body_quat, self._ball_offsets.expand(body_pos.shape[0], -1, -1))
            err = torch.sum(torch.square(ref - self._robot.data.site_pos_w[:, self._ball_site_ids]), dim=-1)
            errors["balls"] = err.mean(dim=-1)
        if "trunk_pos" in terms:
            errors["trunk_pos"] = torch.sum(torch.square(self.anchor_pos_w - self.robot_anchor_pos_w), dim=-1)
        if "trunk_ori" in terms:
            errors["trunk_ori"] = torch.square(quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w))
        if "joint_pos" in terms:
            errors["joint_pos"] = torch.mean(torch.square(self.joint_pos - self.robot_joint_pos), dim=-1)
        return errors

    def compute_phase_rate_ref(self) -> torch.Tensor:
        """v_ref = prod_i min(k ** (1 - e_i / tol_i^2), 1), shape [B], from the current state and reference."""
        cfg = self.phase_cfg
        log_k = math.log(cfg.k)
        v = torch.ones(self.num_envs, device=self.device)
        for name, err in self.rate_errors().items():
            tol = cfg.tolerances[name]
            v = v * torch.clamp(torch.exp(log_k * (1.0 - err / tol**2)), max=1.0)
        return torch.nan_to_num(v, nan=0.0)

    # -- phase dynamics -------------------------------------------------------------------------------------

    def _restart_phase(self, env_ids: torch.Tensor | slice) -> None:
        self.phase[env_ids] = self.time_steps[env_ids].float()
        self.nominal_phase[env_ids] = self.phase[env_ids]
        self.phase_init[env_ids] = self.phase[env_ids]

    def _resample_command(self, env_ids: torch.Tensor):
        # The reference state written to the sim gets the nominal speed's velocities.
        self.speed[env_ids] = self.nominal_speed[env_ids]
        super()._resample_command(env_ids)
        self._restart_phase(env_ids)

    def _update_command(self):
        # Phase set elsewhere (the viewer's scrubber, reset_to_frame): restart there.
        frac = self.phase - self.time_steps.float()
        out_of_sync = ~((frac >= -_PHASE_EPS) & (frac < 1.0))
        if torch.any(out_of_sync):
            self._restart_phase(out_of_sync.nonzero(as_tuple=False).squeeze(-1))

        if self._phase_action is None:
            self._phase_action = cast(PhaseRateAction, self._env.action_manager.get_term(self.phase_cfg.action_name))
        self.phase_rate_ref = self.compute_phase_rate_ref()
        self.phase_rate = self.phase_rate_ref + self.phase_cfg.sigma * self._phase_action.delta
        # Not before the first frame of the motion (the paper's phase lives in [0, 1]); the end of the motion is
        # handled by the base class (resampling).
        speed = self.nominal_speed * self.phase_rate
        self.speed = torch.maximum(speed, motion_first_frame(self).float() - self.phase)
        self.nominal_phase = self.nominal_phase + self.nominal_speed
        super()._update_command()

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["phase_rate_ref"] = self.phase_rate_ref
        self.metrics["phase_rate"] = self.phase_rate
        self.metrics["phase_lag_s"] = self.phase_lag_s()

    # -- helpers --------------------------------------------------------------------------------------------

    def normalized_phase(self) -> torch.Tensor:
        """Phase in [0, 1] over the env's motion, [B]."""
        first = motion_first_frame(self).float()
        return (self.phase - first) / (motion_num_frames(self).float() - 1.0).clamp(min=1.0)

    def phase_lag_s(self) -> torch.Tensor:
        """How far the phase is behind the nominal phase, in seconds of motion (negative: ahead), [B]."""
        return (self.nominal_phase - self.phase) * self._env.step_dt

    def phase_progress(self) -> torch.Tensor:
        """The paper's p_phi = (phi - phi_init) / (phi_nom - phi_init), [B] (1 where no time has passed yet)."""
        done = self.nominal_phase - self.phase_init
        return torch.where(done > _PHASE_EPS, (self.phase - self.phase_init) / done.clamp(min=_PHASE_EPS), 1.0)


def enable_adaptive_phase(command: MotionCommand, cfg: PhaseRateActionCfg) -> MotionCommand:
    """Make a motion command (MotionCommand or a subclass, time-scaled or not) follow the adaptive phase."""
    if isinstance(command, _AdaptivePhaseMotionCommand):
        return command
    if isinstance(command, _TimeScaledMotionCommand):
        nominal = command.speed.clone()
    else:
        nominal = torch.ones(command.num_envs, device=command.device)
        command.phase = command.time_steps.float()
    cls = type(command)
    command.__class__ = type(f"AdaptivePhase{cls.__name__}", (_AdaptivePhaseMotionCommand, cls), {})
    command = cast(_AdaptivePhaseMotionCommand, command)
    command.phase_cfg = cfg
    command._phase_action = None
    command.nominal_speed = nominal
    command.speed = nominal.clone()
    command.nominal_phase = command.phase.clone()
    command.phase_init = command.phase.clone()
    command.phase_rate_ref = torch.ones(command.num_envs, device=command.device)
    command.phase_rate = torch.ones(command.num_envs, device=command.device)
    command._setup_phase_rate_ref()
    return command


def adaptive_phase_command(env: ManagerBasedRlEnv, command_name: str) -> _AdaptivePhaseMotionCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, _AdaptivePhaseMotionCommand):
        raise TypeError(f"command {command_name!r} has no adaptive phase (add a PhaseRateActionCfg action term)")
    return command


# ---------------------------------------------------------------------------------------------------------
# Action: the phase rate residual dv
# ---------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class PhaseRateActionCfg(ActionTermCfg):
    """One extra action, the residual dv on the phase rate: v = v_ref + sigma * dv. Building it turns the motion
    command ``command_name`` into an adaptive phase one (enable_adaptive_phase), with these parameters."""

    entity_name: str = "robot"
    command_name: str = "motion"
    action_name: str = "phase_rate"
    """This term's name in the env's actions (the command reads dv from it)."""
    sigma: float = 0.01
    """sigma_1: scale of dv on the phase rate."""
    k: float = 10.0
    """Each v_ref factor is min(k ** (1 - e / tol^2), 1)."""
    rate_terms: tuple[str, ...] = ("balls", "trunk_pos", "trunk_ori")
    tolerances: dict[str, float] | None = None
    """Tolerance of each rate term (m for balls / trunk_pos, rad for trunk_ori / joint_pos, RMS over joints)."""
    ball_site_names: tuple[str, ...] = ()

    def __post_init__(self):
        defaults = {"balls": 0.025, "trunk_pos": 0.04, "trunk_ori": 0.3, "joint_pos": 0.15}
        self.tolerances = {**defaults, **(self.tolerances or {})}
        if "balls" in self.rate_terms and not self.ball_site_names:
            raise ValueError("the 'balls' rate term needs ball_site_names")

    def build(self, env: ManagerBasedRlEnv) -> PhaseRateAction:
        return PhaseRateAction(self, env)


class PhaseRateAction(ActionTerm):
    cfg: PhaseRateActionCfg

    def __init__(self, cfg: PhaseRateActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._raw = torch.zeros(self.num_envs, 1, device=self.device)
        # The command manager is built before the action manager.
        enable_adaptive_phase(env.command_manager.get_term(cfg.command_name), cfg)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    @property
    def delta(self) -> torch.Tensor:
        """dv, [B]."""
        return self._raw[:, 0]

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions

    def apply_actions(self) -> None:
        pass  # read by the motion command when it advances the phase

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw[slice(None) if env_ids is None else env_ids] = 0.0


# ---------------------------------------------------------------------------------------------------------
# Observation, reward, curriculum
# ---------------------------------------------------------------------------------------------------------


def adaptive_phase_obs(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    """(phi, v_ref), [B, 2]: the normalised phase and the reference phase rate (the paper's phase observations; the
    nominal phase is not observed)."""
    command = adaptive_phase_command(env, command_name)
    return torch.stack([command.normalized_phase(), command.phase_rate_ref], dim=-1)


def phase_progress_reward(env: ManagerBasedRlEnv, command_name: str = "motion", lag_std_s: float = 2.0) -> torch.Tensor:
    """The paper's task progress reward v_ref * exp(-(phase lag / lag_std)^2), with v_ref of the current state."""
    command = adaptive_phase_command(env, command_name)
    v_ref = command.compute_phase_rate_ref()
    return v_ref * torch.exp(-torch.square(command.phase_lag_s() / lag_std_s))


class adaptive_phase_level:
    """The paper's curriculum: a global level l in [0, 1] scaling the weights of the reward terms ``reward_terms``
    (the regularisation penalties) and the motion command's initial pose / velocity randomisation. Every
    ``update_steps`` env steps, the mean phase progress p_phi of the episodes that ended meanwhile moves it by
    ``step``: up above ``up_threshold``, down below ``down_threshold``."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        p = cfg.params
        command = adaptive_phase_command(env, p["command_name"])
        self.level = float(p.get("init_level", 0.0))
        self._base_weights = {n: env.reward_manager.get_term_cfg(n).weight for n in p["reward_terms"]}
        self._base_pose_range = dict(command.cfg.pose_range)
        self._base_velocity_range = dict(command.cfg.velocity_range)
        self._sum = 0.0
        self._count = 0
        self._next_update = int(p.get("update_steps", 24))
        self._last_progress = float("nan")
        self._apply(env, command)

    def _apply(self, env: ManagerBasedRlEnv, command: MotionCommand) -> None:
        lv = self.level
        for name, w in self._base_weights.items():
            env.reward_manager.get_term_cfg(name).weight = w * lv
        scale = lambda r: {k: (lo * lv, hi * lv) for k, (lo, hi) in r.items()}  # noqa: E731
        command.cfg.pose_range = scale(self._base_pose_range)
        command.cfg.velocity_range = scale(self._base_velocity_range)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice,
        command_name: str,
        reward_terms: tuple[str, ...],
        init_level: float = 0.0,
        step: float = 0.25,
        up_threshold: float = 0.95,
        down_threshold: float = 0.75,
        update_steps: int = 24,
    ) -> dict[str, float]:
        del reward_terms, init_level
        command = adaptive_phase_command(env, command_name)
        if not isinstance(env_ids, slice):
            ran = env.episode_length_buf[env_ids] > 0  # not the env's first reset
            if torch.any(ran):
                progress = command.phase_progress()[env_ids][ran]
                self._sum += float(progress.clamp(-1.0, 2.0).sum())
                self._count += int(ran.sum())
        if env.common_step_counter >= self._next_update:
            self._next_update = env.common_step_counter + update_steps
            if self._count > 0:
                self._last_progress = self._sum / self._count
                if self._last_progress > up_threshold:
                    self.level = min(self.level + step, 1.0)
                elif self._last_progress < down_threshold:
                    self.level = max(self.level - step, 0.0)
                self._apply(env, command)
            self._sum, self._count = 0.0, 0
        result = {"level": self.level}
        if not math.isnan(self._last_progress):
            result["phase_progress"] = self._last_progress
        return result
