# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban climbing with adaptive phase dynamics (Sleiman, Mittal & Hutter, CoRL 2024, arXiv:2410.13817; see
adaptive_phase.py): any climbing env config (single board, Conv1d, Conv1d-History, multi-board) where the planned
climb's phase advances at a rate set by how well the robot tracks it, plus a policy residual on that rate.

On top of the base env:
  - actions: one more, the phase rate residual dv (last in the action vector), v = v_ref + 0.01 * dv;
  - observations: the phase phi (normalised over the climb) and v_ref, for actor and critic; the critic also gets
    the phase lag behind the nominal phase (the progress reward depends on it);
  - rewards: the task progress reward v_ref * exp(-(lag / 2 s)^2);
  - curriculum: the paper's level l in [0, 1], moved by the phase progress of the ended episodes, scaling the
    regularisation penalties (residual, action rate, torques, torque barrier) and the initial state randomisation.
The tracking rewards, terminations and domain randomisation are the base env's: they compare the robot to the
adaptive reference.
"""

from collections.abc import Callable

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microban.robot.microban_climb_constants import BALL_SITE_NAMES
from mjlab_microban.tasks.climbing import adaptive_phase
from mjlab_microban.tasks.climbing.microban_climbing_env_cfg import make_microban_climbing_env_cfg

# sigma_1 and k of the paper.
PHASE_RATE_SCALE = 0.01
PHASE_RATE_K = 10.0

# Tracking tolerances of the reference phase rate's factors (each is 1 within its tolerance, 0.1 at 1.41 x it, 0.01
# at 1.73 x it): the phase stops well before the 7 cm terminations, and the reset randomisation (+-1.5 cm) leaves it at 1.
PHASE_RATE_TOLERANCES = {"balls": 0.025, "trunk_pos": 0.04, "trunk_ori": 0.3}

# Task progress reward: weight (the tracking rewards sum to ~10) and phase lag kernel width.
PROGRESS_WEIGHT = 5.0
PROGRESS_LAG_STD_S = 2.0

# Curriculum level: regularisation terms it scales, and the paper's update rule.
CURRICULUM_REWARD_TERMS = ("residual_l2", "action_rate_l2", "joint_torques_l2", "torque_barrier")
CURRICULUM_INIT_LEVEL = 0.0
CURRICULUM_STEP = 0.25
CURRICULUM_UP = 0.95
CURRICULUM_DOWN = 0.75
CURRICULUM_UPDATE_STEPS = 24  # one PPO iteration (rl_cfg num_steps_per_env)


def to_adaptive_phase(cfg: ManagerBasedRlEnvCfg, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Add the adaptive phase to a climbing env config (in place, returned)."""
    cfg.actions["phase_rate"] = adaptive_phase.PhaseRateActionCfg(
        command_name="motion",
        action_name="phase_rate",
        sigma=PHASE_RATE_SCALE,
        k=PHASE_RATE_K,
        rate_terms=tuple(PHASE_RATE_TOLERANCES),
        tolerances=dict(PHASE_RATE_TOLERANCES),
        ball_site_names=BALL_SITE_NAMES,
    )

    cfg.observations["actor"].terms["phase"] = ObservationTermCfg(
        func=adaptive_phase.adaptive_phase_obs,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.005, n_max=0.005),
    )
    cfg.observations["critic"].terms["phase"] = ObservationTermCfg(
        func=adaptive_phase.adaptive_phase_obs, params={"command_name": "motion"}
    )
    cfg.observations["critic"].terms["phase_lag"] = ObservationTermCfg(func=_phase_lag_obs)

    cfg.rewards["phase_progress"] = RewardTermCfg(
        func=adaptive_phase.phase_progress_reward,
        weight=PROGRESS_WEIGHT,
        params={"command_name": "motion", "lag_std_s": PROGRESS_LAG_STD_S},
    )

    if not play:
        cfg.curriculum["adaptive_phase_level"] = CurriculumTermCfg(
            func=adaptive_phase.adaptive_phase_level,
            params={
                "command_name": "motion",
                "reward_terms": tuple(t for t in CURRICULUM_REWARD_TERMS if t in cfg.rewards),
                "init_level": CURRICULUM_INIT_LEVEL,
                "step": CURRICULUM_STEP,
                "up_threshold": CURRICULUM_UP,
                "down_threshold": CURRICULUM_DOWN,
                "update_steps": CURRICULUM_UPDATE_STEPS,
            },
        )
    return cfg


def _phase_lag_obs(env, command_name: str = "motion"):
    return adaptive_phase.adaptive_phase_command(env, command_name).phase_lag_s().unsqueeze(-1)


def adaptive_phase_base_fn(
    base_fn: Callable[..., ManagerBasedRlEnvCfg] = make_microban_climbing_env_cfg,
) -> Callable[..., ManagerBasedRlEnvCfg]:
    """The env config maker ``base_fn`` (make_microban_climbing[_conv1d[_history]]_env_cfg, same arguments), with
    the adaptive phase. Can itself be the multi-board config's base_fn."""

    def make(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
        return to_adaptive_phase(base_fn(play=play, **kwargs), play=play)

    return make


def make_microban_climbing_adaptive_phase_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
    return adaptive_phase_base_fn()(play=play, **kwargs)
