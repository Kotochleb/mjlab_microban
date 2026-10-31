# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban climbing, Conv1d variant: dense 1 s windows of the future reference, encoded by 1D convolutions.

Same environment as microban_climbing_env_cfg.py, except for the future reference observations, which move out
of the flat actor / critic groups into their own groups, dense (every policy step) over the next second:
  - ``actor_future``: future reference joint positions (the actor's "command" term), frames t+1 .. t+50;
  - ``critic_future``: future anchor pose, ball positions and joint positions (the critic's "motion_future"
    term), frames t+1 .. t+50.
They are frame-major flat windows, which the RL config's FutureConv1dModel (future_conv_model.py) reshapes to
[B, C, T] and encodes with Conv1d layers. The critic's contact schedule keeps its sparse offsets.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from mjlab_microban.robot.microban_climb_constants import BALL_SITE_NAMES
from mjlab_microban.tasks.climbing import mdp
from mjlab_microban.tasks.climbing.microban_climbing_env_cfg import make_microban_climbing_env_cfg

# Horizon of the dense future windows.
FUTURE_HORIZON_S = 1.0

ACTOR_FUTURE_GROUP = "actor_future"
CRITIC_FUTURE_GROUP = "critic_future"


def future_conv_num_frames(cfg: ManagerBasedRlEnvCfg) -> int:
    """Policy steps in FUTURE_HORIZON_S (50 at 5 ms x decimation 4)."""
    return round(FUTURE_HORIZON_S / (cfg.sim.mujoco.timestep * cfg.decimation))


def make_microban_climbing_conv1d_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
    """kwargs: forwarded to make_microban_climbing_env_cfg (wall, motion_file)."""
    cfg = make_microban_climbing_env_cfg(play=play, **kwargs)
    num_frames = future_conv_num_frames(cfg)
    assert num_frames == FUTURE_CONV_FRAMES, (num_frames, FUTURE_CONV_FRAMES)

    # Actor: the future joint positions move to their own group, every step over the next second.
    del cfg.observations["actor"].terms["command"]
    cfg.observations[ACTOR_FUTURE_GROUP] = ObservationGroupCfg(
        terms={
            "command": ObservationTermCfg(
                func=mdp.motion_future_joint_pos,
                params={"command_name": "motion", "num_steps": num_frames},
            )
        },
        concatenate_terms=True,
        enable_corruption=not play,
    )

    # Critic: the future reference frames likewise (the contact schedule keeps its sparse offsets).
    del cfg.observations["critic"].terms["motion_future"]
    cfg.observations[CRITIC_FUTURE_GROUP] = ObservationGroupCfg(
        terms={
            "motion_future": ObservationTermCfg(
                func=mdp.motion_future_window,
                params={
                    "command_name": "motion",
                    "site_names": BALL_SITE_NAMES,
                    "offsets": tuple(range(1, num_frames + 1)),
                },
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )
    return cfg


# Frames of the future windows, as the RL config needs it statically (checked against the env above).
FUTURE_CONV_FRAMES = 50
