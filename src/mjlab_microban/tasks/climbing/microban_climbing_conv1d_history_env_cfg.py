# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban climbing, Conv1d + history variant: as the Conv1d variant (microban_climbing_conv1d_env_cfg.py), and
the actor's proprioception becomes a dense 0.5 s history, also encoded by 1D convolutions.

The ``actor`` group (base angular velocity, projected gravity, heading, joint positions / velocities, last
actions) keeps every policy step of the last HISTORY_HORIZON_S (25 frames at 50 Hz, the current one included),
unflattened and frame-major: [B, 25, C], oldest frame first. The RL config's TemporalConv1dModel
(temporal_conv_model.py) encodes it with a Conv1d stack, and feeds its newest frame to the MLP head as is.
The critic is the Conv1d variant's.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microban.tasks.climbing.microban_climbing_conv1d_env_cfg import make_microban_climbing_conv1d_env_cfg

# Horizon of the actor's proprioceptive history (the current step included).
HISTORY_HORIZON_S = 0.5

ACTOR_HISTORY_GROUP = "actor"

# Frames of the history, as the RL config needs it statically (checked against the env below).
HISTORY_CONV_FRAMES = 25


def history_conv_num_frames(cfg: ManagerBasedRlEnvCfg) -> int:
    """Policy steps in HISTORY_HORIZON_S (25 at 5 ms x decimation 4)."""
    return round(HISTORY_HORIZON_S / (cfg.sim.mujoco.timestep * cfg.decimation))


def make_microban_climbing_conv1d_history_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
    """kwargs: forwarded to make_microban_climbing_env_cfg (wall, motion_file)."""
    cfg = make_microban_climbing_conv1d_env_cfg(play=play, **kwargs)
    num_frames = history_conv_num_frames(cfg)
    assert num_frames == HISTORY_CONV_FRAMES, (num_frames, HISTORY_CONV_FRAMES)

    # Group-level history override (replaces the terms' own 5-step history): every term keeps num_frames steps,
    # unflattened, so the concatenated group is [B, T, C] (terms concatenated per frame).
    group = cfg.observations[ACTOR_HISTORY_GROUP]
    assert "command" not in group.terms  # the future reference is in its own group (Conv1d variant)
    group.history_length = num_frames
    group.flatten_history_dim = False
    return cfg
