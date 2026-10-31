# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_microban.tasks.climbing.microban_climbing_env_cfg import make_microban_climbing_env_cfg
from mjlab_microban.tasks.climbing.microban_climbing_conv1d_env_cfg import make_microban_climbing_conv1d_env_cfg
from mjlab_microban.tasks.climbing.microban_climbing_conv1d_history_env_cfg import (
    make_microban_climbing_conv1d_history_env_cfg,
)
from mjlab_microban.tasks.climbing.rl_cfg import (
    MicrobanClimbingConv1dHistoryRlCfg,
    MicrobanClimbingConv1dRlCfg,
    MicrobanClimbingResidualConv1dHistoryRlCfg,
    MicrobanClimbingResidualConv1dRlCfg,
    MicrobanClimbingResidualRlCfg,
    MicrobanClimbingRlCfg,
)

register_mjlab_task(
    task_id="Mjlab-Climbing-Microban",
    env_cfg=make_microban_climbing_env_cfg(),
    play_env_cfg=make_microban_climbing_env_cfg(play=True),
    rl_cfg=MicrobanClimbingRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Climbing-Microban-No-State-Estimation",
    env_cfg=make_microban_climbing_env_cfg(has_state_estimation=False),
    play_env_cfg=make_microban_climbing_env_cfg(has_state_estimation=False, play=True),
    rl_cfg=MicrobanClimbingRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Climbing-Microban-Residual",
    env_cfg=make_microban_climbing_env_cfg(residual=True),
    play_env_cfg=make_microban_climbing_env_cfg(residual=True, play=True),
    rl_cfg=MicrobanClimbingResidualRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Climbing-Microban-Residual-No-State-Estimation",
    env_cfg=make_microban_climbing_env_cfg(has_state_estimation=False, residual=True),
    play_env_cfg=make_microban_climbing_env_cfg(has_state_estimation=False, residual=True, play=True),
    rl_cfg=MicrobanClimbingResidualRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Conv1d variants: dense 1 s future reference windows, encoded by 1D convolutions.
register_mjlab_task(
    task_id="Mjlab-Climbing-Microban-Conv1d",
    env_cfg=make_microban_climbing_conv1d_env_cfg(),
    play_env_cfg=make_microban_climbing_conv1d_env_cfg(play=True),
    rl_cfg=MicrobanClimbingConv1dRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Climbing-Microban-Residual-Conv1d",
    env_cfg=make_microban_climbing_conv1d_env_cfg(residual=True),
    play_env_cfg=make_microban_climbing_conv1d_env_cfg(residual=True, play=True),
    rl_cfg=MicrobanClimbingResidualConv1dRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Conv1d + history variants: as the Conv1d ones, plus a 0.5 s proprioceptive history encoded by 1D convolutions.
register_mjlab_task(
    task_id="Mjlab-Climbing-Microban-Conv1d-History",
    env_cfg=make_microban_climbing_conv1d_history_env_cfg(),
    play_env_cfg=make_microban_climbing_conv1d_history_env_cfg(play=True),
    rl_cfg=MicrobanClimbingConv1dHistoryRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Climbing-Microban-Residual-Conv1d-History",
    env_cfg=make_microban_climbing_conv1d_history_env_cfg(residual=True),
    play_env_cfg=make_microban_climbing_conv1d_history_env_cfg(residual=True, play=True),
    rl_cfg=MicrobanClimbingResidualConv1dHistoryRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)
