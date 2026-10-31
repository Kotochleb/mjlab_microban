# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_microban.tasks.climbing.microban_climbing_env_cfg import make_microban_climbing_env_cfg
from mjlab_microban.tasks.climbing.microban_climbing_conv1d_env_cfg import make_microban_climbing_conv1d_env_cfg
from mjlab_microban.tasks.climbing.microban_climbing_conv1d_history_env_cfg import (
    make_microban_climbing_conv1d_history_env_cfg,
)
from mjlab_microban.tasks.climbing.microban_climbing_multi_board_env_cfg import (
    make_microban_climbing_multi_board_env_cfg,
)
from mjlab_microban.tasks.climbing.rl_cfg import (
    MicrobanClimbingConv1dHistoryRlCfg,
    MicrobanClimbingConv1dRlCfg,
    MicrobanClimbingRlCfg,
)

# All climbing tasks use residual actions on the reference, and no state estimation.

# MLP: flat future reference frames and a short proprioceptive history.
register_mjlab_task(
    task_id="Mjlab-Climbing-Microban",
    env_cfg=make_microban_climbing_env_cfg(),
    play_env_cfg=make_microban_climbing_env_cfg(play=True),
    rl_cfg=MicrobanClimbingRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Conv1d: dense 1 s future reference windows, encoded by 1D convolutions.
register_mjlab_task(
    task_id="Mjlab-Climbing-Microban-Conv1d",
    env_cfg=make_microban_climbing_conv1d_env_cfg(),
    play_env_cfg=make_microban_climbing_conv1d_env_cfg(play=True),
    rl_cfg=MicrobanClimbingConv1dRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Conv1d + history: as Conv1d, plus a 0.5 s proprioceptive history encoded by 1D convolutions.
register_mjlab_task(
    task_id="Mjlab-Climbing-Microban-Conv1d-History",
    env_cfg=make_microban_climbing_conv1d_history_env_cfg(),
    play_env_cfg=make_microban_climbing_conv1d_history_env_cfg(play=True),
    rl_cfg=MicrobanClimbingConv1dHistoryRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Multi-board: the variants above on several boards at once (one level 1 board, nine level 2 boards by default,
# from tasks/climbing/boards/multi_board.json or $MICROBAN_BOARDS_FILE), with a curriculum over the board levels.
for _suffix, _base_fn, _rl_cfg in (
    ("", make_microban_climbing_env_cfg, MicrobanClimbingRlCfg),
    ("-Conv1d", make_microban_climbing_conv1d_env_cfg, MicrobanClimbingConv1dRlCfg),
    ("-Conv1d-History", make_microban_climbing_conv1d_history_env_cfg, MicrobanClimbingConv1dHistoryRlCfg),
):
    try:
        _env_cfg = make_microban_climbing_multi_board_env_cfg(base_fn=_base_fn)
        _play_env_cfg = make_microban_climbing_multi_board_env_cfg(play=True, base_fn=_base_fn)
    except (OSError, ValueError, KeyError) as _e:  # a bad board file only disables these tasks
        print(f"[WARN] Mjlab-Climbing-Microban-Multi-Board{_suffix} not registered: {_e}")
        continue
    register_mjlab_task(
        task_id=f"Mjlab-Climbing-Microban-Multi-Board{_suffix}",
        env_cfg=_env_cfg,
        play_env_cfg=_play_env_cfg,
        rl_cfg=replace(_rl_cfg, experiment_name=f"{_rl_cfg.experiment_name}_multi_board"),
        runner_cls=MotionTrackingOnPolicyRunner,
    )
