# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Multi-board climbing environment: the climbing env on several boards at once (multi_board.py), with a
curriculum over their difficulty levels in the style of the locomotion tasks' rough-terrain curriculum.

The boards and their seeds come from a JSON board file (multi_board.BOARDS_DIR / multi_board.json, or
$MICROBAN_BOARDS_FILE); by default one level 1 board (the easiest curriculum level) and nine level 2 boards.
Everything else (robot, observations, rewards, terminations, domain randomisation) is the single-board env's,
or that of the variant given as ``base_fn`` (Conv1d, Conv1d-History).
"""

from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from mjlab_microban.tasks.climbing import multi_board
from mjlab_microban.tasks.climbing.microban_climbing_env_cfg import make_microban_climbing_env_cfg
from mjlab_microban.tasks.climbing.multi_board import BoardSet, MultiBoardMotionCommandCfg

# Board curriculum (multi_board.board_levels): up a level after a successful full episode (to the end of its motion,
# or >= 90 % of its limit with its board's horizon at the maximum), down a level after one failing before 50 % of its
# limit; all envs start on the easiest level.
CURRICULUM_UP_FRACTION = 0.9
CURRICULUM_DOWN_FRACTION = 0.5
CURRICULUM_MAX_INIT_LEVEL = 0
# Per-board episode horizons and sampling bias (multi_board.board_horizons): from HORIZON_INIT_S, within
# [HORIZON_MIN_S, the maximum episode length]; x HORIZON_GROW per PPO iteration while the board's success rate is
# above HORIZON_UP_RATE, x HORIZON_SHRINK while below HORIZON_DOWN_RATE. Boards are sampled with weight
# 1 - success rate (at least HORIZON_MIN_BIAS).
HORIZON_MIN_S = 5.0
HORIZON_INIT_S = 10.0
HORIZON_UP_RATE = 0.8
HORIZON_DOWN_RATE = 0.5
HORIZON_GROW = 1.25
HORIZON_SHRINK = 0.8
HORIZON_EMA = 0.2
HORIZON_MIN_BIAS = 0.2
HORIZON_UPDATE_STEPS = 24  # one PPO iteration (rl_cfg num_steps_per_env)


def to_multi_board(cfg: ManagerBasedRlEnvCfg, board_set: BoardSet, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Turn a single-board climbing env config into the multi-board one."""
    # Scene: all the boards in the one wall entity, at the world origin.
    cfg.scene.entities["wall"] = multi_board.make_multi_board_wall_cfg(board_set)
    cfg.events["reset_wall"] = EventTermCfg(
        func=multi_board.reset_wall_to_world_origin, mode="reset", params={"asset_cfg": SceneEntityCfg("wall")}
    )

    # Motion: one per board.
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    level_names = tuple(f"{d}" for d in board_set.difficulties)
    cfg.commands["motion"] = MultiBoardMotionCommandCfg(
        **{f.name: getattr(motion_cmd, f.name) for f in fields(motion_cmd)},
        motion_files=tuple(b.motion_file for b in board_set.boards),
        board_offsets=board_set.offsets,
        board_levels=board_set.board_levels,
        level_names=level_names,
        board_weights=board_set.weights,
    )
    cfg.commands["motion"].motion_file = board_set.boards[0].motion_file

    # Critic (and actor): the contact schedule against each board's own holds.
    for group, name in (("critic", "motion_contact_schedule"), ("actor", "contact_schedule")):
        if name not in cfg.observations[group].terms:
            continue
        old = cfg.observations[group].terms[name]
        params = {k: v for k, v in old.params.items() if k != "hold_positions"}
        params["board_hold_positions"] = multi_board.board_hold_positions(board_set)
        cfg.observations[group].terms[name] = ObservationTermCfg(
            func=multi_board.multi_board_motion_contact_schedule, params=params
        )

    # Curriculum over the boards' levels, after the per-board horizons (which judge the episodes on the boards they
    # ended on, before board_levels moves the envs on).
    if not play:
        cfg.curriculum["board_horizons"] = CurriculumTermCfg(
            func=multi_board.board_horizons,
            params={
                "command_name": "motion",
                "min_s": HORIZON_MIN_S,
                "max_s": cfg.episode_length_s,
                "init_s": HORIZON_INIT_S,
                "up_rate": HORIZON_UP_RATE,
                "down_rate": HORIZON_DOWN_RATE,
                "grow": HORIZON_GROW,
                "shrink": HORIZON_SHRINK,
                "ema": HORIZON_EMA,
                "min_bias": HORIZON_MIN_BIAS,
                "update_steps": HORIZON_UPDATE_STEPS,
            },
        )
        # Episodes end at their limit (a time-out: not a failure).
        cfg.terminations["episode_limit"] = TerminationTermCfg(
            func=multi_board.episode_limit_reached, params={"command_name": "motion"}, time_out=True
        )
        cfg.curriculum["board_levels"] = CurriculumTermCfg(
            func=multi_board.board_levels,
            params={
                "command_name": "motion",
                "up_fraction": CURRICULUM_UP_FRACTION,
                "down_fraction": CURRICULUM_DOWN_FRACTION,
                "max_init_level": CURRICULUM_MAX_INIT_LEVEL,
            },
        )
    return cfg


def make_microban_climbing_multi_board_env_cfg(
    play: bool = False,
    boards_file: str | Path | None = None,
    base_fn: Callable[..., ManagerBasedRlEnvCfg] = make_microban_climbing_env_cfg,
) -> ManagerBasedRlEnvCfg:
    """boards_file: the JSON board file (default: $MICROBAN_BOARDS_FILE, or boards/multi_board.json).
    base_fn: the single-board env config maker (make_microban_climbing[_conv1d[_history]]_env_cfg)."""
    board_set = multi_board.load_board_set(boards_file)
    first = board_set.boards[0]
    # The single-board env on the first board (its robot start pose); its wall and motion are replaced.
    cfg = base_fn(play=play, wall=first.wall, motion_file=first.motion_file)
    return to_multi_board(cfg, board_set, play)
