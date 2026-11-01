# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Fine-tuning of a trained climbing policy (any climbing env config: single board or multi-board, with or without the
adaptive phase), resumed from its checkpoint (scripts/train_climbing.sh --finetune).

On top of the base env:
  - holds: the used holds collide as the legacy 5-box cup (wall.FIVE_BOX_CUPS["five_box_legacy"], the cup before git
    20a0573: five upright boxes, no trims, tilts or posts) instead of the current one;
  - curricula: the step-scheduled ones at their final values from the first reset (the env's step counter is only
    restored from the checkpoint after the env is built): full actuator, ball position and motion speed DR; nominal
    ball radius; realistic motor power. Multi-board: envs spread over all board levels, board horizons at the
    maximum episode length. Their own state is not in the checkpoint, so it starts there; the board levels still
    move with the episodes' outcomes;
  - adaptive phase level (scales the regularisation penalties: residual, action rate, torques, torque barrier): for
    more exploration, it starts at FINETUNE_PHASE_LEVEL_INIT instead of 1, and moves by FINETUNE_PHASE_LEVEL_STEP
    (instead of 0.25) per update, every FINETUNE_PHASE_LEVEL_UPDATE_STEPS (10 iterations, instead of 1), up only while the mean phase progress is above FINETUNE_PHASE_LEVEL_UP (instead
    of 0.95), so the penalties come back slowly and only once the policy climbs well on the new cups;
  - policy std: reset to FINETUNE_INIT_STD on every action after the checkpoint is loaded (FinetuneRunner; the
    trained policy's std is ~0.05-0.1), for exploration;
  - firmware P gain: KP * U(1 - KP_REL_RANGE, 1 + KP_REL_RANGE) per env and joint, instead of KP everywhere; the
    residual action scale keeps the nominal KP (mdp.randomize_pd_gain).
"""

import math

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_microban.tasks.climbing import mdp
from mjlab_microban.tasks.climbing.microban_climbing_env_cfg import KP
from mjlab_microban.tasks.climbing.wall import with_five_box_cup

FINETUNE_FIVE_BOX_CUP = "five_box_legacy"
KP_REL_RANGE = 0.10
FINETUNE_INIT_STD = 0.45
FINETUNE_SAVE_INTERVAL = 100  # PPO iterations between saved checkpoints (model_<it>.pt)
FINETUNE_PHASE_LEVEL_INIT = 0.5
FINETUNE_PHASE_LEVEL_STEP = 0.125
FINETUNE_PHASE_LEVEL_UP = 0.97
FINETUNE_PHASE_LEVEL_UPDATE_STEPS = 10 * 24  # every 10 PPO iterations: at least 40 iterations from 0.5 to 1

# _dr_ramp(step, hold_steps=-1, full_steps=0) = 1 for every step >= 0: the DR curricula at their full ranges.
_RAMP_DONE = {"hold_steps": -1, "full_steps": 0}
# multi_board.board_levels draws each env's start level in 0..max_init_level (clamped to the number of levels).
_ALL_BOARD_LEVELS = 1_000_000


def to_finetune(cfg: ManagerBasedRlEnvCfg, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Turn a climbing env config into its fine-tuning one (in place, returned)."""
    cfg.scene.entities["wall"] = with_five_box_cup(cfg.scene.entities["wall"], FINETUNE_FIVE_BOX_CUP)

    if not play:
        pd_gains = cfg.events["pd_gains"]
        assert pd_gains.func is mdp.set_pd_gain and pd_gains.params["kp"] == KP
        cfg.events["pd_gains"] = EventTermCfg(
            func=mdp.randomize_pd_gain, mode="startup", params={"kp": KP, "rel_range": KP_REL_RANGE}
        )

    curriculum = cfg.curriculum
    for name in ("actuator_dr", "ball_pos_dr", "ball_radius", "motion_speed"):
        if name in curriculum:
            curriculum[name].params.update(_RAMP_DONE)
    if "ball_radius" in curriculum:
        curriculum["ball_radius"].params["undersize"] = 0.0
    if "motion_speed" in curriculum:
        params = curriculum["motion_speed"].params
        params["start_range"] = params["full_range"]
    if "bam_power" in curriculum:
        curriculum["bam_power"].params["num_steps"] = 0
    if "adaptive_phase_level" in curriculum:
        curriculum["adaptive_phase_level"].params.update(
            init_level=FINETUNE_PHASE_LEVEL_INIT,
            step=FINETUNE_PHASE_LEVEL_STEP,
            up_threshold=FINETUNE_PHASE_LEVEL_UP,
            update_steps=FINETUNE_PHASE_LEVEL_UPDATE_STEPS,
        )
    if "board_levels" in curriculum:
        curriculum["board_levels"].params["max_init_level"] = _ALL_BOARD_LEVELS
    if "board_horizons" in curriculum:
        params = curriculum["board_horizons"].params
        params["init_s"] = params["max_s"]
    return cfg


class FinetuneRunner(MotionTrackingOnPolicyRunner):
    """The tracking runner, whose policy std is reset to FINETUNE_INIT_STD on every action once a checkpoint is loaded
    (the resumed policy keeps its means; the optimiser's state is kept too). Play loads through it as well, but plays
    the deterministic policy (the mean), so the std does not matter there."""

    def load(self, path: str, *args, **kwargs) -> dict:
        infos = super().load(path, *args, **kwargs)
        dist = self.alg.actor.distribution
        with torch.no_grad():
            if dist.std_type == "scalar":
                dist.std_param.fill_(FINETUNE_INIT_STD)
            else:
                dist.log_std_param.fill_(math.log(FINETUNE_INIT_STD))
        print(f"[INFO] Fine-tuning: policy std reset to {FINETUNE_INIT_STD}")
        return infos
