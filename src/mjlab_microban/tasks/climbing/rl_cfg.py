# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban climbing RL configuration (copied from tasks/tracking/rl_cfg.py)"""

from copy import deepcopy

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


MicrobanClimbingRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microban_climbing",
    experiment_name="mjlab_microban_climbing",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
)

# Residual variant: the action scale is 0.25 * effort_limit / stiffness (~0.47 rad at kp 120); an
# initial std of 0.5 gives ~0.25 rad of exploration noise around the reference instead of swamping it.
MicrobanClimbingResidualRlCfg = deepcopy(MicrobanClimbingRlCfg)
MicrobanClimbingResidualRlCfg.actor.distribution_cfg["init_std"] = 0.5
MicrobanClimbingResidualRlCfg.experiment_name = "mjlab_microban_climbing_residual"

# Conv1d variant (microban_climbing_conv1d_env_cfg.py): the dense 1 s future windows (actor_future /
# critic_future groups, [B, 50 * C]) go through a Conv1d encoder (future_conv_model.py) before the MLP head.
# 50 frames -> (k5, s2) 23 -> (k5, s2) 10 -> (k3, s1) 8, x 32 channels: 256 features per window.
_FUTURE_CONV1D_MODEL = "mjlab_microban.tasks.climbing.future_conv_model:FutureConv1dModel"
_FUTURE_CONV1D_LAYERS = {"channels": (32, 32, 32), "kernel_size": (5, 5, 3), "stride": (2, 2, 1)}


def _with_future_conv1d(base: RslRlOnPolicyRunnerCfg, experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    from mjlab_microban.tasks.climbing.microban_climbing_conv1d_env_cfg import (
        ACTOR_FUTURE_GROUP,
        CRITIC_FUTURE_GROUP,
        FUTURE_CONV_FRAMES,
    )

    cfg = deepcopy(base)
    cfg.obs_groups = {"actor": ("actor", ACTOR_FUTURE_GROUP), "critic": ("critic", CRITIC_FUTURE_GROUP)}
    for model, group in ((cfg.actor, ACTOR_FUTURE_GROUP), (cfg.critic, CRITIC_FUTURE_GROUP)):
        model.class_name = _FUTURE_CONV1D_MODEL
        model.cnn_cfg = {"future_group": group, "num_frames": FUTURE_CONV_FRAMES, **_FUTURE_CONV1D_LAYERS}
    cfg.experiment_name = experiment_name
    return cfg


MicrobanClimbingConv1dRlCfg = _with_future_conv1d(MicrobanClimbingRlCfg, "mjlab_microban_climbing_conv1d")
MicrobanClimbingResidualConv1dRlCfg = _with_future_conv1d(
    MicrobanClimbingResidualRlCfg, "mjlab_microban_climbing_residual_conv1d"
)
