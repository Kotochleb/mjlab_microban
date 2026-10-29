# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban motion tracking environment (BeyondMimic-style)"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microban.robot.microban_constants import MICROBAN_ROBOT_CFG

# Microban is roughly 4x smaller than the G1 the base tracking config was tuned
# for, so position-based stds, thresholds and ranges are scaled down accordingly.

# Note that the Onshape export names the right foot "foot" and the left one
# "foot_2" (same for the forearms "radius"/"radius_2", used as hands).
FEET_BODY_NAMES = ("foot_2", "foot")
HANDS_BODY_NAMES = ("radius_2", "radius")
END_EFFECTOR_BODY_NAMES = FEET_BODY_NAMES + HANDS_BODY_NAMES

# The anchor (trunk) has to be part of the motion command bodies. The policy
# tracks the position of the feet and hands, and the orientation of the head.
TRACKED_BODY_NAMES = ("trunk",) + END_EFFECTOR_BODY_NAMES + ("head",)


def make_microban_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_env_cfg()

    # The default contact/constraint caps are sized for the G1; Microban's
    # multi-box feet produce many more contacts, so let mujoco_warp size them.
    cfg.sim.nconmax = None
    cfg.sim.njmax = None

    # Larger physics timestep (0.005 -> 0.01) with half the decimation, so the
    # policy still runs at 50 Hz with half the physics steps.
    cfg.sim.mujoco.timestep = 0.01
    cfg.decimation = 2

    # Actuator delays are counted in physics steps: 2-3 steps of 10 ms keep the
    # latency close to the 3-6 steps of 5 ms used by the velocity task.
    robot_cfg = deepcopy(MICROBAN_ROBOT_CFG)
    for actuator in robot_cfg.articulation.actuators:
        actuator.delay_min_lag = 2
        actuator.delay_max_lag = 3
    cfg.scene.entities = {"robot": robot_cfg}

    #---------------------------- Sensors ---------------------------
    self_collision_sensor_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_sensor_cfg,)

    #---------------------------- Actions ---------------------------
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    #---------------------------- Commands --------------------------
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "trunk"
    motion_cmd.body_names = TRACKED_BODY_NAMES
    motion_cmd.pose_range = {
        "x": (-0.015, 0.015),
        "y": (-0.015, 0.015),
        "z": (-0.005, 0.005),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
    }
    motion_cmd.velocity_range = {
        "x": (-0.3, 0.3),
        "y": (-0.3, 0.3),
        "z": (-0.1, 0.1),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
    }

    #---------------------------- Observations ----------------------
    cfg.observations["actor"].terms["motion_anchor_pos_b"].noise = Unoise(n_min=-0.06, n_max=0.06)
    cfg.observations["actor"].terms["base_lin_vel"].noise = Unoise(n_min=-0.15, n_max=0.15)

    #---------------------------- Rewards ---------------------------
    cfg.rewards["motion_global_root_pos"].params["std"] = 0.08
    cfg.rewards["motion_body_pos"].params["std"] = 0.08
    cfg.rewards["motion_body_pos"].params["body_names"] = END_EFFECTOR_BODY_NAMES
    cfg.rewards["motion_body_lin_vel"].params["std"] = 0.3
    cfg.rewards["motion_body_lin_vel"].params["body_names"] = END_EFFECTOR_BODY_NAMES

    cfg.rewards["motion_body_ori"].params["body_names"] = ("head",)
    cfg.rewards["motion_body_ang_vel"].params["body_names"] = ("head",)

    cfg.rewards["self_collisions"].params["force_threshold"] = 1.0

    #---------------------------- Events ----------------------------
    del cfg.events["push_robot"]

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        r"^(left|right)_foot_collision_\d+$",
    )

    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.005, 0.005),
        1: (-0.005, 0.005),
        2: (-0.005, 0.005),
    }

    cfg.events["dof_armature_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )

    cfg.events["dof_friction_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )

    #---------------------------- Terminations ----------------------
    cfg.terminations["anchor_pos"].params["threshold"] = 0.07
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.07
    cfg.terminations["ee_body_pos"].params["body_names"] = END_EFFECTOR_BODY_NAMES

    #---------------------------- Viewer ----------------------------
    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 1.0

    #---------------------------- State estimation ------------------
    # Without a state estimator, the policy has no access to the global anchor
    # position nor to the base linear velocity.
    if not has_state_estimation:
        cfg.observations["actor"] = ObservationGroupCfg(
            terms={
                k: v
                for k, v in cfg.observations["actor"].terms.items()
                if k not in ["motion_anchor_pos_b", "base_lin_vel"]
            },
            concatenate_terms=True,
            enable_corruption=True,
        )

    #---------------------------- Play mode -------------------------
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False

        # Disable RSI randomization.
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}

        motion_cmd.sampling_mode = "start"

    return cfg
