# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban climbing environment: the climbing Microban (ball hands and feet) on a simple wall.

The robot spawns in front of the wall (tasks/climbing/wall.py), hanging on its four start holds, and
tracks a planned climb (BeyondMimic-style, as the tracking task). The motion is made from
../planning/trajectory.npz by scripts/planning_to_npz.py.

Rewards: the policy tracks the four ball positions; the joint configuration and the base placement
only slightly; joint torques and action rate are penalised. The RL config is a copy of
tasks/tracking/rl_cfg.py.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microban.robot.microban_climb_constants import (
    BALL_SITE_NAMES,
    FEET_BODY_NAMES,
    HANDS_BODY_NAMES,
    MICROBAN_CLIMB_ROBOT_CFG,
)
from mjlab_microban.tasks.climbing import mdp
from mjlab_microban.tasks.climbing.wall import WALL_CFG

END_EFFECTOR_BODY_NAMES = FEET_BODY_NAMES + HANDS_BODY_NAMES

# Number of future motion frames (policy steps, 50 Hz) the policy observes, as a rolling window.
FUTURE_WINDOW_STEPS = 20

# Residual variant: action scale (rad per unit action) on top of the reference joint positions.
RESIDUAL_ACTION_SCALE = 0.25

# The anchor (trunk) has to be part of the motion command bodies. As in the tracking task, the policy
# tracks the position of the feet and hands, and the orientation of the head.
TRACKED_BODY_NAMES = ("trunk",) + END_EFFECTOR_BODY_NAMES + ("head",)


def make_microban_climbing_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
    residual: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_env_cfg()

    # Microban's contacts outgrow the G1-sized caps, let mujoco_warp size them.
    cfg.sim.nconmax = None
    cfg.sim.njmax = None

    # 5 ms physics timestep with a decimation of 4, so the policy runs at 50 Hz.
    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 4

    #---------------------------- Scene -----------------------------
    cfg.scene.entities = {
        "robot": deepcopy(MICROBAN_CLIMB_ROBOT_CFG),
        "wall": deepcopy(WALL_CFG),
    }
    cfg.scene.env_spacing = 2.0  # the wall is 0.8 m wide

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
    if residual:
        # The policy outputs corrections on the reference joint positions (of the frame the step ends
        # on); a small scale so that the untrained policy roughly plays back the reference.
        cfg.actions["joint_pos"] = mdp.ResidualJointPositionActionCfg(
            entity_name=joint_pos_action.entity_name,
            actuator_names=joint_pos_action.actuator_names,
            scale=RESIDUAL_ACTION_SCALE,
            command_name="motion",
        )
    else:
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

    # Rolling window over the next FUTURE_WINDOW_STEPS frames of the planned climb (anchor pose, ball
    # positions and joint positions), for both the actor and the critic.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["motion_future"] = ObservationTermCfg(
            func=mdp.motion_future_window,
            params={
                "command_name": "motion",
                "site_names": BALL_SITE_NAMES,
                "num_steps": FUTURE_WINDOW_STEPS,
            },
        )

    #---------------------------- Rewards ---------------------------
    # Climbing rewards: track the balls (what goes into the holds), very slightly the joint configuration
    # and the base placement, and minimise joint torques and action rate. Joint limits and self
    # collisions are kept as in the tracking task.
    cfg.rewards = {
        "ball_pos": RewardTermCfg(
            func=mdp.ball_position_error_exp,
            weight=5.0,
            params={"command_name": "motion", "std": 0.05, "site_names": BALL_SITE_NAMES},
        ),
        "joint_pos": RewardTermCfg(
            func=mdp.motion_joint_pos_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.5},
        ),
        "motion_global_root_pos": RewardTermCfg(
            func=tracking_mdp.motion_global_anchor_position_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.05},
        ),
        # The trunk height is tracked more tightly than the rest of the base placement.
        "motion_global_root_pos_z": RewardTermCfg(
            func=mdp.motion_global_anchor_position_z_error_exp,
            weight=3.0,
            params={"command_name": "motion", "std": 0.01},
        ),
        "motion_global_root_ori": RewardTermCfg(
            func=tracking_mdp.motion_global_anchor_orientation_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.7},
        ),
        "joint_torques_l2": RewardTermCfg(
            func=envs_mdp.joint_torques_l2,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
        ),
        "action_rate_l2": RewardTermCfg(func=tracking_mdp.action_rate_l2, weight=-0.1),
        "joint_limit": RewardTermCfg(
            func=tracking_mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "self_collisions": RewardTermCfg(
            func=tracking_mdp.self_collision_cost,
            weight=-10.0,
            params={"sensor_name": self_collision_sensor_cfg.name, "force_threshold": 1.0},
        ),
    }

    #---------------------------- Events ----------------------------
    del cfg.events["push_robot"]

    # Put each env's wall at its env origin (the wall is a mocap body).
    cfg.events["reset_wall"] = EventTermCfg(
        func=envs_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "asset_cfg": SceneEntityCfg("wall")},
    )

    # The balls are what touches the holds: randomise their friction (the feet's in the tracking task).
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (r"^.*_ball_coll$",)

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
    cfg.viewer.azimuth = 200.0  # behind the robot, looking at the wall

    #---------------------------- State estimation ------------------
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
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg
