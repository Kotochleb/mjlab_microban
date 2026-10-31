# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban climbing environment: the climbing Microban (ball hands and feet) on a simple wall.

The robot spawns in front of the wall (tasks/climbing/wall.py: a route of the organisers' simulator,
climbing_sim, generated as rock-climbing-workshop/scripts/plan_sim_wall.py does), hanging on its four start
holds, and tracks a planned climb of that wall (BeyondMimic-style, as the tracking task). The motion is made
from plan_sim_wall.py's --save-npz output by scripts/workshop_to_npz.py, for the same wall.

Actions are residuals on the reference: the policy outputs corrections on the joint positions of the planned
climb. The actor observes proprioception only (no state estimation: no base linear velocity or global pose).

Rewards: DeepMimic's imitation reward (pose, velocity, end effectors = balls, root position) with
BeyondMimic's regularisers and a penalty on the residual's magnitude. The RL config is a copy of
tasks/tracking/rl_cfg.py.
"""

import math
from copy import deepcopy
from dataclasses import replace

import numpy as np

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microban.robot.microban_climb_constants import (
    BALL_GEOM_NAMES,
    BALL_SITE_NAMES,
    FEET_BODY_NAMES,
    GRIP_GEOM_NAMES,
    HANDLE_GEOM_NAMES,
    HANDS_BODY_NAMES,
    MICROBAN_CLIMB_ROBOT_CFG,
)
from mjlab_microban.tasks.climbing import mdp
from mjlab_microban.tasks.climbing.wall import (
    COLLISION_HANDLES,
    COLLISION_WORLD,
    ONE_BOX_BODY,
    WallSpec,
    hold_positions,
    make_wall_cfg,
    start_positions,
    wall_spec_from_argv,
)

END_EFFECTOR_BODY_NAMES = FEET_BODY_NAMES + HANDS_BODY_NAMES

# Number of future motion frames (policy steps, 50 Hz) the policy observes, as a rolling window.
FUTURE_WINDOW_STEPS = 20

# Policy steps of history of the actor's proprioceptive terms (the current one and the 4 before it, 0.1 s).
ACTOR_HISTORY_STEPS = 5

# Future frames (policy steps ahead) of the critic's motion window and contact schedule: three stages,
# sparser further ahead (every step, every 2nd, every 4th), 0.36 s in total.
CRITIC_FUTURE_OFFSETS = (1, 2, 3, 4, 6, 8, 10, 14, 18)

# A ball is expected on a hold where its reference position is this close to a hold point.
EXPECTED_CONTACT_THRESHOLD = 0.003

# Residual action scale on top of the reference joint positions, as in BeyondMimic:
# 0.25 * effort_limit / stiffness, at the P gain (0.47 rad at kp 120).
RESIDUAL_STIFFNESS_SCALE = 0.25

# Exponential torque barrier: penalty exp((u - 0.95) / 0.0125) - 1 per joint once the motor uses more than 95 % of its
# available torque u (0 below; 53.6 at saturation).
TORQUE_BARRIER_THRESHOLD = 0.95
TORQUE_BARRIER_SHARPNESS = 0.0125

# BAM firmware P gain (kp_fw), fixed for the whole training.
KP = 120.0

# DeepMimic's humanoid is ~5x Microban's size (pelvis ~0.9 m vs trunk ~0.2 m): its length-based reward
# kernels are rescaled by this factor (root std 0.063 m).
DEEPMIMIC_LENGTH_SCALE = 0.2

# Ball placement: the end-effector kernel is exp(-mean_ball(err^2) / std^2), i.e. 1/e at 5 mm per ball.
BALL_POS_STD = 0.005

# Start pose (the default root/joint state, replaced by the motion's frames on reset): the first frame of
# rock-climbing-workshop's plan_sim_wall.py references (level 1, seed 0), relative to the four start sockets: the
# trunk upright, 0.1016 m out from the socket plane, centred on them and 0.015 m above their mean height
# (rcr.kinematics.ik.TRUNK_HEIGHT). Its balls are in the start sockets of the level 1 wall (~1 cm off for others).
START_TRUNK_OFFSET = (-0.1016, 0.0, 0.015)
START_JOINT_POS = {
    "head": 0.0,
    "left_hip_yaw": 0.0068, "left_hip_roll": 0.0059, "left_hip_pitch": -0.1768, "left_knee": 0.0565,
    "left_ankle_pitch": 0.3386, "left_ankle_roll": -0.0062,
    "right_hip_yaw": -0.0068, "right_hip_roll": -0.0059, "right_hip_pitch": -0.1768, "right_knee": 0.0565,
    "right_ankle_pitch": 0.3386, "right_ankle_roll": 0.0062,
    "left_shoulder_pitch": -1.0618, "left_shoulder_roll": 0.1183, "left_elbow": 0.5387,
    "right_shoulder_pitch": -1.0618, "right_shoulder_roll": -0.1183, "right_elbow": 0.5387,
}  # fmt: skip
# The model's own pose (qpos0: this trunk position, all joints at 0) is what the simulation is first evaluated in,
# and mujoco_warp checks its constraint rows against njmax: it is kept out of the wall's reach (x < 0.2 m).
QPOS0_TRUNK_X = -0.25


def _robot_spec_fn(get_spec, trunk_pos):
    def spec_fn():
        spec = get_spec()
        spec.body("trunk").pos = trunk_pos
        spec.body("trunk").quat = (1.0, 0.0, 0.0, 0.0)
        # Collision groups (wall.py, COLLISION_WORLD): every robot collision geom is in the world group, the
        # four balls and the two hand capsules also in the handles group (the inside of the 5-box cups).
        missing = set(HANDLE_GEOM_NAMES) - {geom.name for geom in spec.geoms}
        assert not missing, f"handle geoms missing from the robot MJCF: {sorted(missing)}"
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                assert geom.contype & COLLISION_WORLD, f"robot geom {geom.name!r} not in the world collision group"
            if geom.name in HANDLE_GEOM_NAMES:
                geom.contype |= COLLISION_HANDLES
        return spec

    return spec_fn

# The anchor (trunk) has to be part of the motion command bodies. As in the tracking task, the policy
# tracks the position of the feet and hands, and the orientation of the head.
TRACKED_BODY_NAMES = ("trunk",) + END_EFFECTOR_BODY_NAMES + ("head",)


def make_microban_climbing_env_cfg(
    play: bool = False,
    wall: WallSpec | None = None,
    motion_file: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """wall: the climbing_sim route (default: the ``wall`` of the motion on the train / play command line, see
    wall.wall_spec_from_argv). It has to be the wall the motion was planned on.
    motion_file: the motion whose ``used_holds`` get the 5-box collision cups (the other holds 1-box blocks);
    None: the train / play command line's --env.commands.motion.motion-file (see wall.make_wall_cfg)."""
    wall = wall_spec_from_argv() if wall is None else wall
    holds = tuple(tuple(float(v) for v in p) for p in hold_positions(wall))
    cfg = make_tracking_env_cfg()

    # Microban's contacts outgrow the G1-sized caps, and climbing_sim's holds are made of many small geoms
    # (hulls, skin patches, ~30 cup spheres each): mujoco_warp's default (64 contacts per world, from the model
    # alone) overflows in the broadphase as soon as a limb is near a hold, and drops contacts. Contacts are
    # pooled across worlds (nconmax per world on average); constraint rows are fixed per world (4 per
    # pyramidal condim-3 contact). An open-loop level-1 climb peaked at ~250 contacts with the robot fallen.
    cfg.sim.nconmax = 384
    cfg.sim.njmax = 1536

    # 5 ms physics timestep with a decimation of 4, so the policy runs at 50 Hz.
    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 4

    #---------------------------- Scene -----------------------------
    robot_cfg = deepcopy(MICROBAN_CLIMB_ROBOT_CFG)
    start = np.array(list(start_positions(wall).values()))
    centre = (0.0, float(start[:, 1].mean()), float(start[:, 2].mean())) if len(start) else (0.0, 0.0, 0.2)
    robot_cfg.init_state = replace(
        robot_cfg.init_state,
        pos=tuple(c + o for c, o in zip(centre, START_TRUNK_OFFSET, strict=True)),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=dict(START_JOINT_POS),
    )
    robot_cfg.spec_fn = _robot_spec_fn(robot_cfg.spec_fn, (QPOS0_TRUNK_X, *robot_cfg.init_state.pos[1:]))
    cfg.scene.entities = {
        "robot": robot_cfg,
        "wall": make_wall_cfg(wall, motion_file),
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
    ball_contact_sensor_cfg = ContactSensorCfg(
        name="ball_contact",
        primary=ContactMatch(mode="geom", pattern=BALL_GEOM_NAMES, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="wall", entity="wall"),  # with the 1-box blocks' body
        fields=("found", "force"),
        reduce="netforce",  # net force on each ball, world frame
        num_slots=1,
    )
    # The balls on the 1-box blocks (the holds the motion does not use): penalised.
    ball_block_contact_sensor_cfg = ContactSensorCfg(
        name="ball_block_contact",
        primary=ContactMatch(mode="geom", pattern=BALL_GEOM_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern=ONE_BOX_BODY, entity="wall"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (self_collision_sensor_cfg, ball_contact_sensor_cfg, ball_block_contact_sensor_cfg)

    #---------------------------- Actions ---------------------------
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    # The policy outputs corrections on the reference joint positions (of the frame the step ends on); a small
    # scale so that the untrained policy roughly plays back the reference.
    cfg.actions["joint_pos"] = mdp.ResidualJointPositionActionCfg(
        entity_name=joint_pos_action.entity_name,
        actuator_names=joint_pos_action.actuator_names,
        stiffness_scale=RESIDUAL_STIFFNESS_SCALE,
        command_name="motion",
    )

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
    # Actor: proprioception only (as the velocity task), and as command the future reference joint
    # configurations (no reference anchor pose, base linear velocity or ball positions).
    actor_terms = cfg.observations["actor"].terms
    cfg.observations["actor"].terms = {
        "base_ang_vel": actor_terms["base_ang_vel"],
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        ),
        # Heading since the episode start, wound up, from a random offset in [0, 2 pi): (cos, sin).
        "base_yaw": ObservationTermCfg(
            func=mdp.base_yaw_windup_cos_sin, params={"offset_range": (0.0, 2.0 * math.pi)}
        ),
        "joint_pos": actor_terms["joint_pos"],
        "joint_vel": actor_terms["joint_vel"],
        "actions": actor_terms["actions"],
        "command": ObservationTermCfg(
            func=mdp.motion_future_joint_pos,
            params={"command_name": "motion", "num_steps": FUTURE_WINDOW_STEPS},
        ),
    }
    # Short history of the proprioceptive terms (the MLP has no memory otherwise). The command is already a
    # future window of the reference, so it is not stacked.
    for name, term in cfg.observations["actor"].terms.items():
        if name != "command":
            cfg.observations["actor"].terms[name] = replace(term, history_length=ACTOR_HISTORY_STEPS)

    # Privileged critic observations: world placement (position relative to the env origin, and
    # orientation) of the robot's trunk and of the reference trunk. The wall is fixed in the env frame,
    # so this tells the critic where the robot is relative to the holds.
    cfg.observations["critic"].terms["base_pose_w"] = ObservationTermCfg(
        func=mdp.robot_anchor_pose_w, params={"command_name": "motion"}
    )
    cfg.observations["critic"].terms["motion_base_pose_w"] = ObservationTermCfg(
        func=mdp.motion_anchor_pose_w, params={"command_name": "motion"}
    )

    # Critic (privileged):
    critic_terms = cfg.observations["critic"].terms
    # - future frames of the planned climb (anchor pose, ball positions and joint positions), sparser further ahead,
    critic_terms["motion_future"] = ObservationTermCfg(
        func=mdp.motion_future_window,
        params={"command_name": "motion", "site_names": BALL_SITE_NAMES, "offsets": CRITIC_FUTURE_OFFSETS},
    )
    # - ball-wall contacts and contact forces,
    critic_terms["ball_contacts"] = ObservationTermCfg(
        func=mdp.ball_contacts,
        params={"sensor_name": ball_contact_sensor_cfg.name, "geom_names": BALL_GEOM_NAMES},
    )
    # - where and when the reference expects the balls on holds (now, at the future frames, next contact),
    critic_terms["motion_contact_schedule"] = ObservationTermCfg(
        func=mdp.motion_contact_schedule,
        params={
            "command_name": "motion",
            "site_names": BALL_SITE_NAMES,
            "offsets": CRITIC_FUTURE_OFFSETS,
            "hold_positions": holds,
            "threshold": EXPECTED_CONTACT_THRESHOLD,
        },
    )
    # - the rewarded tracking errors (balls, trunk position) in the trunk frame,
    critic_terms["tracking_errors"] = ObservationTermCfg(
        func=mdp.motion_tracking_errors_b, params={"command_name": "motion", "site_names": BALL_SITE_NAMES}
    )
    # - the applied joint torques.
    critic_terms["joint_torques"] = ObservationTermCfg(func=mdp.applied_joint_torques)
    # - the heading, privileged: rotation about the board's z axis relative to the board, no offset: (cos, sin).
    critic_terms["base_yaw"] = ObservationTermCfg(func=mdp.base_yaw_rel_board_cos_sin, params={"board_name": "wall"})

    #---------------------------- Rewards ---------------------------
    # Imitation: DeepMimic's reward (Peng et al., SIGGRAPH 2018), the standard reward for policies tracking
    # a reference motion, used e.g. with residual (reference + correction) actions in Xie et al., "Feedback
    # control for Cassie with deep RL" (IROS 2018). Weights 0.65 / 0.1 / 0.15 / 0.1 (x10) and kernels
    # exp(-k * e) with k = 2 / 0.1 / - / 10; the root kernel is rescaled by (1 / DEEPMIMIC_LENGTH_SCALE)^2 for
    # Microban's size (centre of mass: the trunk). The end effectors are the balls, which go into the holds:
    # their kernel is set by the placement accuracy wanted, BALL_POS_STD (5 mm) per ball.
    # Regularisation: BeyondMimic's (action rate, joint limits, self collisions), joint torques, and a
    # penalty on the residual's magnitude, as usual in residual RL, so the policy stays close to the reference.
    length_k = 1.0 / DEEPMIMIC_LENGTH_SCALE**2
    cfg.rewards = {
        "pose": RewardTermCfg(
            func=mdp.deepmimic_pose_exp, weight=6.5, params={"command_name": "motion", "k": 2.0}
        ),
        "velocity": RewardTermCfg(
            func=mdp.deepmimic_velocity_exp, weight=1.0, params={"command_name": "motion", "k": 0.1}
        ),
        "end_effector": RewardTermCfg(
            func=mdp.ball_position_error_exp,
            weight=1.5,
            params={
                "command_name": "motion",
                "std": BALL_POS_STD,
                "site_names": BALL_SITE_NAMES,
                "reduction": "mean",
            },
        ),
        "root_position": RewardTermCfg(
            func=mdp.deepmimic_root_position_exp,
            weight=1.0,
            params={"command_name": "motion", "k": 10.0 * length_k},
        ),
        "residual_l2": RewardTermCfg(func=mdp.residual_action_l2, weight=-0.01),
        "action_rate_l2": RewardTermCfg(func=tracking_mdp.action_rate_l2, weight=-0.1),
        "joint_torques_l2": RewardTermCfg(
            func=envs_mdp.joint_torques_l2,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
        ),
        # Exponential barrier past 95 % of the available motor torque (BAM: velocity and battery dependent).
        "torque_barrier": RewardTermCfg(
            func=mdp.motor_torque_exp_barrier,
            weight=-0.05,
            params={"threshold": TORQUE_BARRIER_THRESHOLD, "sharpness": TORQUE_BARRIER_SHARPNESS},
        ),
        # Small penalty per ball (hand / foot sphere) touching a 1-box block, i.e. a hold the motion does not use.
        "ball_on_unused_hold": RewardTermCfg(
            func=mdp.contact_count,
            weight=-0.5,
            params={"sensor_name": ball_block_contact_sensor_cfg.name},
        ),
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

    # Fixed BAM firmware P gain, for all envs.
    cfg.events["pd_gains"] = EventTermCfg(func=mdp.set_pd_gain, mode="startup", params={"kp": KP})

    # Put each env's wall at its env origin (the wall is a mocap body).
    cfg.events["reset_wall"] = EventTermCfg(
        func=envs_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "asset_cfg": SceneEntityCfg("wall")},
    )

    # The balls and capsules are what touches the holds: randomise their friction (the feet's in the tracking
    # task), sliding coefficient U(0.1, 0.5), shared by all of them in an env. They have priority 1, so this is
    # the friction of their contacts with the holds (and with the board / floor), not the holds' HOLD_FRICTION.
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = GRIP_GEOM_NAMES
    cfg.events["foot_friction"].params["ranges"] = (0.1, 0.5)

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
    # No dof_friction_randomization event: BAM zeroes MuJoCo's frictionloss and models the joint friction
    # itself, so the friction is randomised by the actuator_dr curriculum below instead.

    #---------------------------- Curriculum ------------------------
    # Boosted motors (15 V, 3 A) at the start, linearly down to the realistic ones (the BAM actuator config's
    # vin_range / max_current) over 500 training iterations of 24 env steps each.
    if not play:
        # Actuator DR: joint friction +-10 % and battery voltage 7-9 V (8 V +-12.5 %), per env. Both ranges
        # start at +-1 %, held until iteration 500, then widen linearly to the full ones at iteration 1000.
        # Listed before bam_power, which interpolates its boosted voltage to these.
        cfg.curriculum["actuator_dr"] = CurriculumTermCfg(
            func=mdp.actuator_dr_curriculum,
            params={
                "vin_nominal": 8.0,
                "vin_rel_range": 0.125,
                "friction_rel_range": 0.10,
                "rel_start": 0.01,
                "hold_steps": 500 * 24,
                "full_steps": 1000 * 24,
            },
        )
        # Ball sphere (collision geom) positions: +-1 mm in x, y and z, per env and ball, same schedule.
        cfg.curriculum["ball_pos_dr"] = CurriculumTermCfg(
            func=mdp.geom_pos_dr_curriculum,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=BALL_GEOM_NAMES),
                "max_offset": 0.001,
                "start_frac": 0.01,
                "hold_steps": 500 * 24,
                "full_steps": 1000 * 24,
            },
        )
        # Reference motion speed, per env and episode: U(0.95, 1.05) up to iteration 500, then the bounds widen
        # linearly to U(0.9, 1.5) at iteration 1500.
        cfg.curriculum["motion_speed"] = CurriculumTermCfg(
            func=mdp.motion_speed_curriculum,
            params={
                "command_name": "motion",
                "start_range": (0.95, 1.05),
                "full_range": (0.9, 1.5),
                "hold_steps": 500 * 24,
                "full_steps": 1500 * 24,
            },
        )
        cfg.curriculum["bam_power"] = CurriculumTermCfg(
            func=mdp.bam_power_curriculum,
            params={"vin_start": 15.0, "max_current_start": 3.0, "num_steps": 500 * 24},
        )

    #---------------------------- Terminations ----------------------
    cfg.terminations["anchor_pos"].params["threshold"] = 0.07
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.07
    cfg.terminations["ee_body_pos"].params["body_names"] = END_EFFECTOR_BODY_NAMES
    # Stopgap: rare physics blow-ups (NaN state) under violent early-training actions; reset those envs
    # before their observations are computed (their reward is already zeroed by the reward manager).
    cfg.terminations["nan_detection"] = TerminationTermCfg(func=envs_mdp.nan_detection)
    # The trunk turned away from the board: more than 60 deg about the board's z axis.
    cfg.terminations["base_yaw_rel_board"] = TerminationTermCfg(
        func=mdp.base_yaw_rel_board_exceeded,
        params={"max_yaw": math.radians(60.0), "board_name": "wall"},
    )

    #---------------------------- Viewer ----------------------------
    # Camera on the robot's side of the wall, looking straight at the holds (+x), ~1.5 m from the wall.
    # It tracks the trunk (x ~ -0.12 m), so the horizontal offset is 1.5 - 0.12 = 1.38 m.
    cfg.viewer.body_name = "trunk"
    cfg.viewer.azimuth = 0.0  # looking along +x, towards the wall
    cfg.viewer.elevation = -20.0  # from above: the camera is ~0.5 m above the trunk
    cfg.viewer.distance = 1.38 / math.cos(math.radians(-cfg.viewer.elevation))

    #---------------------------- Play mode -------------------------
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg
