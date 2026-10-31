# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban climbing variant (ball hands and feet).

The MJCF in microban_climb/ is copied from ../microban_model/src/microban_model/mjcf/robot_climb.xml
(the model used by ../planning/trajecoty.py and ../microban_mpc/climb_mpc.py). It has the same joints as
the walking Microban, but the forearms and feet end in 12.5 mm balls that hook into the holds.
"""

import os
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

MICROBAN_CLIMB_XML: Path = Path(os.path.dirname(__file__)) / "microban_climb" / "robot_climb.xml"
assert MICROBAN_CLIMB_XML.exists(), f"XML not found: {MICROBAN_CLIMB_XML}"

# Ball collision geoms (named in the MJCF), and the bodies carrying them.
BALL_GEOM_NAMES = ("left_hand_ball_coll", "right_hand_ball_coll", "left_foot_ball_coll", "right_foot_ball_coll")
BALL_SITE_NAMES = ("left_hand_ball", "right_hand_ball", "left_foot_ball", "right_foot_ball")
HANDS_BODY_NAMES = ("radius_left", "radius_right")
FEET_BODY_NAMES = ("foot_climb_2", "foot_climb")  # left, right


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MICROBAN_CLIMB_XML))

    # mjlab adds its own (BAM) actuators: drop the XML torque motors so they are not duplicated.
    for act in list(spec.actuators):
        spec.delete(act)

    # Name the collision geoms ("<body>_collision_<i>") so they can be selected by regex. The
    # contype/conaffinity of the MJCF are kept on purpose: body parts (contype 1, conaffinity 2) do not
    # collide with each other, the balls (contype 3) collide with the body, and everything (contype 1)
    # collides with the floor and the wall (conaffinity 1).
    for body in spec.bodies:
        i = 0
        for geom in body.geoms:
            if geom.group == 3 and not geom.name:
                geom.name = f"{body.name}_collision_{i}"
                i += 1

    # Put the trunk at the climb start pose in the model, so that qpos0 is not the trunk at the origin,
    # buried in the floor. mjlab builds its initial MjData at qpos0 and mujoco_warp sizes its per-world
    # contact / constraint buffers from it: buried, it has ~1100 contacts and ~4500 constraint rows,
    # which made njmax default to 6144 and the buffers take many GB at 4096 envs. The free joint state is
    # written in absolute world coordinates, so this changes nothing else.
    trunk = spec.body("trunk")
    trunk.pos = CLIMB_START_FRAME.pos
    trunk.quat = CLIMB_START_FRAME.rot

    for geom in spec.geoms:
        if geom.name in BALL_GEOM_NAMES:
            geom.condim = 3
            geom.priority = 1
            geom.friction[0] = 1.0
    return spec


# Start pose: the first frame of the planned climb (../planning/trajectory.npz), all four balls on holds.
# The trunk position is set in the climbing env from the wall layout.
CLIMB_START_FRAME = EntityCfg.InitialStateCfg(
    pos=(-0.1072, 0.0, 0.2477),
    rot=(1.0, 0.0, -0.00116, 0.0),
    joint_pos={
        "head": 0.0,
        "right_shoulder_pitch": -0.6119,
        "right_shoulder_roll": -0.0310,
        "right_elbow": -0.3503,
        "right_hip_yaw": -0.0933,
        "right_hip_roll": -0.2237,
        "right_hip_pitch": -0.1985,
        "right_knee": 0.01,
        "right_ankle_pitch": 0.2150,
        "right_ankle_roll": -0.0110,
        "left_shoulder_pitch": -0.6119,
        "left_shoulder_roll": 0.0310,
        "left_elbow": -0.3503,
        "left_hip_yaw": 0.0933,
        "left_hip_roll": 0.2237,
        "left_hip_pitch": -0.1985,
        "left_knee": 0.01,
        "left_ankle_pitch": 0.2150,
        "left_ankle_roll": 0.0110,
    },
    joint_vel={r".*": 0.0},
)

from bam.mjlab import BamActuatorCfg  # noqa: E402

# Same actuator model as the walking Microban (microban_constants.py).
actuators = BamActuatorCfg(
    motor_name="xl330",
    model="m6",
    target_names_expr=(r".*",),
    kp_fw=125,
    vin_range=(7.0, 8.0),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=6.0,
    max_current=1.75,
    delay_min_lag=3,
    delay_max_lag=6,
)

MICROBAN_CLIMB_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=CLIMB_START_FRAME,
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        # The planned climb keeps the shoulder rolls at their hard limit (0) and brings the ankle pitches to
        # theirs: with the walking robot's 0.9 the joint limit cost would fight the reference.
        soft_joint_pos_limit_factor=1.0,
    ),
)
