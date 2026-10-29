# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import os
import numpy as np
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

MICROBAN_XML: Path = Path(os.path.dirname(__file__)) / "microban" / "robot.xml"
assert MICROBAN_XML.exists(), f"XML not found: {MICROBAN_XML}"

# Unnamed collision geoms of the generated XML, as (body, index of the geom within the body).
# get_spec names them "<body>_collision_<index>" so CollisionCfg and contact pairs can select them.
NAMED_COLLISION_GEOMS = (
    ("trunk", 2),
    ("radius", 0),
    ("radius", 1),
    ("radius_2", 0),
    ("radius_2", 1),
    ("hip_block__configuration_right", 2),
    ("hip_block__configuration_left", 2),
    ("ankle_block__configuration_right", 2),
    ("ankle_block__configuration_left", 2),
)

# The only robot self-collisions; every other pair of robot geoms never collides.
SELF_COLLISION_PAIRS = (
    ("left_foot_collision_2", "right_foot_collision_2"),
    ("ankle_block__configuration_left_collision_2", "ankle_block__configuration_right_collision_2"),
    ("radius_2_collision_0", "trunk_collision_2"),
    ("radius_collision_0", "trunk_collision_2"),
)


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MICROBAN_XML))
    for body_name, index in NAMED_COLLISION_GEOMS:
        geom = spec.body(body_name).geoms[index]
        assert geom.classname.name == "collision", f"{body_name} geom {index} is not a collision geom"
        geom.name = f"{body_name}_collision_{index}"
    # Disable every geom; ENV_COLLISION re-enables the ones touching the environment.
    for geom in spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0
    # Explicit pairs bypass contype/conaffinity.
    for geom1, geom2 in SELF_COLLISION_PAIRS:
        spec.add_pair(name=f"{geom1}__{geom2}", geomname1=geom1, geomname2=geom2)
    return spec

HOME_FRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.168), #0.1676),
    joint_pos={
        "head": float(np.deg2rad(0.0)),
        "left_shoulder_roll": float(np.deg2rad(10.0)),
        "right_shoulder_roll": float(np.deg2rad(-10.0)),
        "left_shoulder_pitch": float(np.deg2rad(0.0)),
        "right_shoulder_pitch": float(np.deg2rad(0.0)),
        "left_elbow": float(np.deg2rad(-20.0)),
        "right_elbow": float(np.deg2rad(-20.0)),
        "left_hip_roll": float(np.deg2rad(5.0)),
        "right_hip_roll": float(np.deg2rad(-5.0)),
        "left_hip_pitch": float(np.deg2rad(-10.0)),
        "right_hip_pitch": float(np.deg2rad(-10.0)),
        "left_hip_yaw": float(np.deg2rad(0.0)),
        "right_hip_yaw": float(np.deg2rad(0.0)),
        "left_knee": float(np.deg2rad(0.0)),
        "right_knee": float(np.deg2rad(0.0)),
        "left_ankle_roll": float(np.deg2rad(-5.0)),
        "right_ankle_roll": float(np.deg2rad(5.0)),
        "left_ankle_pitch": float(np.deg2rad(0.0)),
        "right_ankle_pitch": float(np.deg2rad(0.0)),
    },
    joint_vel={r".*": 0.0},
)

# Geoms colliding with the environment. contype=1/conaffinity=0 collides with the terrain
# (contype=conaffinity=1) but never with another robot geom; see SELF_COLLISION_PAIRS for those.
ENV_COLLISION = CollisionCfg(
    geom_names_expr=(
        r"^(left|right)_foot_collision_[3-6]$",
        r"^radius(_2)?_collision_[01]$",
        r"^trunk_collision_2$",
        r"^hip_block__configuration_(right|left)_collision_2$",
    ),
    contype=1,
    conaffinity=0,
    condim={r"^(left|right)_foot_collision$": 3, r".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

from bam.mjlab import BamActuatorCfg

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

# -- Old actuator (XML position, MuJoCo default) --
# actuators = XmlActuatorCfg(
#     target_names_expr=(r".*",),
#     delay_min_lag=0,
#     delay_max_lag=3,
# )

MICROBAN_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=HOME_FRAME,
    collisions=(ENV_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": MICROBAN_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cpu")
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    viewer.launch(model, data=data)