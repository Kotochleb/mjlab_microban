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
# Same robot with the collision approximation of the climbing model (hulls and servo boxes), used through
# get_hulls_spec() / MICROBAN_HULLS_ROBOT_CFG. MICROBAN_XML and MICROBAN_ROBOT_CFG keep the primitive
# collisions.
MICROBAN_HULLS_XML: Path = MICROBAN_XML.parent / "robot_hulls.xml"
assert MICROBAN_HULLS_XML.exists(), f"XML not found: {MICROBAN_HULLS_XML}"

# End-effector collision spheres, in the body frames of robot_hulls.xml: (body, name, radius, pos).
# The XML's collision approximation (hulls and servo boxes copied from microban_climb/robot_climb.xml) is
# kept for the rest of the body; the forearms and feet are replaced by spheres, cheaper and smoother
# contacts.
#  - Feet: one sphere in each rounded corner of the sole (sole plane y = -0.023, corner radius 5 mm).
#  - Hands: one sphere at the forearm tip, as wide as the forearm (18 mm).
FOOT_SPHERE_RADIUS = 0.005
HAND_SPHERE_RADIUS = 0.009
END_EFFECTOR_SPHERES = tuple(
    (body, f"{side}_foot_collision_{k}", FOOT_SPHERE_RADIUS, (x, -0.023 + FOOT_SPHERE_RADIUS, z))
    for body, side in (("foot", "right"), ("foot_2", "left"))
    for k, (x, z) in enumerate(((-0.0175, 0.002), (0.0175, 0.002), (-0.0175, 0.082), (0.0175, 0.082)), start=1)
) + (
    ("radius", "right_hand_collision", HAND_SPHERE_RADIUS, (0.0, -0.0715 + HAND_SPHERE_RADIUS, 0.0135)),
    ("radius_2", "left_hand_collision", HAND_SPHERE_RADIUS, (0.0, 0.0715 - HAND_SPHERE_RADIUS, -0.0135)),
)


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICROBAN_XML))


def get_hulls_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MICROBAN_HULLS_XML))
    for body_name in {body for body, *_ in END_EFFECTOR_SPHERES}:
        body = spec.body(body_name)
        for geom in list(body.geoms):
            if geom.group == 3:
                spec.delete(geom)
    for body_name, name, radius, pos in END_EFFECTOR_SPHERES:
        spec.body(body_name).add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0.0, 0.0],
            pos=pos,
            group=3,
            rgba=[0.1, 0.85, 0.85, 1.0],
        )
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

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    condim={r"^(left|right)_foot_collision$": 3, r".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# FULL_COLLISION with the foot patterns matching the numbered foot geoms ("left_foot_collision_3"): the
# anchored "^(left|right)_foot_collision$" above matches none, so there the feet get condim 1 and no
# priority / friction override.
HULLS_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    condim={r"^(left|right)_foot_collision_\d+$": 3, r".*_collision": 1},
    priority={r"^(left|right)_foot_collision_\d+$": 1},
    friction={r"^(left|right)_foot_collision_\d+$": (1.0,)},
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
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Hull collision approximation (robot_hulls.xml) with sphere hands and feet.
MICROBAN_HULLS_ROBOT_CFG = EntityCfg(
    spec_fn=get_hulls_spec,
    init_state=HOME_FRAME,
    collisions=(HULLS_COLLISION,),
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