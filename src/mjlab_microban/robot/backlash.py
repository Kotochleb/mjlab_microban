# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Joint backlash, modelled as in the MuJoCo documentation (Modeling > Tips and tricks > Backlash,
https://mujoco.readthedocs.io/en/latest/modeling.html#backlash):

    "Instead of having a single hinge joint within the body, define two hinge joints with identical position and
    orientation [...] Thus the overall rotation of the body relative to its parent is J1+J2. Now define an actuator
    acting only on J1. The small joint range on J2 keeps it near 0, yet allows it to move a bit in the direction of
    the force acting on it, producing a backlash effect. Note the armature attribute in J1. Without it the
    joint-space inertia matrix will be singular [...]"

add_backlash_joints() adds, right after each hinge J1 of the spec, a passive hinge ``<J1>{BACKLASH_SUFFIX}`` (J2)
in the same body, same position and axis, limited to +-max_backlash. J1 keeps its actuator and its armature (the
BAM actuator sets J1's armature to the gearbox's reflected inertia), J2 has none. The range of each J2 is then
drawn per env by the climbing task's ``backlash`` startup event (tasks/climbing/mdp.randomize_backlash).

The actuated joint J1 is the motor side (what the servo's controller and the policy see); J2 is the slack between
the motor and the link, unobserved. ACTUATED_JOINT_EXPR selects the actuated joints only.
"""

import math

import mujoco

BACKLASH_SUFFIX = "_backlash"
# Regex (re.fullmatch, as mjlab's name matching) of every joint but the backlash joints.
ACTUATED_JOINT_EXPR = rf"(?!.*{BACKLASH_SUFFIX}$).*"
# Regex of the backlash joints.
BACKLASH_JOINT_EXPR = rf".*{BACKLASH_SUFFIX}"

# [rad] largest backlash (half-range of the J2 joint: it moves within +-this), 1 degree.
MAX_BACKLASH = math.radians(1.0)

# Limit constraint of the backlash joints: the docs' "solref and solimp parameters of the joint limit constraint
# could also be adjusted, to make the backlash rotation end at a softer or a harder limit". MuJoCo's default limit
# (solref 0.02, solimp 0.9 / 0.95) is far too soft for a 1 degree gap: a Microban hanging sideways at a 5 ms
# timestep pushed a +-0.5 deg backlash joint to 2.8 deg. A time constant of twice the 5 ms timestep (the smallest
# MuJoCo keeps stable) and a near-hard impedance keep it within ~0.04 deg of its range under gravity.
BACKLASH_SOLREF = (0.01, 1.0)
BACKLASH_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)


def is_backlash_joint(name: str) -> bool:
    return name.endswith(BACKLASH_SUFFIX)


def add_backlash_joints(spec: mujoco.MjSpec, max_backlash: float = MAX_BACKLASH) -> mujoco.MjSpec:
    """Add a passive backlash hinge after every hinge of ``spec`` (in place; returns it), limited to +-max_backlash.

    Each body of the robot has a single hinge, so J2 is the body's last joint, and comes right after J1 in MuJoCo's
    joint order: the actuated joints keep their relative order (that of the motions' joint_pos).
    """
    hinges = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE and not is_backlash_joint(j.name)]
    for j1 in hinges:
        body = j1.parent
        assert list(body.joints)[-1].name == j1.name, f"backlash: {j1.name!r} is not the last joint of its body"
        body.add_joint(
            name=f"{j1.name}{BACKLASH_SUFFIX}",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            pos=j1.pos,
            axis=j1.axis,
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            range=[-max_backlash, max_backlash],
            solref_limit=list(BACKLASH_SOLREF),
            solimp_limit=list(BACKLASH_SOLIMP),
            armature=0.0,
            damping=0.0,
            frictionloss=0.0,
            stiffness=0.0,
        )
    return spec
