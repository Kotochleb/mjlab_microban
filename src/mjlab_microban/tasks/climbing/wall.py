"""A simple climbing wall: a vertical board with two columns of hold cups.

The layout is the one of ../planning/trajecoty.py (the planned climb this task tracks): holds at x = 0, in
two columns 0.15 m apart, every 0.06 m in height. The planner's frame is shifted by HOLDS_OFFSET so that
the robot is centred on its env origin (y = 0) and the lowest holds are off the floor.

Frame (env origin): x towards the wall (the robot is at x < 0, facing +x), y to the robot's left, z up.

Each hold is a cup made of two boxes: a ledge the ball rests on and a lip in front of it, so a ball
sitting in the cup has its centre exactly on the hold point (the contact point of the planner). The
ball has to be lifted ~1 cm to get out of the cup, the planner lifts it by 3 cm.
"""

import mujoco
import numpy as np

from mjlab.entity import EntityCfg

# Planned layout (../planning/trajecoty.py).
Y_SPACING = 0.15
Z_SPACING = 0.06
NUM_HOLD_ROWS = 17  # z in [0, 1) every 0.06 m

# Planner frame -> env frame.
HOLDS_OFFSET = np.array([0.0, -Y_SPACING / 2, 0.10])

BALL_RADIUS = 0.0125  # climbing ball of robot_climb.xml

# Board: its front face touches the balls sitting in the cups.
BOARD_FRONT_X = BALL_RADIUS
BOARD_THICKNESS = 0.02
BOARD_WIDTH = 0.8
BOARD_HEIGHT = 1.4

# Cup.
CUP_HALF_WIDTH = 0.02  # along y
LEDGE_THICKNESS = 0.01
LIP_THICKNESS = 0.006  # along x
LIP_HEIGHT = 0.0135  # top of the lip, above the ledge: ~1 cm below the ball centre
LIP_GAP = 0.001  # between the ball and the lip

HOLD_FRICTION = 0.5  # same as the hackathon simulator (climbing_sim.scene)


def hold_positions() -> np.ndarray:
    """(N, 3) hold points (centre of a ball sitting in the cup) in the env frame."""
    holds = [[0.0, y, z] for y in (0.0, Y_SPACING) for z in np.arange(NUM_HOLD_ROWS) * Z_SPACING]
    return np.array(holds) + HOLDS_OFFSET


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.add_material(name="wall_wood", rgba=[0.86, 0.75, 0.56, 1.0])
    spec.add_material(name="wall_hold", rgba=[0.85, 0.35, 0.15, 1.0])

    wall = spec.worldbody.add_body(name="wall")
    wall.add_geom(
        name="board",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[BOARD_THICKNESS / 2, BOARD_WIDTH / 2, BOARD_HEIGHT / 2],
        pos=[BOARD_FRONT_X + BOARD_THICKNESS / 2, HOLDS_OFFSET[1] + Y_SPACING / 2, BOARD_HEIGHT / 2],
        material="wall_wood",
        condim=3,
        friction=[HOLD_FRICTION, 0.005, 0.0001],
    )

    lip_x = -BALL_RADIUS - LIP_GAP - LIP_THICKNESS / 2
    ledge_back, ledge_front = BOARD_FRONT_X, lip_x - LIP_THICKNESS / 2
    for k, (x, y, z) in enumerate(hold_positions()):
        ledge_top = z - BALL_RADIUS
        wall.add_site(name=f"hold{k}", pos=[x, y, z], size=[0.003, 0, 0], rgba=[0.1, 0.9, 0.2, 1], group=4)
        wall.add_geom(
            name=f"hold{k}_ledge",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[(ledge_back - ledge_front) / 2, CUP_HALF_WIDTH, LEDGE_THICKNESS / 2],
            pos=[(ledge_back + ledge_front) / 2, y, ledge_top - LEDGE_THICKNESS / 2],
            material="wall_hold",
            condim=3,
            friction=[HOLD_FRICTION, 0.005, 0.0001],
        )
        lip_bottom, lip_top = ledge_top - LEDGE_THICKNESS, ledge_top + LIP_HEIGHT
        wall.add_geom(
            name=f"hold{k}_lip",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[LIP_THICKNESS / 2, CUP_HALF_WIDTH, (lip_top - lip_bottom) / 2],
            pos=[lip_x, y, (lip_top + lip_bottom) / 2],
            material="wall_hold",
            condim=3,
            friction=[HOLD_FRICTION, 0.005, 0.0001],
        )
    return spec


# Fixed, non-articulated entity: mjlab wraps it in a mocap body, placed at each env origin by the
# "reset_wall" event of the climbing env.
WALL_CFG = EntityCfg(spec_fn=get_spec)
