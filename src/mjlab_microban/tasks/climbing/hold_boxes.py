"""Box approximations of the climbing_sim hold ("cup"), as an alternative to its mesh hulls + skin + spheres.

All boxes are in the hold frame of climbing_sim (the frame of climbing_sim/assets/hold/*.stl): origin at the
socket (the centre of a ball sitting in the cup), x to the right along the wall, y into the wall, z up.
The wall face is the plane y = +26 mm (SOCKET_OUT). In the env frame, a box centre goes through
``SIM_TO_ENV_ROT`` and is offset by the socket, and its orientation is ``SIM_TO_ENV_QUAT`` (see wall.py).

Measured on hold_visual.stl (mm, hold frame):
  outer extent    x [-20.1, 20.1] (mounting plate; body +-17), y [-17, 26], z [-15, 18]
  bowl            a sphere of radius ~12.7 around the socket, open at the top, bottom at z ~ -13.7
  front lip       outer face y ~ -17, inner face y ~ -12.7, top z ~ +4.5 at x = 0 (rim drops to ~-2 at the sides)
  back            rear block y [12.3, 23] z [-15, 0], mounting plate y [23, 26] z [-15, 18]

Three boxes (THREE_BOX), the cup of the simple wall (ledge + lip, git befc6c7) with a back wall:
  ledge  the floor of the cup: its top touches the ball (z = -BALL_RADIUS), over the hold's whole footprint
  lip    the front of the cup, 0.1 mm (GAP) in front of the ball, up to the lip top of the mesh
  back   the rear block down to the board, 0.1 mm (GAP) behind the ball, up to the back rim of the bowl

Five boxes (FIVE_BOX): the three boxes plus a wall on each side, so the ball cannot roll off sideways:
  side   as high as the back (z up to BACK_TOP_Z), as deep as the ledge (front of the hold to the wall), as thick
         along x as the lip is along y
  then the lip, back and sides are thinned by 0.5 mm on their inner faces: a 0.6 mm gap all around the ball

One box (ONE_BOX), coarse: a block with the ledge's footprint (width and depth) and the height of the 5-box lip (z up
to LIP_TOP_Z): the cup filled in, so a ball rests on its top, BALL_RADIUS + LIP_TOP_Z above the socket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BALL_RADIUS = 0.0125  # robot_climb.xml: *_ball_coll
GAP = 0.0001  # between a ball sitting in the socket and the lip / back

# Hold extents (m, hold frame), from hold_visual.stl.
HOLD_HALF_WIDTH = 0.017  # the body (the mounting plate is +-20.1 mm, but it lies flat on the board)
HOLD_FRONT_Y = -0.017
HOLD_BACK_Y = 0.026  # the wall face
HOLD_BOTTOM_Z = -0.015
LIP_TOP_Z = 0.0045
BACK_TOP_Z = 0.0


@dataclass(frozen=True)
class Box:
    name: str
    lo: tuple[float, float, float]  # min corner, hold frame (m)
    hi: tuple[float, float, float]  # max corner

    @property
    def center(self) -> np.ndarray:
        return (np.array(self.lo) + np.array(self.hi)) / 2

    @property
    def half_size(self) -> np.ndarray:
        """MuJoCo box size."""
        return (np.array(self.hi) - np.array(self.lo)) / 2


LEDGE = Box(
    "ledge",
    lo=(-HOLD_HALF_WIDTH, HOLD_FRONT_Y, HOLD_BOTTOM_Z),
    hi=(HOLD_HALF_WIDTH, HOLD_BACK_Y, -BALL_RADIUS),
)
LIP = Box(
    "lip",
    lo=(-HOLD_HALF_WIDTH, HOLD_FRONT_Y, HOLD_BOTTOM_Z),
    hi=(HOLD_HALF_WIDTH, -BALL_RADIUS - GAP, LIP_TOP_Z),
)
BACK = Box(
    "back",
    lo=(-HOLD_HALF_WIDTH, BALL_RADIUS + GAP, HOLD_BOTTOM_Z),
    hi=(HOLD_HALF_WIDTH, HOLD_BACK_Y, BACK_TOP_Z),
)

LIP_THICKNESS = LIP.hi[1] - LIP.lo[1]  # along y

# Five boxes: the walls (lip, back, sides) thinned by FIVE_BOX_THINNING on their inner faces, for a little more
# room around the ball (0.6 mm gap instead of 0.1 mm); the ledge is kept, so the ball still sits on the socket.
FIVE_BOX_THINNING = 0.0005
FIVE_BOX_GAP = GAP + FIVE_BOX_THINNING
LIP_5 = Box(
    "lip",
    lo=LIP.lo,
    hi=(LIP.hi[0], LIP.hi[1] - FIVE_BOX_THINNING, LIP.hi[2]),
)
BACK_5 = Box(
    "back",
    lo=(BACK.lo[0], BACK.lo[1] + FIVE_BOX_THINNING, BACK.lo[2]),
    hi=BACK.hi,
)
SIDE_THICKNESS = LIP_THICKNESS - FIVE_BOX_THINNING  # along x
SIDE_POS_X = Box(
    "side_pos_x",
    lo=(HOLD_HALF_WIDTH - SIDE_THICKNESS, HOLD_FRONT_Y, HOLD_BOTTOM_Z),
    hi=(HOLD_HALF_WIDTH, HOLD_BACK_Y, BACK_TOP_Z),
)
SIDE_NEG_X = Box(
    "side_neg_x",
    lo=(-HOLD_HALF_WIDTH, HOLD_FRONT_Y, HOLD_BOTTOM_Z),
    hi=(-HOLD_HALF_WIDTH + SIDE_THICKNESS, HOLD_BACK_Y, BACK_TOP_Z),
)

THREE_BOX: tuple[Box, ...] = (LEDGE, LIP, BACK)
FIVE_BOX: tuple[Box, ...] = (LEDGE, LIP_5, BACK_5, SIDE_POS_X, SIDE_NEG_X)
BLOCK = Box(
    "block",
    lo=LEDGE.lo,
    hi=(LEDGE.hi[0], LEDGE.hi[1], LIP_5.hi[2]),
)
ONE_BOX: tuple[Box, ...] = (BLOCK,)

APPROXIMATIONS: dict[str, tuple[Box, ...]] = {"three_box": THREE_BOX, "five_box": FIVE_BOX, "one_box": ONE_BOX}
