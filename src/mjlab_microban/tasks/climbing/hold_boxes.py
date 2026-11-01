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
  then the lip, back and sides are thinned by 0.5 mm on their inner faces: a 0.6 mm gap all around the ball,
  and the sides and the lip are trimmed by 5 mm (FIVE_BOX_BOTTOM_TRIM) from the bottom, and the ledge by 7 mm
  (LEDGE_FRONT_TRIM) from the front; the ledge and the lip are 10 mm narrower along x (FIVE_BOX_NARROWING, half on
  each side); the sides are 8 mm shorter along y (SIDE_FRONT_TRIM), from the front; the sides and the lip are
  tilted outwards by 15 degrees (FIVE_BOX_TILT_DEG) about their bottom inner edge, so the cup opens upwards, then
  moved along x (sides) / y (lip) so their inner face is tangent to the ball inflated by 0.1 mm (CUP_SLACK);
  the tilted lip and sides are 0.5 mm thinner (TILTED_WALL_THINNING), taken off their outer face; the back is
  0.1 mm (CUP_SLACK) behind the ball: 0.1 mm of slack all around; the tilted lip's top is 3 mm lower
  (LIP_TOP_TRIM)
  front_post  an upright box in front of the ball, as thick along y as the 5-box lip before its tilt (3.9 mm), its top
         at the lip's original top (LIP_TOP_Z), 4 mm high and 12 mm wide (FRONT_POST_*), moved along y so that it
         touches the ball inflated by CUP_SLACK (its nearest point, the bottom inner edge, at TANGENT_RADIUS)
  side_post_{pos,neg}_x  upright boxes as thick along x as the sides before their tilt (3.9 mm), their top at
         BACK_TOP_Z, as deep as the sides now (y from the front trim to the wall), 3 mm high (SIDE_POST_HEIGHT),
         moved along x so that they touch the ball inflated by CUP_SLACK (inner face at TANGENT_RADIUS)

One box (ONE_BOX), coarse: a block with the ledge's footprint (width and depth) and the height of the 5-box lip (z up
to LIP_TOP_Z): the cup filled in, so a ball rests on its top, BALL_RADIUS + LIP_TOP_Z above the socket.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    """A box given by its corners before rotation, then rotated by ``quat`` about ``pivot`` (default: no rotation)."""

    name: str
    lo: tuple[float, float, float]  # min corner, hold frame (m), before rotation
    hi: tuple[float, float, float]  # max corner
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz, hold frame
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)  # the point the rotation keeps fixed

    @property
    def rot(self) -> np.ndarray:
        w, x, y, z = self.quat
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    @property
    def center(self) -> np.ndarray:
        """MuJoCo geom position, hold frame."""
        c = (np.array(self.lo) + np.array(self.hi)) / 2
        p = np.array(self.pivot)
        return p + self.rot @ (c - p)

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
# room around the ball (0.6 mm gap instead of 0.1 mm); the ledge keeps its top, so the ball still sits on the socket.
FIVE_BOX_THINNING = 0.0005
FIVE_BOX_GAP = GAP + FIVE_BOX_THINNING
CUP_SLACK = 0.0001  # the final 5-box cup: between the ball and the back, the lip and the sides
FIVE_BOX_BOTTOM_TRIM = 0.005  # the lip and the sides start this far above the hold's bottom
LEDGE_FRONT_TRIM = 0.007  # the ledge starts this far behind the front of the hold
FIVE_BOX_NARROWING = 0.010  # the ledge and the lip are this much narrower along x (half on each side)
LEDGE_5 = Box(
    "ledge",
    lo=(LEDGE.lo[0] + FIVE_BOX_NARROWING / 2, LEDGE.lo[1] + LEDGE_FRONT_TRIM, LEDGE.lo[2]),
    hi=(LEDGE.hi[0] - FIVE_BOX_NARROWING / 2, LEDGE.hi[1], LEDGE.hi[2]),
)
LIP_5 = Box(
    "lip",
    lo=(LIP.lo[0] + FIVE_BOX_NARROWING / 2, LIP.lo[1], LIP.lo[2] + FIVE_BOX_BOTTOM_TRIM),
    hi=(LIP.hi[0] - FIVE_BOX_NARROWING / 2, LIP.hi[1] - FIVE_BOX_THINNING, LIP.hi[2]),
)
# Upright post at the untilted lip's place (thickness along y), its top at the lip's original top.
FRONT_POST_HEIGHT = 0.004
FRONT_POST_WIDTH = 0.012
FRONT_POST = Box(
    "front_post",
    lo=(-FRONT_POST_WIDTH / 2, LIP_5.lo[1], LIP_5.hi[2] - FRONT_POST_HEIGHT),
    hi=(FRONT_POST_WIDTH / 2, LIP_5.hi[1], LIP_5.hi[2]),
)
BACK_5 = Box(
    "back",
    lo=(BACK.lo[0], BALL_RADIUS + CUP_SLACK, BACK.lo[2]),
    hi=BACK.hi,
)
SIDE_THICKNESS = LIP_THICKNESS - FIVE_BOX_THINNING  # along x
SIDE_FRONT_TRIM = 0.008  # the sides start this far behind the front of the hold
SIDE_POS_X = Box(
    "side_pos_x",
    lo=(HOLD_HALF_WIDTH - SIDE_THICKNESS, HOLD_FRONT_Y + SIDE_FRONT_TRIM, HOLD_BOTTOM_Z + FIVE_BOX_BOTTOM_TRIM),
    hi=(HOLD_HALF_WIDTH, HOLD_BACK_Y, BACK_TOP_Z),
)
SIDE_NEG_X = Box(
    "side_neg_x",
    lo=(-HOLD_HALF_WIDTH, HOLD_FRONT_Y + SIDE_FRONT_TRIM, HOLD_BOTTOM_Z + FIVE_BOX_BOTTOM_TRIM),
    hi=(-HOLD_HALF_WIDTH + SIDE_THICKNESS, HOLD_BACK_Y, BACK_TOP_Z),
)
# Upright posts at the untilted sides' place (then moved to be tangent, below), as deep as the sides.
SIDE_POST_HEIGHT = 0.003
SIDE_POST_POS_X = Box(
    "side_post_pos_x",
    lo=(SIDE_POS_X.lo[0], SIDE_POS_X.lo[1], SIDE_POS_X.hi[2] - SIDE_POST_HEIGHT),
    hi=SIDE_POS_X.hi,
)
SIDE_POST_NEG_X = Box(
    "side_post_neg_x",
    lo=(SIDE_NEG_X.lo[0], SIDE_NEG_X.lo[1], SIDE_NEG_X.hi[2] - SIDE_POST_HEIGHT),
    hi=SIDE_NEG_X.hi,
)



def tilted(box: Box, axis, angle_deg: float, pivot) -> Box:
    """``box`` rotated by ``angle_deg`` about ``axis`` (hold frame, right-handed) through ``pivot``."""
    a = np.radians(angle_deg) / 2
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    quat = (float(np.cos(a)), *(float(v) for v in np.sin(a) * u))
    return replace(box, quat=quat, pivot=tuple(float(v) for v in pivot))


# The sides and the lip lean outwards (their tops away from the ball), hinged on their bottom inner edge.
FIVE_BOX_TILT_DEG = 15.0
TILTED_WALL_THINNING = 0.0005  # taken off the outer face of the lip and the sides (the inner face stays tangent)
SIDE_POS_X = replace(SIDE_POS_X, hi=(SIDE_POS_X.hi[0] - TILTED_WALL_THINNING, *SIDE_POS_X.hi[1:]))
SIDE_NEG_X = replace(SIDE_NEG_X, lo=(SIDE_NEG_X.lo[0] + TILTED_WALL_THINNING, *SIDE_NEG_X.lo[1:]))
LIP_TOP_TRIM = 0.003  # the tilted lip's top is this much below LIP_TOP_Z (before the tilt)
LIP_5 = replace(LIP_5, lo=(LIP_5.lo[0], LIP_5.lo[1] + TILTED_WALL_THINNING, LIP_5.lo[2]),
                hi=(LIP_5.hi[0], LIP_5.hi[1], LIP_5.hi[2] - LIP_TOP_TRIM))
SIDE_POS_X_T = tilted(SIDE_POS_X, (0, 1, 0), FIVE_BOX_TILT_DEG, (SIDE_POS_X.lo[0], 0.0, SIDE_POS_X.lo[2]))
SIDE_NEG_X_T = tilted(SIDE_NEG_X, (0, 1, 0), -FIVE_BOX_TILT_DEG, (SIDE_NEG_X.hi[0], 0.0, SIDE_NEG_X.lo[2]))
LIP_5_T = tilted(LIP_5, (1, 0, 0), FIVE_BOX_TILT_DEG, (0.0, LIP_5.hi[1], LIP_5.lo[2]))


def tangent(box: Box, outward, radius: float) -> Box:
    """``box`` (tilted, its ``pivot`` on its inner face) moved along the horizontal ``outward`` so that its inner
    face (the face towards -outward) is at ``radius`` from the socket: tangent to a sphere of that radius."""
    u = np.asarray(outward, float)
    n = box.rot @ -u  # inner face normal, towards the ball
    q = np.array(box.pivot)
    dist = -n @ q  # distance from the socket to the inner face plane
    shift = (dist - radius) / (n @ u) * u
    move = lambda v: tuple(float(x) for x in np.asarray(v) + shift)
    return replace(box, lo=move(box.lo), hi=move(box.hi), pivot=move(box.pivot))


# The tilted lip and sides moved (along x / y only) to be tangent to the ball inflated by CUP_SLACK.
TANGENT_RADIUS = BALL_RADIUS + CUP_SLACK
SIDE_POS_X_T = tangent(SIDE_POS_X_T, (1, 0, 0), TANGENT_RADIUS)
SIDE_NEG_X_T = tangent(SIDE_NEG_X_T, (-1, 0, 0), TANGENT_RADIUS)
LIP_5_T = tangent(LIP_5_T, (0, -1, 0), TANGENT_RADIUS)


def _post_tangent(box: Box, axis: int, sign: int, radius: float) -> Box:
    """The upright ``box``, on the ``sign`` side of the socket along ``axis`` (0: x, 1: y), moved along that axis so
    that its nearest point to the socket is at ``radius``: tangent to the sphere of that radius (on its inner face,
    or on an edge of it)."""
    near = np.clip(np.zeros(3), box.lo, box.hi)  # nearest point, but for the coordinate along ``axis``
    near[axis] = 0.0
    inner = sign * np.sqrt(radius ** 2 - near @ near)  # where the inner face goes
    d = np.zeros(3)
    d[axis] = inner - (box.lo[axis] if sign > 0 else box.hi[axis])
    return replace(box, lo=tuple(float(v) for v in np.array(box.lo) + d), hi=tuple(float(v) for v in np.array(box.hi) + d))


# The posts moved (along y / x only) to be tangent to the ball inflated by CUP_SLACK, like the tilted walls.
FRONT_POST = _post_tangent(FRONT_POST, 1, -1, TANGENT_RADIUS)
SIDE_POST_POS_X = _post_tangent(SIDE_POST_POS_X, 0, +1, TANGENT_RADIUS)
SIDE_POST_NEG_X = _post_tangent(SIDE_POST_NEG_X, 0, -1, TANGENT_RADIUS)

THREE_BOX: tuple[Box, ...] = (LEDGE, LIP, BACK)
FIVE_BOX: tuple[Box, ...] = (LEDGE_5, LIP_5_T, BACK_5, SIDE_POS_X_T, SIDE_NEG_X_T)
# The upright posts added to the 5-box cup. They collide (wall.py), but on their own body outside the "wall"
# subtree, so the wall contact sensors (and the rewards / observations on them) only see the FIVE_BOX boxes.
FIVE_BOX_POSTS: tuple[Box, ...] = (FRONT_POST, SIDE_POST_POS_X, SIDE_POST_NEG_X)
BLOCK = Box(
    "block",
    lo=LEDGE.lo,
    hi=(LEDGE.hi[0], LEDGE.hi[1], LIP_TOP_Z),
)
ONE_BOX: tuple[Box, ...] = (BLOCK,)

# Legacy five boxes (git 8156523 and before 20a0573): the three boxes plus the two sides, upright, untrimmed and
# without posts; the lip, back and sides thinned by FIVE_BOX_THINNING on their inner faces (0.6 mm gap all around
# the ball), the ledge kept whole. Same box names as FIVE_BOX, so wall.py gives them the same collision groups.
FIVE_BOX_LEGACY: tuple[Box, ...] = (
    LEDGE,
    Box("lip", lo=LIP.lo, hi=(LIP.hi[0], LIP.hi[1] - FIVE_BOX_THINNING, LIP.hi[2])),
    Box("back", lo=(BACK.lo[0], BACK.lo[1] + FIVE_BOX_THINNING, BACK.lo[2]), hi=BACK.hi),
    Box(
        "side_pos_x",
        lo=(HOLD_HALF_WIDTH - SIDE_THICKNESS, HOLD_FRONT_Y, HOLD_BOTTOM_Z),
        hi=(HOLD_HALF_WIDTH, HOLD_BACK_Y, BACK_TOP_Z),
    ),
    Box(
        "side_neg_x",
        lo=(-HOLD_HALF_WIDTH, HOLD_FRONT_Y, HOLD_BOTTOM_Z),
        hi=(-HOLD_HALF_WIDTH + SIDE_THICKNESS, HOLD_BACK_Y, BACK_TOP_Z),
    ),
)

APPROXIMATIONS: dict[str, tuple[Box, ...]] = {
    "three_box": THREE_BOX,
    "five_box": FIVE_BOX + FIVE_BOX_POSTS,
    "five_box_legacy": FIVE_BOX_LEGACY,
    "one_box": ONE_BOX,
}
