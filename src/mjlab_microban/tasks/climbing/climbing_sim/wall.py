# SPDX-License-Identifier: GPL-3.0-or-later
"""The hackathon wall (see the rulebook): an 800 x 1400 mm board standing on the floor, with a grid of
52 x 93 M3 holes at a 15 mm pitch (17.5 mm side margins, 10 mm top and bottom margins).

Frames
------
World (MuJoCo) frame: origin at the bottom-left corner of the wall's front face seen from the climbing
side, x to the right along the wall, z up, y into the wall.  The climber is on the y < 0 side, the
board occupies 0 <= y <= 10 mm, the floor is z = 0.

Wall frame (the frame of the ArUco markers and of the camera extrinsics): same origin, x to the right,
y up, z out of the wall towards the climber.  ``wall_to_world`` / ``world_to_wall`` convert points.

Holds
-----
A hold is bolted through two grid holes two pitches apart on the same row: ``Hold(col, row)`` gives
its LEFT hole (seen from the climbing side), its right hole is (col + 2, row).  Columns count from 0
at the left, rows from 0 at the bottom.  The robot's balls sit in the hold's spherical socket, 26 mm
out of the wall and 14 mm below the bolt axis, centred between the two holes.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np

from . import aruco

# all lengths in metres
WALL_WIDTH = 0.800
WALL_HEIGHT = 1.400
WALL_THICKNESS = 0.010
HOLE_PITCH = 0.015
GRID_COLS, GRID_ROWS = 52, 93
GRID_X0, GRID_Z0 = 0.0175, 0.010          # centre of hole (0, 0) from the wall's bottom-left corner
HOLE_DIAMETER = 0.0032

SOCKET_OUT = 0.026                         # socket centre: out of the wall face
SOCKET_BELOW = 0.014                       # socket centre: below the bolt axis

#: (d_col, d_row) between two holds whose bodies overlap or that share a hole (the hold's real outline: flat and
#: wide at the top, rounded at the bottom, so a hold two rows lower tucks under its neighbour's corner)
OVERLAP = frozenset((dc, dr) for dc in range(-2, 3) for dr in range(-2, 3) if not (abs(dc) == 2 and abs(dr) == 2))
#: (rows up, cols across) at which a hold blocks the vertical way out of the hold below it: a ball can only leave
#: a socket within ~10 deg of vertical, and it touches a hold 45 mm above within 15 mm sideways, or 60 mm above
EXIT_BLOCK = frozenset({(3, -1), (3, 0), (3, 1), (4, 0)})

#: the hold's outline on the board around its socket (x right, z up, m), measured on the part
HOLD_OUTLINE = np.array(json.loads((pathlib.Path(__file__).parent / "assets" / "hold" / "outline.json").read_text())) / 1000


def marker_zones() -> dict[int, tuple[float, float, float, float]]:
    """Corner marker ID -> (x0, z0, x1, z1) of its white plate on the board (m): no hold may cover it."""
    out = {}
    for mid in aruco.CORNERS:
        x0, z0 = aruco.corner_plate_origin(mid, WALL_WIDTH, WALL_HEIGHT)
        out[mid] = (float(x0), float(z0), float(x0 + aruco.PLATE), float(z0 + aruco.PLATE))
    return out


def _covers(poly: np.ndarray, rect) -> bool:
    """True when the polygon (N, 2) and the rectangle (x0, z0, x1, z1) overlap."""
    x0, z0, x1, z1 = rect
    edges = np.concatenate([np.linspace(a, b, 40) for a, b in zip(poly, np.roll(poly, -1, axis=0))])
    if np.any((edges[:, 0] > x0) & (edges[:, 0] < x1) & (edges[:, 1] > z0) & (edges[:, 1] < z1)):
        return True

    def inside(px, pz):                        # a rectangle corner inside the polygon (even-odd rule)
        x, z = poly[:, 0], poly[:, 1]
        xn, zn = np.roll(x, -1), np.roll(z, -1)
        cross = ((z > pz) != (zn > pz)) & (px < (xn - x) * (pz - z) / (zn - z + 1e-12) + x)
        return bool(cross.sum() % 2)
    return any(inside(px, pz) for px in (x0, x1) for pz in (z0, z1))


#: the bundled walls: the rulebook's competition walls (JSON files)
WALLS_DIR = pathlib.Path(__file__).parent / "walls"

# wall frame axes expressed in the world frame (columns): x_w = x, y_w = z, z_w = -y
_R_WALL = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def wall_to_world(p):
    """Points (..., 3) from the wall frame (x right, y up, z towards the climber) to the world frame."""
    return np.asarray(p, float) @ _R_WALL.T


def world_to_wall(p):
    """Points (..., 3) from the world frame to the wall frame (the inverse of :func:`wall_to_world`)."""
    return np.asarray(p, float) @ _R_WALL


def hole_position(col: int, row: int) -> np.ndarray:
    """World position of the centre of grid hole (col, row) on the wall's front face."""
    if not (0 <= col < GRID_COLS and 0 <= row < GRID_ROWS):
        raise ValueError(f"hole ({col}, {row}) outside the {GRID_COLS} x {GRID_ROWS} grid")
    return np.array([GRID_X0 + col * HOLE_PITCH, 0.0, GRID_Z0 + row * HOLE_PITCH])


@dataclass(frozen=True)
class Hold:
    """A hold bolted on holes (col, row) and (col + 2, row)."""
    col: int
    row: int

    def __post_init__(self):          # both holes must be on the grid
        hole_position(self.col, self.row)
        hole_position(self.col + 2, self.row)

    @property
    def holes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """The two grid holes (col, row) the hold is bolted on."""
        return (self.col, self.row), (self.col + 2, self.row)

    @property
    def bolt_axis(self) -> np.ndarray:
        """World position of the middle of the two bolts on the wall face."""
        return (hole_position(self.col, self.row) + hole_position(self.col + 2, self.row)) / 2

    @property
    def socket(self) -> np.ndarray:
        """World position of the centre of the socket (where a ball end rests when hooked)."""
        return self.bolt_axis + np.array([0.0, -SOCKET_OUT, -SOCKET_BELOW])

    def outline(self) -> np.ndarray:
        """The hold's body on the board: its outline (N, 2) as (x, z) in the world frame (m)."""
        s = self.socket
        return HOLD_OUTLINE + [s[0], s[2]]

    def on_markers(self) -> list[int]:
        """IDs of the corner markers whose plate the hold's body would cover (the hold cannot go there)."""
        return [mid for mid, rect in marker_zones().items() if _covers(self.outline(), rect)]


@dataclass
class Wall:
    """A set of holds on the board, in climbing order when it is a route.

    ``start`` gives the hold of each limb (``"left_hand"``, ``"right_hand"``, ``"left_foot"``, ``"right_foot"``)
    at the start, where :meth:`ClimbingSim.reset_hanging` hangs the robot.

    JSON form (wall files, the wall editor): ``{"name", "description", "holds": [{"col", "row"}, ...],
    "start": {limb: {"col", "row"}}}``.
    """
    name: str = "custom"
    holds: list[Hold] = field(default_factory=list)
    start: dict[str, Hold] = field(default_factory=dict)       # limb -> start hold (optional)
    description: str = ""

    def add(self, col: int, row: int) -> Hold:
        """Add the hold (col, row); ValueError if it overlaps a hold already there or a corner marker."""
        h = Hold(col, row)
        clash = self.clashes(h)
        if clash:
            raise ValueError(f"hold {h} overlaps {clash}")
        if h.on_markers():
            raise ValueError(f"hold {h} covers the corner marker(s) {h.on_markers()}")
        self.holds.append(h)
        return h

    def clashes(self, h: Hold) -> list[Hold]:
        """Holds that share a hole with h or whose body overlaps h's (they cannot both be bolted)."""
        return [o for o in self.holds if o != h and (o.col - h.col, o.row - h.row) in OVERLAP]

    def exit_blocks(self, h: Hold) -> list[Hold]:
        """Holds that block the way out of h's socket, or whose way out h blocks (both can be bolted, but a ball
        cannot leave the lower one)."""
        out = []
        for o in self.holds:
            dr, dc = o.row - h.row, o.col - h.col
            if o != h and ((dr, dc) in EXIT_BLOCK or (-dr, -dc) in EXIT_BLOCK):
                out.append(o)
        return out

    def sockets(self) -> np.ndarray:
        """World positions of the holds' sockets, (N, 3) in the order of ``holds``."""
        return np.array([h.socket for h in self.holds]).reshape(-1, 3)

    # ---- JSON ----------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict, default_name: str = "custom") -> "Wall":
        """A wall from its JSON form (duplicate holds dropped; a start hold missing from the list is added)."""
        holds, seen = [], set()
        for h in d.get("holds", []):
            k = (int(h["col"]), int(h["row"]))
            if k not in seen:
                seen.add(k); holds.append(Hold(*k))
        start = {l: Hold(int(h["col"]), int(h["row"])) for l, h in d.get("start", {}).items()}
        for s in start.values():
            if s not in holds:
                holds.insert(0, s)
        return cls(d.get("name") or default_name, holds, start, d.get("description", ""))

    def to_dict(self) -> dict:
        """The JSON form of the wall."""
        return {"name": self.name, "description": self.description,
                "holds": [{"col": h.col, "row": h.row} for h in self.holds],
                "start": {l: {"col": h.col, "row": h.row} for l, h in self.start.items()}}

    @classmethod
    def from_json(cls, path) -> "Wall":
        """Read a wall file (named after the file when it has no name)."""
        return cls.from_dict(json.loads(pathlib.Path(path).read_text()), pathlib.Path(path).stem)

    def to_json(self, path):
        """Write the wall file."""
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, name: str) -> "Wall":
        """A bundled wall (a name of :func:`bundled_walls`, e.g. "N2") or a path to a JSON file."""
        p = pathlib.Path(name)
        if not p.suffix:
            p = WALLS_DIR / f"{name}.json"
        return cls.from_json(p)


def bundled_walls() -> list[str]:
    """Names of the bundled walls (the rulebook's competition walls)."""
    return sorted(p.stem for p in WALLS_DIR.glob("*.json"))
