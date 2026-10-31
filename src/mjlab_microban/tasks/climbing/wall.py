"""The climbing wall: the hackathon wall of the organisers' simulator (climbing_sim), with its routes.

The holds are made by climbing_sim itself, the same way as rock-climbing-workshop/scripts/plan_sim_wall.py
does to plan the reference climbs: ``climbing_sim.routes.generate_wall(level, seed)``, or a bundled wall /
a wall JSON file (``climbing_sim.Wall.load``). The holds are at the simulator's sockets with its visual mesh,
but collide as boxes (hold_boxes.py): the holds the planner uses as a 5-box cup (the ball sits on the socket,
as in the simulator), the others as a 1-box block (see ``make_wall_cfg``). The used holds are stored in the
motion .npz as ``used_holds`` (planner hold ids r<row>_c<col>): plan_sim_wall.py --save-npz writes them, and
scripts/workshop_to_npz.py carries them into the env motion.

Which wall: ``WallSpec`` (level + seed, or a name / JSON file). It is stored in the motion .npz as ``wall``
(text: "2:5" for level 2 seed 5, or a bundled wall name N1-N3 / a wall JSON file, ``WallSpec.parse``):
plan_sim_wall.py --save-npz writes it, workshop_to_npz.py carries it into the env motion. The registered tasks
take it from the motion of the train / play command line (--env.commands.motion.motion-file, see
``wall_spec_from_argv``), so the motion file is all there is to give.

Frames
------
Simulator (workshop) world frame: origin at the bottom-left corner of the wall face, x to the right along
the wall, y into the wall, z up; the climber is at y < 0.
Env frame (env origin): x towards the wall (the robot is at x < 0, facing +x), y to the robot's left, z up.

    p_env = SIM_TO_ENV_ROT @ p_sim + SIM_TO_ENV_OFFSET

a -90 deg yaw, then the offset that puts the hold sockets (the ball centres of a hooked robot, 26 mm out of
the wall face) on the plane x = 0 and the middle of the 0.8 m wide wall on y = 0.

climbing_sim's wall / routes modules (and the assets they need) are copied into this project, in
tasks/climbing/climbing_sim (see its __init__): the simulator checkout is not needed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import EntityCfg

from mjlab_microban.tasks.climbing import hold_boxes

# Simulator world frame -> env frame.
SIM_TO_ENV_ROT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
SIM_TO_ENV_QUAT = np.array([np.cos(-np.pi / 4), 0.0, 0.0, np.sin(-np.pi / 4)])  # wxyz
SOCKET_OUT = 0.026  # climbing_sim.wall.SOCKET_OUT: socket centre out of the wall face
SIM_WALL_WIDTH = 0.800  # climbing_sim.wall.WALL_WIDTH
SIM_TO_ENV_OFFSET = np.array([SOCKET_OUT, SIM_WALL_WIDTH / 2, 0.0])

HOLD_FRICTION = 0.5  # climbing_sim.scene.build_spec's default

# Collision groups (contype / conaffinity bits). Two geoms collide if (contype1 & conaffinity2) or
# (contype2 & conaffinity1); the wall geoms have contype 0, so their conaffinity alone picks the robot geoms:
#   WORLD    every robot collision geom (contype bit 0, robot_climb.xml) <-> the floor, the board, the 1-box
#            blocks and the lips (fronts) of the 5-box cups
#   SELF     robot <-> robot (robot_climb.xml: conaffinity 2), not used by the wall
#   HANDLES  the four hand / foot balls (contype |= HANDLES, see the climbing env) <-> all five boxes of the
#            5-box cups
# So the body boxes never reach the inside of a cup (ledge, back, sides), and the narrowphase only sees
# the pairs that matter.
#
#                     board  1-box  5-box lip  5-box ledge/back/sides
#   balls               x      x       x               x
#   other robot geoms   x      x       x
COLLISION_WORLD = 1
COLLISION_SELF = 2
COLLISION_HANDLES = 4
# Body (child of "wall") carrying the 1-box blocks of the holds the motion does not use.
ONE_BOX_BODY = "one_box_blocks"
# 5-box cup boxes in the WORLD group (besides HANDLES): the front of the cup.
FIVE_BOX_WORLD_BOXES = ("lip",)

DEFAULT_WALL = "2:0"  # without a motion on the command line: level 2, seed 0, the competition wall N2

# The env's motion on the train / play command line (its used holds collide as 5-box cups, see make_wall_cfg).
MOTION_FILE_FLAGS = ("--env.commands.motion.motion-file", "--env.commands.motion.motion_file")
CONTACT_PLAN_SCHEMA = "contact_plan/1"  # rock-climbing-workshop: rcr.io


# ---------------------------------------------------------------------------------------------------
# climbing_sim
# ---------------------------------------------------------------------------------------------------


def _climbing_sim():
    """The climbing_sim modules used here: (wall, routes), from the copy in tasks/climbing/climbing_sim."""
    from mjlab_microban.tasks.climbing.climbing_sim import routes, wall

    return wall, routes


@dataclass(frozen=True)
class WallSpec:
    """A route of climbing_sim: ``generate_wall(level, seed)``, or ``Wall.load(name)`` when name is given."""

    level: int = 2
    seed: int = 0
    name: str | None = None  # a bundled wall (N1, N2, N3) or a wall JSON file

    @classmethod
    def parse(cls, text: str) -> WallSpec:
        """'level:seed' (e.g. '2:5'), or a bundled wall name / JSON file."""
        level, sep, seed = text.partition(":")
        if sep and level.strip().isdigit() and seed.strip().isdigit():
            return cls(level=int(level), seed=int(seed))
        return cls(name=text)

    def __str__(self) -> str:
        return self.name if self.name else f"level {self.level}, seed {self.seed}"

    def text(self) -> str:
        """The inverse of ``parse``: what a motion .npz stores as ``wall``."""
        return self.name if self.name else f"{self.level}:{self.seed}"


def motion_wall(motion_file: str | Path) -> WallSpec:
    """The wall a motion .npz was planned on (its ``wall``); a motion without it is an error."""
    data = np.load(motion_file)
    if "wall" not in data.files:
        raise ValueError(
            f"{motion_file} has no 'wall': re-plan it with rock-climbing-workshop/scripts/plan_sim_wall.py "
            "--save-npz (it stores it), then convert it with scripts/workshop_to_npz.py"
        )
    return WallSpec.parse(str(data["wall"]))


def resolve_wall(motion_file: str | Path, level: int | None = None, seed: int | None = None,
                 name: str | None = None) -> WallSpec:
    """The motion's wall, or the one given (level + seed, or name), which then has to be the motion's."""
    if name is None and level is None and seed is None:
        return motion_wall(motion_file)
    given = WallSpec(name=name) if name else WallSpec(level=2 if level is None else level, seed=seed or 0)
    data = np.load(motion_file)
    if "wall" in data.files and WallSpec.parse(str(data["wall"])) != given:
        raise ValueError(f"{motion_file} was planned on wall {WallSpec.parse(str(data['wall']))}, not {given}")
    return given


@lru_cache(maxsize=1)
def wall_spec_from_argv() -> WallSpec:
    """The wall of the train / play command line's motion (--env.commands.motion.motion-file); without one,
    DEFAULT_WALL (level 2, seed 0 = N2)."""
    path = motion_file_from_argv()
    if path and Path(path).exists():
        return motion_wall(path)
    print(f"[wall] no motion file on the command line: wall {WallSpec.parse(DEFAULT_WALL)}")
    return WallSpec.parse(DEFAULT_WALL)


@lru_cache(maxsize=None)
def sim_wall(spec: WallSpec):
    """The climbing_sim Wall, made as rock-climbing-workshop/scripts/plan_sim_wall.py makes it."""
    wall_mod, routes = _climbing_sim()
    if spec.name:
        return wall_mod.Wall.load(spec.name)
    return routes.generate_wall(spec.level, spec.seed)


def sim_to_env(p) -> np.ndarray:
    """Points (..., 3) from the simulator world frame to the env frame."""
    return np.asarray(p, float) @ SIM_TO_ENV_ROT.T + SIM_TO_ENV_OFFSET


def hold_positions(spec: WallSpec) -> np.ndarray:
    """(N, 3) hold sockets (centre of a ball hooked in the hold) in the env frame, in the wall's hold order."""
    return sim_to_env(sim_wall(spec).sockets())


def start_positions(spec: WallSpec) -> dict[str, np.ndarray]:
    """Limb -> socket of its start hold, in the env frame."""
    return {limb: sim_to_env(h.socket) for limb, h in sim_wall(spec).start.items()}


# ---------------------------------------------------------------------------------------------------
# Holds used by the motion
# ---------------------------------------------------------------------------------------------------


def motion_file_from_argv() -> str | None:
    """The motion the env tracks, from the train / play command line (--env.commands.motion.motion-file), or None.
    The wall is built before the motion command, from the command line, so it cannot read it from the cfg."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        for flag in MOTION_FILE_FLAGS:
            if arg == flag and i + 1 < len(argv):
                return argv[i + 1]
            if arg.startswith(flag + "="):
                return arg[len(flag) + 1 :]
    return None


def hold_id(hold) -> str:
    """The planner's id of a climbing_sim hold (rock-climbing-workshop/scripts/plan_sim_wall.py: hold_id)."""
    return f"r{hold.row}_c{hold.col}"


def contact_plan_hold_ids(plan_file: str | Path) -> list[str]:
    """The holds of a contact plan (plan_sim_wall.py --save): every hold a limb is on in some stance (the first
    stance is the start), sorted. A limb on nothing (None) or on the ground is not a hold."""
    data = json.loads(Path(plan_file).read_text())
    if data.get("schema") != CONTACT_PLAN_SCHEMA:
        raise ValueError(f"{plan_file}: expected schema {CONTACT_PLAN_SCHEMA!r}, found {data.get('schema')!r}")
    ids = {h for stance in data["stances"] for h in stance["contacts"].values() if h not in (None, "ground")}
    return sorted(ids)


def hold_indices(spec: WallSpec, ids) -> frozenset[int]:
    """Indices (wall.holds order) of the holds with these planner ids; an id not on the wall is an error."""
    index = {hold_id(h): k for k, h in enumerate(sim_wall(spec).holds)}
    missing = sorted(set(ids) - set(index))
    if missing:
        raise ValueError(f"holds {missing} are not on wall {spec}: planned on another wall?")
    return frozenset(index[i] for i in ids)


@lru_cache(maxsize=None)
def used_holds(spec: WallSpec, motion_file: str) -> frozenset[int]:
    """Indices (wall.holds order) of the holds the planner uses, from the motion's ``used_holds`` (planner ids):
    an env motion (workshop_to_npz.py) or the planner's own trajectory (plan_sim_wall.py --save-npz)."""
    return hold_indices(spec, motion_used_hold_ids(motion_file))


def motion_used_hold_ids(motion_file: str | Path) -> list[str]:
    """The ``used_holds`` of a motion .npz; a motion without them is an error."""
    data = np.load(motion_file)
    if "used_holds" not in data.files:
        raise ValueError(
            f"{motion_file} has no 'used_holds': re-plan it with rock-climbing-workshop/scripts/plan_sim_wall.py "
            "--save-npz (it stores them), then convert it with scripts/workshop_to_npz.py"
        )
    return [str(i) for i in data["used_holds"]]


# ---------------------------------------------------------------------------------------------------
# MuJoCo
# ---------------------------------------------------------------------------------------------------


def _add_hold_boxes(body, name: str, socket: np.ndarray, boxes, friction, rgba, conaffinity) -> None:
    """conaffinity: box name -> its collision groups (contype 0, see COLLISION_WORLD)."""
    quat = SIM_TO_ENV_QUAT.tolist()
    for b in boxes:
        body.add_geom(
            name=f"{name}_{b.name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=b.half_size.tolist(),
            pos=(socket + SIM_TO_ENV_ROT @ b.center).tolist(),
            quat=quat,
            group=3,
            rgba=rgba,
            friction=friction,
            condim=3,
            contype=0,
            conaffinity=conaffinity(b.name),
        )


def _five_box_conaffinity(box_name: str) -> int:
    return COLLISION_HANDLES | (COLLISION_WORLD if box_name in FIVE_BOX_WORLD_BOXES else 0)


def _one_box_conaffinity(box_name: str) -> int:
    return COLLISION_WORLD


def get_spec(spec: WallSpec, used: frozenset[int] | None = None) -> mujoco.MjSpec:
    """The board and the holds of climbing_sim.scene.build_spec, in the env frame, all in one body "wall"
    (the ball contact sensor matches its geoms).

    The holds keep climbing_sim's visual mesh, but collide as boxes (hold_boxes.py): the holds in ``used``
    (those the motion puts a ball on, see ``used_holds``) as the 5-box cup, the others as the 1-box block.
    used=None: every hold is a 5-box cup. Collision groups: see COLLISION_WORLD."""
    wall_mod, _ = _climbing_sim()
    wall = sim_wall(spec)
    assets = Path(wall_mod.__file__).parent / "assets" / "hold"
    friction = [HOLD_FRICTION, 0.005, 0.0001]

    mj = mujoco.MjSpec()
    mj.add_material(name="wall_wood", rgba=[0.86, 0.75, 0.56, 1.0])
    mj.add_material(name="wall_hold", rgba=[0.85, 0.35, 0.15, 1.0])
    mj.add_mesh(name="hold_visual", file=str(assets / "hold_visual.stl"))

    body = mj.worldbody.add_body(name="wall")
    # Board: climbing_sim's box (0 <= x <= W, 0 <= y <= thickness, 0 <= z <= H in the simulator frame).
    w, t, h = wall_mod.WALL_WIDTH, wall_mod.WALL_THICKNESS, wall_mod.WALL_HEIGHT
    body.add_geom(
        name="board",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[t / 2, w / 2, h / 2],  # rotated: env x = sim y, env y = -sim x
        pos=sim_to_env([w / 2, t / 2, h / 2]).tolist(),
        material="wall_wood",
        condim=3,
        friction=friction,
        contype=0,
        conaffinity=COLLISION_WORLD,
    )

    quat = SIM_TO_ENV_QUAT.tolist()
    # The 1-box blocks are on a child body of the wall (fixed to it, no joint), so that a contact sensor can single
    # them out (ONE_BOX_BODY: the climbing env penalises the balls touching them).
    blocks_body = body.add_body(name=ONE_BOX_BODY)
    for k, hold in enumerate(wall.holds):
        socket = sim_to_env(hold.socket)
        name = f"hold{k}_c{hold.col}_r{hold.row}"
        body.add_site(name=f"hold{k}_socket", pos=socket.tolist(), size=[0.002, 0, 0], rgba=[0.1, 0.9, 0.2, 1], group=4)
        body.add_geom(
            name=f"{name}_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="hold_visual",
            pos=socket.tolist(),
            quat=quat,
            contype=0,
            conaffinity=0,
            group=1,
            material="wall_hold",
        )
        if used is None or k in used:
            _add_hold_boxes(body, name, socket, hold_boxes.FIVE_BOX, friction, rgba=[0.25, 0.55, 0.95, 1],
                            conaffinity=_five_box_conaffinity)
        else:
            _add_hold_boxes(blocks_body, name, socket, hold_boxes.ONE_BOX, friction, rgba=[0.7, 0.3, 0.85, 1],
                            conaffinity=_one_box_conaffinity)
    return mj


def make_wall_cfg(spec: WallSpec, motion_file: str | None = None) -> EntityCfg:
    """Fixed, non-articulated entity: mjlab wraps it in a mocap body, placed at each env origin by the
    "reset_wall" event of the climbing env.

    motion_file: the motion whose used holds (its ``used_holds``, from the planner) are 5-box cups, the others
    1-box blocks; None: the one of the train / play command line (``motion_file_from_argv``), looked up when the
    scene is built. Without a motion file, every hold is a 5-box cup; a motion without ``used_holds`` is an
    error."""

    def spec_fn() -> mujoco.MjSpec:
        path = motion_file or motion_file_from_argv()
        if not path or not Path(path).exists():
            print(f"[wall] no motion file ({path!r}): every hold collides as the 5-box cup")
            return get_spec(spec)
        planned_on = motion_wall(path)
        if planned_on != spec:
            raise ValueError(f"{path} was planned on wall {planned_on}, the env's wall is {spec}")
        used = used_holds(spec, str(Path(path).resolve()))
        n = len(sim_wall(spec).holds)
        print(f"[wall] {Path(path).name}: {len(used)}/{n} holds used (5-box cup), {n - len(used)} unused (1-box block)")
        return get_spec(spec, used)

    return EntityCfg(spec_fn=spec_fn)
