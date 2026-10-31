"""Show the climbing Microban (ball hands and feet) in viser with its collision geoms.

The model is compiled through MICROBAN_CLIMB_ROBOT_CFG, so the collision geoms shown are exactly those
used in training (contype or conaffinity non-zero), in the climb start pose. Geoms are coloured with their
MJCF rgba (blue: trunk, forearm, thigh and sole boxes, yellow: head and servo boxes, cyan: hook spheres).
The default model has 14 collision primitives: trunk and head boxes, two arm servo boxes, two forearm boxes,
two thigh boxes, four hook spheres, and two sole boxes. The head box follows the head joint and touches the trunk box.
Each arm has one merged servo box, enlarged by 3% in each dimension.
Each forearm box spans the elbow joint and the full hand sphere defined by the velocity model.
Each thigh has one box covering its full width, length and thickness.

Collision geoms are named "trunk_box_coll", "head_box_coll", "*_servo_coll", "<side>_<hand|foot>_ball_coll",
"<side>_forearm_box_coll", "<side>_thigh_box_coll", and "<side>_foot_sole_coll".
For alternate models with numbered pieces, labels and highlighting group the pieces of each part.

GUI:
  - Display: toggle visual meshes / collision geoms / labels / joint frames, collision opacity
    (joint frames: a small x=red, y=green, z=blue triad at each joint anchor, in the joint's body frame;
    the four ball-site frames at the hand/foot sphere centres are always shown)
  - Highlight: pick a part (or a whole segment) to show in red
  - Joints: move every joint; the self-contacts of the current pose are listed below
  - Collision geoms: table of every geom with contype/conaffinity/condim

--xml loads another climbing MJCF (same bodies and joints) in place of robot_climb.xml; --compare overlays
the collision geoms of a second one, in red, e.g. the old detailed primitive approximation:

    uv run python src/mjlab_microban/scripts/view_climb_collisions.py \
        --compare src/mjlab_microban/robot/microban_climb/robot_climb_primitives.xml

Usage:
    uv run python src/mjlab_microban/scripts/view_climb_collisions.py [--port 8080] [--xml X] [--compare Y]
"""

import argparse
import re
from pathlib import Path
import time
from collections import defaultdict

import mujoco
import numpy as np
import viser
from mjlab.entity import Entity
from mjviser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh

from mjlab_microban.robot import microban_climb_constants
from mjlab_microban.robot.microban_climb_constants import (
    BALL_SITE_NAMES,
    CLIMB_START_FRAME,
    MICROBAN_CLIMB_ROBOT_CFG,
)

VISUAL_GROUP = 2
HIGHLIGHT_RGB = np.array([235, 50, 50], dtype=np.uint8)
NONE = "(none)"
JOINT_FRAME_LENGTH = 0.04  # m, axis length of the per-joint coordinate frames
BALL_FRAME_LENGTH = 0.025  # m, axis length of the always-on ball-site frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8080, help="Viser server port (default: 8080)")
    p.add_argument("--xml", type=Path, default=None, help="Climbing MJCF to show (default: robot_climb.xml)")
    p.add_argument("--compare", type=Path, default=None, help="Second MJCF whose collision geoms are overlaid in red")
    return p.parse_args()


COMPARE_RGBA = np.array([235, 60, 60, 90], dtype=np.uint8)


def compile_climb_model(xml: Path | None) -> mujoco.MjModel:
    """The climbing robot compiled through MICROBAN_CLIMB_ROBOT_CFG, from another MJCF if given."""
    default = microban_climb_constants.MICROBAN_CLIMB_XML
    if xml is not None:
        microban_climb_constants.MICROBAN_CLIMB_XML = xml.resolve()  # read by get_spec() at call time
    try:
        return Entity(MICROBAN_CLIMB_ROBOT_CFG).spec.compile()
    finally:
        microban_climb_constants.MICROBAN_CLIMB_XML = default


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name:
        return name
    body_id = model.geom_bodyid[geom_id]
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"
    return f"{body}/geom{geom_id - model.body_geomadr[body_id]}"


def part_name(name: str) -> str:
    """'right_thigh_hull_03_coll' -> 'right_thigh_hull'; 'right_hand_ball_coll' -> 'right_hand_ball'."""
    return re.sub(r"(_\d\d)?_coll$", "", name)


def segment_name(model: mujoco.MjModel, geom_id: int, part: str) -> str:
    """The '<side>_<segment>' prefix of a part ('right_hip_block_spacer' -> 'right_hip_block'); the body
    name for geoms outside the naming scheme."""
    for seg in SEGMENTS:
        if part == seg or part.startswith(seg + "_"):
            return seg
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or "?"


# Longest first, so "right_hip_block" wins over "right_hip".
SEGMENTS = sorted(
    ["trunk", "head"]
    + [
        f"{side}_{seg}"
        for side in ("left", "right")
        for seg in ("shoulder", "upper_arm", "forearm", "hand", "hip", "hip_block", "thigh", "shin",
                    "ankle_block", "foot")
    ],
    key=len,
    reverse=True,
)


def set_start_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[:] = model.qpos0
    for joint_name, value in CLIMB_START_FRAME.joint_pos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def geom_pose(data: mujoco.MjData, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(wxyz, data.geom_xmat[geom_id])
    return data.geom_xpos[geom_id].copy(), wxyz


def self_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[str, str, float]]:
    """(part 1, part 2, penetration depth) of the current contacts, deepest first, one row per part pair."""
    deepest: dict[tuple[str, str], float] = {}
    for c in data.contact[: data.ncon]:
        pair = tuple(sorted((part_name(geom_name(model, c.geom1)), part_name(geom_name(model, c.geom2)))))
        deepest[pair] = max(deepest.get(pair, 0.0), -c.dist)
    return sorted(((a, b, d) for (a, b), d in deepest.items()), key=lambda r: -r[2])


def main() -> None:
    args = parse_args()

    model = compile_climb_model(args.xml)
    data = mujoco.MjData(model)
    set_start_pose(model, data)
    if args.compare is not None:
        ref_model = compile_climb_model(args.compare)
        assert ref_model.nq == model.nq, "--compare model has different joints"
        ref_data = mujoco.MjData(ref_model)

    collision_ids = [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    visual_ids = [i for i in range(model.ngeom) if model.geom_group[i] == VISUAL_GROUP]
    parts: dict[str, list[int]] = defaultdict(list)
    segments: dict[str, list[int]] = defaultdict(list)
    for i in collision_ids:
        part = part_name(geom_name(model, i))
        parts[part].append(i)
        segments[segment_name(model, i, part)].append(i)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=1.0, height=1.0, cell_size=0.05)

    # Visual meshes.
    visual_handles: dict[int, viser.SceneNodeHandle] = {}
    for i in visual_ids:
        pos, wxyz = geom_pose(data, i)
        visual_handles[i] = server.scene.add_mesh_trimesh(
            f"/visual/{i}", mujoco_mesh_to_trimesh(model, i), position=pos, wxyz=wxyz
        )

    # Collision geoms; rebuilt when the opacity or the highlight changes.
    def collision_mesh(i: int):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            return mujoco_mesh_to_trimesh(model, i)
        return create_primitive_mesh(model, i)

    collision_meshes = {i: collision_mesh(i) for i in collision_ids}
    collision_handles: dict[int, viser.SceneNodeHandle] = {}

    def draw_collisions(alpha: float, highlighted: set[int], visible: bool) -> None:
        for i in collision_ids:
            mesh = collision_meshes[i]
            rgb = HIGHLIGHT_RGB if i in highlighted else (model.geom_rgba[i, :3] * 255).astype(np.uint8)
            a = 230 if i in highlighted else int(alpha * 255)
            mesh.visual.vertex_colors = np.tile(np.append(rgb, a).astype(np.uint8), (len(mesh.vertices), 1))
            pos, wxyz = geom_pose(data, i)
            collision_handles[i] = server.scene.add_mesh_trimesh(
                f"/collision/{i}", mesh, position=pos, wxyz=wxyz, visible=visible
            )

    # Collision geoms of the --compare model, overlaid in red, following the same joint positions.
    ref_handles: dict[int, viser.SceneNodeHandle] = {}
    if args.compare is not None:
        ref_data.qpos[:] = data.qpos
        mujoco.mj_forward(ref_model, ref_data)
        for i in range(ref_model.ngeom):
            if not (ref_model.geom_contype[i] or ref_model.geom_conaffinity[i]):
                continue
            if ref_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = mujoco_mesh_to_trimesh(ref_model, i)
            else:
                mesh = create_primitive_mesh(ref_model, i)
            mesh.visual.vertex_colors = np.tile(COMPARE_RGBA, (len(mesh.vertices), 1))
            pos, wxyz = geom_pose(ref_data, i)
            ref_handles[i] = server.scene.add_mesh_trimesh(f"/compare/{i}", mesh, position=pos, wxyz=wxyz)
        print(f"--compare: {len(ref_handles)} collision geoms of {args.compare} overlaid in red")

    # One label per part, at the centre of its pieces.
    label_handles: dict[str, viser.LabelHandle] = {}

    def part_centre(ids: list[int]) -> np.ndarray:
        return data.geom_xpos[ids].mean(axis=0)

    for part, ids in parts.items():
        label_handles[part] = server.scene.add_label(f"/labels/{part}", part, position=part_centre(ids))

    # A small RGB (x/y/z) frame at each hinge/slide joint anchor, oriented like the joint's body frame
    # (the frame in which the MJCF joint axis is expressed).
    joint_ids = [
        j for j in range(model.njnt) if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    ]
    joint_frame_handles: dict[int, viser.FrameHandle] = {}

    def joint_pose(j: int) -> tuple[np.ndarray, np.ndarray]:
        return data.xanchor[j].copy(), data.xquat[model.jnt_bodyid[j]].copy()

    for j in joint_ids:
        pos, wxyz = joint_pose(j)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        joint_frame_handles[j] = server.scene.add_frame(
            f"/joint_frames/{name}", position=pos, wxyz=wxyz, axes_length=JOINT_FRAME_LENGTH,
            axes_radius=JOINT_FRAME_LENGTH / 20,
        )

    # Always-on frames at the ball sites (sphere centres, with the site orientation).
    ball_site_ids = [
        s for s in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in BALL_SITE_NAMES) if s >= 0
    ]
    ball_frame_handles: dict[int, viser.FrameHandle] = {}

    def site_pose(s: int) -> tuple[np.ndarray, np.ndarray]:
        wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(wxyz, data.site_xmat[s])
        return data.site_xpos[s].copy(), wxyz

    for s in ball_site_ids:
        pos, wxyz = site_pose(s)
        ball_frame_handles[s] = server.scene.add_frame(
            f"/ball_frames/{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s)}", position=pos, wxyz=wxyz,
            axes_length=BALL_FRAME_LENGTH, axes_radius=BALL_FRAME_LENGTH / 20,
        )

    print(f"{len(collision_ids)} collision geoms in {len(parts)} parts:")
    for part, ids in parts.items():
        print(f"  {part:42s} {len(ids):3d} geom(s)")

    # ------------------------------------------------------------------ GUI
    with server.gui.add_folder("Display"):
        show_visual = server.gui.add_checkbox("Visual meshes", initial_value=True)
        show_collision = server.gui.add_checkbox("Collision geoms", initial_value=True)
        show_labels = server.gui.add_checkbox("Part labels", initial_value=False)
        show_joint_frames = server.gui.add_checkbox("Joint frames", initial_value=True)

        @show_joint_frames.on_update
        def _(_) -> None:
            for handle in joint_frame_handles.values():
                handle.visible = show_joint_frames.value

        opacity = server.gui.add_slider("Collision opacity", min=0.05, max=1.0, step=0.05, initial_value=0.55)
        if ref_handles:
            show_ref = server.gui.add_checkbox(f"Compare: {args.compare.name} (red)", initial_value=True)

            @show_ref.on_update
            def _(_) -> None:
                for handle in ref_handles.values():
                    handle.visible = show_ref.value

    with server.gui.add_folder("Highlight"):
        segment_dd = server.gui.add_dropdown("Segment", options=[NONE, *sorted(segments)], initial_value=NONE)
        part_dd = server.gui.add_dropdown("Part", options=[NONE, *sorted(parts)], initial_value=NONE)
        highlight_info = server.gui.add_markdown("")

    def highlighted_ids() -> set[int]:
        if part_dd.value != NONE:
            return set(parts[part_dd.value])
        if segment_dd.value != NONE:
            return set(segments[segment_dd.value])
        return set()

    def redraw() -> None:
        ids = highlighted_ids()
        draw_collisions(opacity.value, ids, show_collision.value)
        for part, handle in label_handles.items():
            handle.visible = show_labels.value or (bool(ids) and set(parts[part]) <= ids)
        if ids:
            names = sorted({geom_name(model, i) for i in ids})
            highlight_info.content = f"{len(names)} geom(s): " + ", ".join(f"`{n}`" for n in names)
        else:
            highlight_info.content = ""

    @show_visual.on_update
    def _(_) -> None:
        for handle in visual_handles.values():
            handle.visible = show_visual.value

    @show_collision.on_update
    def _(_) -> None:
        for handle in collision_handles.values():
            handle.visible = show_collision.value

    @show_labels.on_update
    def _(_) -> None:
        redraw()

    @opacity.on_update
    def _(_) -> None:
        redraw()

    @segment_dd.on_update
    def _(_) -> None:
        if segment_dd.value != NONE:
            part_dd.value = NONE
        redraw()

    @part_dd.on_update
    def _(_) -> None:
        if part_dd.value != NONE:
            segment_dd.value = NONE
        redraw()

    with server.gui.add_folder("Self-contacts"):
        contacts_md = server.gui.add_markdown("")

    def update_contacts() -> None:
        rows = self_contacts(model, data)
        if not rows:
            contacts_md.content = "No contacts in this pose."
            return
        contacts_md.content = (
            f"{len(rows)} part pair(s) in contact:\n\n| part 1 | part 2 | depth (mm) |\n|---|---|---|\n"
            + "\n".join(f"| `{a}` | `{b}` | {d * 1e3:.2f} |" for a, b, d in rows)
        )

    def refresh() -> None:
        mujoco.mj_forward(model, data)
        for i, handle in visual_handles.items():
            handle.position, handle.wxyz = geom_pose(data, i)
        for i, handle in collision_handles.items():
            handle.position, handle.wxyz = geom_pose(data, i)
        for part, handle in label_handles.items():
            handle.position = part_centre(parts[part])
        for j, handle in joint_frame_handles.items():
            handle.position, handle.wxyz = joint_pose(j)
        for s, handle in ball_frame_handles.items():
            handle.position, handle.wxyz = site_pose(s)
        if ref_handles:
            ref_data.qpos[:] = data.qpos
            mujoco.mj_forward(ref_model, ref_data)
            for i, handle in ref_handles.items():
                handle.position, handle.wxyz = geom_pose(ref_data, i)
        update_contacts()

    with server.gui.add_folder("Joints", expand_by_default=False):
        sliders = []
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            adr = model.jnt_qposadr[joint_id]
            lo, hi = model.jnt_range[joint_id] if model.jnt_limited[joint_id] else (-np.pi, np.pi)
            slider = server.gui.add_slider(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                min=float(lo), max=float(hi), step=0.01, initial_value=float(data.qpos[adr]),
            )
            sliders.append((slider, adr))

            @slider.on_update
            def _(event, adr=adr) -> None:
                data.qpos[adr] = event.target.value
                refresh()

        reset = server.gui.add_button("Reset to climb start pose")

    @reset.on_click
    def _(_) -> None:
        set_start_pose(model, data)
        for slider, adr in sliders:
            slider.value = float(data.qpos[adr])  # each update calls refresh()
        refresh()

    rows = [
        f"| {i} | `{geom_name(model, i)}` | {int(model.geom_contype[i])}/{int(model.geom_conaffinity[i])}"
        f" | {int(model.geom_condim[i])} |"
        for i in collision_ids
    ]
    with server.gui.add_folder(f"Collision geoms ({len(collision_ids)})", expand_by_default=False):
        server.gui.add_markdown("| id | name | contype/conaff | condim |\n|---|---|---|---|\n" + "\n".join(rows))

    redraw()
    refresh()
    print(f"Viser running at http://localhost:{args.port} (Ctrl+C to quit)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
