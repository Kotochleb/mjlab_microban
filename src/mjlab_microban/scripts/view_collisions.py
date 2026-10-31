"""Show the Microban model in viser with a name label on every collision geom.

The model is compiled through MICROBAN_HULLS_ROBOT_CFG (the velocity and tracking envs' robot), so the
collision geoms shown are exactly those enabled in training (contype or conaffinity non-zero after
HULLS_COLLISION): the hulls and servo boxes of robot_hulls.xml, and the sphere hands and feet.
Unnamed geoms are labelled "<body>/geom<i>", i being the geom index within its body.

Usage:
    uv run python src/mjlab_microban/scripts/view_collisions.py [--port 8080]
"""

import argparse
import time

import mujoco
import numpy as np
import viser
from mjlab.entity import Entity
from mjviser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh

from mjlab_microban.robot.microban_constants import HOME_FRAME, MICROBAN_HULLS_ROBOT_CFG

COLLISION_RGBA = np.array([230, 120, 40, 140], dtype=np.uint8)
FOOT_RGBA = np.array([40, 140, 230, 160], dtype=np.uint8)
VISUAL_GROUP = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8080, help="Viser server port (default: 8080)")
    return p.parse_args()


def geom_label(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name:
        return name
    body_id = model.geom_bodyid[geom_id]
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"
    local = geom_id - model.body_geomadr[body_id]
    return f"{body}/geom{local}"


def set_home_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = HOME_FRAME.pos
    for joint_name, value in HOME_FRAME.joint_pos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def geom_pose(data: mujoco.MjData, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(wxyz, data.geom_xmat[geom_id])
    return data.geom_xpos[geom_id].copy(), wxyz


def main() -> None:
    args = parse_args()

    model = Entity(MICROBAN_HULLS_ROBOT_CFG).spec.compile()
    data = mujoco.MjData(model)
    set_home_pose(model, data)

    collision_ids = [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    visual_ids = [i for i in range(model.ngeom) if model.geom_group[i] == VISUAL_GROUP]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=1.0, height=1.0, cell_size=0.05)

    # (handle, geom_id) pairs whose pose follows the simulation state.
    posed: list[tuple[viser.SceneNodeHandle, int]] = []

    visual_handles = []
    for i in visual_ids:
        pos, wxyz = geom_pose(data, i)
        handle = server.scene.add_mesh_trimesh(f"/visual/{i}", mujoco_mesh_to_trimesh(model, i), position=pos, wxyz=wxyz)
        visual_handles.append(handle)
        posed.append((handle, i))

    collision_handles = []
    label_handles = []
    rows = []
    for i in collision_ids:
        label = geom_label(model, i)
        is_foot = "foot_collision" in label
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = mujoco_mesh_to_trimesh(model, i)
        else:
            mesh = create_primitive_mesh(model, i)
        mesh.visual.vertex_colors = np.tile(FOOT_RGBA if is_foot else COLLISION_RGBA, (len(mesh.vertices), 1))
        pos, wxyz = geom_pose(data, i)
        handle = server.scene.add_mesh_trimesh(f"/collision/{i}", mesh, position=pos, wxyz=wxyz)
        label_handle = server.scene.add_label(f"/labels/{i}", label, position=pos)
        collision_handles.append(handle)
        label_handles.append(label_handle)
        posed.append((handle, i))
        posed.append((label_handle, i))
        rows.append(
            f"| {i} | `{label}` | {int(model.geom_contype[i])}/{int(model.geom_conaffinity[i])} | {int(model.geom_condim[i])} |"
        )

    print(f"{len(collision_ids)} collision geoms:")
    for i in collision_ids:
        print(f"  [{i:3d}] {geom_label(model, i)}")

    def refresh() -> None:
        mujoco.mj_forward(model, data)
        for handle, i in posed:
            pos, wxyz = geom_pose(data, i)
            handle.position = pos
            if not isinstance(handle, viser.LabelHandle):
                handle.wxyz = wxyz

    with server.gui.add_folder("Display"):
        show_visual = server.gui.add_checkbox("Visual meshes", initial_value=True)
        show_collision = server.gui.add_checkbox("Collision geoms", initial_value=True)
        show_labels = server.gui.add_checkbox("Labels", initial_value=True)

    @show_visual.on_update
    def _(_) -> None:
        for handle in visual_handles:
            handle.visible = show_visual.value

    @show_collision.on_update
    def _(_) -> None:
        for handle in collision_handles:
            handle.visible = show_collision.value

    @show_labels.on_update
    def _(_) -> None:
        for handle in label_handles:
            handle.visible = show_labels.value

    with server.gui.add_folder("Joints", expand_by_default=False):
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            adr = model.jnt_qposadr[joint_id]
            lo, hi = model.jnt_range[joint_id] if model.jnt_limited[joint_id] else (-np.pi, np.pi)
            slider = server.gui.add_slider(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                min=float(lo), max=float(hi), step=0.01, initial_value=float(data.qpos[adr]),
            )

            @slider.on_update
            def _(event, adr=adr) -> None:
                data.qpos[adr] = event.target.value
                refresh()

    with server.gui.add_folder(f"Collision geoms ({len(collision_ids)})", expand_by_default=False):
        server.gui.add_markdown("| id | name | contype/conaff | condim |\n|---|---|---|---|\n" + "\n".join(rows))

    print(f"Viser running at http://localhost:{args.port} (Ctrl+C to quit)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
