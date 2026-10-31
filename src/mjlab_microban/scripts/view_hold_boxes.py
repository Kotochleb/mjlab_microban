"""Show the climbing_sim hold ("cup") mesh in viser against its box approximations (tasks/climbing/hold_boxes.py).

Everything is in the hold frame (origin at the socket = centre of a ball sitting in the cup, x along the wall,
y into the wall, z up). Shown:
  - the visual mesh (hold_visual.stl), with an opacity slider
  - the simulator's collision geometry: the hulls + skin patches, and the cup spheres
  - the centre of the mesh: its centre of mass (volume) and the centre of its bounding box, and the socket frame
  - a ball of the robot (r = 12.5 mm) sitting in the socket
  - the 3-box, 5-box and 1-box approximations, chosen from a dropdown

Usage:
    uv run python src/mjlab_microban/scripts/view_hold_boxes.py [--port 8080]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from mjlab_microban.tasks.climbing import hold_boxes
from mjlab_microban.tasks.climbing.wall import _climbing_sim

MESH_COLOR = (217, 89, 38)  # wall_hold material
HULL_COLOR = (64, 140, 242)
SPHERE_COLOR = (255, 128, 0)
BALL_COLOR = (26, 217, 217)
APPROX_COLORS = {"three_box": (60, 200, 90), "five_box": (240, 200, 40), "one_box": (180, 80, 220)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8080, help="Viser server port (default: 8080)")
    return p.parse_args()


def hold_assets() -> Path:
    wall_mod, _ = _climbing_sim()
    return Path(wall_mod.__file__).parent / "assets" / "hold"


def fmt_mm(v) -> str:
    return "(" + ", ".join(f"{1000 * x:+.1f}" for x in v) + ") mm"


def main() -> None:
    args = parse_args()
    assets = hold_assets()

    visual = trimesh.load(assets / "hold_visual.stl")
    hull_files = sorted(assets.glob("hull_*.stl")) + sorted(assets.glob("skin_*.stl"))
    spheres = json.loads((assets / "cup_spheres.json").read_text())

    center_mass = np.asarray(visual.center_mass)
    bbox_center = visual.bounds.mean(axis=0)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=0.1, height=0.1, cell_size=0.005, position=(0.0, 0.0, hold_boxes.HOLD_BOTTOM_Z))
    # The board face (y = +26 mm), as a thin translucent plate.
    server.scene.add_box(
        "/board", dimensions=(0.08, 0.001, 0.06), position=(0.0, hold_boxes.HOLD_BACK_Y + 0.0005, 0.0),
        color=(220, 190, 140), opacity=0.3,
    )

    # ---- mesh -------------------------------------------------------------------------------------------
    mesh_handle = server.scene.add_mesh_simple(
        "/mesh/visual", visual.vertices, visual.faces, color=MESH_COLOR, opacity=0.5, side="double",
    )
    hull_handles = []
    for f in hull_files:
        m = trimesh.load(f)
        hull_handles.append(
            server.scene.add_mesh_simple(f"/mesh/collision/{f.stem}", m.vertices, m.faces, color=HULL_COLOR,
                                         opacity=0.4, visible=False)
        )
    sphere_handles = [
        server.scene.add_icosphere(f"/mesh/cup_spheres/{i}", radius=sp["r"], position=tuple(sp["pos"]),
                                   color=SPHERE_COLOR, visible=False)
        for i, sp in enumerate(spheres)
    ]

    # ---- centres ----------------------------------------------------------------------------------------
    center_handles = [
        server.scene.add_frame("/centers/socket", axes_length=0.01, axes_radius=0.0004, origin_radius=0.0008),
        server.scene.add_label("/centers/socket/label", "socket (origin)", position=(0.0, 0.0, 0.002)),
        server.scene.add_icosphere("/centers/center_mass", radius=0.001, position=center_mass, color=(255, 30, 30)),
        server.scene.add_label("/centers/center_mass/label", f"mesh centre of mass {fmt_mm(center_mass)}"),
        server.scene.add_icosphere("/centers/bbox_center", radius=0.001, position=bbox_center, color=(30, 30, 255)),
        server.scene.add_label("/centers/bbox_center/label", f"mesh bbox centre {fmt_mm(bbox_center)}"),
    ]
    bbox_handle = server.scene.add_box(
        "/centers/bbox", dimensions=tuple(visual.extents), position=bbox_center, color=(30, 30, 255), wireframe=True,
    )
    ball_handle = server.scene.add_icosphere(
        "/ball", radius=hold_boxes.BALL_RADIUS, color=BALL_COLOR, opacity=0.35, visible=True,
    )

    # ---- approximations ---------------------------------------------------------------------------------
    approx_handles: dict[str, list] = {}
    for key, boxes in hold_boxes.APPROXIMATIONS.items():
        handles = []
        for b in boxes:
            handles.append(server.scene.add_box(
                f"/approx/{key}/{b.name}", dimensions=tuple(2 * b.half_size), position=tuple(b.center),
                color=APPROX_COLORS[key], opacity=0.55,
            ))
            handles.append(server.scene.add_box(
                f"/approx/{key}/{b.name}_edges", dimensions=tuple(2 * b.half_size), position=tuple(b.center),
                color=APPROX_COLORS[key], wireframe=True,
            ))
        approx_handles[key] = handles

    # ---- GUI --------------------------------------------------------------------------------------------
    def toggle(handles, visible: bool) -> None:
        for h in handles:
            h.visible = visible

    with server.gui.add_folder("Approximation"):
        approx = server.gui.add_dropdown("Show", options=("three_box", "five_box", "one_box", "all", "none"),
                                         initial_value="three_box")

    @approx.on_update
    def _(_) -> None:
        for key, handles in approx_handles.items():
            toggle(handles, approx.value in (key, "all"))

    for key in ("five_box", "one_box"):
        toggle(approx_handles[key], False)

    with server.gui.add_folder("Mesh"):
        show_mesh = server.gui.add_checkbox("Visual mesh", initial_value=True)
        mesh_opacity = server.gui.add_slider("Mesh opacity", min=0.05, max=1.0, step=0.05, initial_value=0.5)
        show_hulls = server.gui.add_checkbox("Sim collision hulls + skin", initial_value=False)
        show_spheres = server.gui.add_checkbox("Sim cup spheres", initial_value=False)
        show_centers = server.gui.add_checkbox("Centres (socket, CoM, bbox)", initial_value=True)
        show_bbox = server.gui.add_checkbox("Mesh bounding box", initial_value=True)
        show_ball = server.gui.add_checkbox("Ball in socket", initial_value=True)

    show_mesh.on_update(lambda _: setattr(mesh_handle, "visible", show_mesh.value))
    mesh_opacity.on_update(lambda _: setattr(mesh_handle, "opacity", mesh_opacity.value))
    show_hulls.on_update(lambda _: toggle(hull_handles, show_hulls.value))
    show_spheres.on_update(lambda _: toggle(sphere_handles, show_spheres.value))
    show_centers.on_update(lambda _: toggle(center_handles, show_centers.value))
    show_bbox.on_update(lambda _: setattr(bbox_handle, "visible", show_bbox.value))
    show_ball.on_update(lambda _: setattr(ball_handle, "visible", show_ball.value))

    rows = [
        f"| {key} | {b.name} | {fmt_mm(b.center)} | {fmt_mm(b.half_size)} |"
        for key, boxes in hold_boxes.APPROXIMATIONS.items()
        for b in boxes
    ]
    with server.gui.add_folder("Boxes (hold frame)", expand_by_default=False):
        server.gui.add_markdown(
            f"mesh bounds: {fmt_mm(visual.bounds[0])} .. {fmt_mm(visual.bounds[1])}\n\n"
            "| approx | box | centre | half size |\n|---|---|---|---|\n" + "\n".join(rows)
        )

    print(f"Viser running at http://localhost:{args.port} (Ctrl+C to quit)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
