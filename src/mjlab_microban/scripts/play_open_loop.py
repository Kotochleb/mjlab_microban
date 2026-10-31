"""Play a motion open loop in the climbing env, in viser, with a slider over its frames.

The motion's joint positions are sent as PD targets to the robot of the climbing env (same scene, physics,
decimation and BAM actuators as Mjlab-Climbing-Microban), with no feedback: the robot starts on the motion's
first frame and is simulated from there. A ghost shows the reference frame being played.

Motion file: an .npz with ``q`` (frames, 26), as exported by the rock-climbing-workshop optimiser:
    q[:, 0:3]  trunk position in the workshop's wall frame (origin at the bottom-left corner of the wall face,
               x right, y into the wall, z up)
    q[:, 3:7]  trunk orientation, quaternion (w, x, y, z)
    q[:, 7:]   the 19 joint angles, in FILE_JOINTS order
    used_holds the planner ids of the holds the plan uses (r<row>_c<col>), as plan_sim_wall.py --save-npz stores
               them: those holds collide as 5-box cups, the others as 1-box blocks
It is mapped into the env frame (x towards the wall, y to the robot's left, z up, wall centred on y = 0) by the
transform the env's wall is built with (tasks/climbing/wall.py: a -90 deg yaw, then the hold sockets on x = 0
and the middle of the 0.8 m wide wall on y = 0), plus the optional --offset.

The wall is the climbing_sim route the motion was planned on: its ``wall`` (plan_sim_wall.py --save-npz stores
it). --level / --seed (generate_wall) or --wall (a bundled wall N1-N3 or a JSON file), as to the workshop's
scripts/plan_sim_wall.py, only check it, or give it for an older motion without it. The distance of the first frame's balls to their start sockets is printed: a few mm
at most when the wall is the right one.

GUI:
- Play / Pause: run the simulation open loop from the current frame, at --fps times the speed factor.
- Frame slider: pauses and shows that frame of the motion (the robot is put kinematically on it). Pressing
  Play then runs open loop from there.
- Reset to frame: put the robot back on the frame of the slider.
- Collisions: show the collision geoms (the holds' boxes, group 3: 5-box cups on the holds of the motion's
  used_holds, 1-box blocks on the others, see tasks/climbing/wall.py), the contact points and
  forces, and list the contacts of the simulated robot or of the reference (the ghost): ball in a hold is
  expected, anything else (a ball on the board, another part of the robot on the wall or the floor, the robot
  on itself) is flagged. "Scan reference" checks every frame of the motion, "Next issue" jumps to the next
  flagged frame.

Usage:
    uv run python src/mjlab_microban/scripts/play_open_loop.py --motion-file src/mjlab_microban/motion/l1s0_q.npz
"""

import argparse
import copy
import re
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import viser
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation
from mjlab.viewer.viser.scene import MjlabViserScene

from mjlab_microban.tasks.climbing.mdp import set_bam_kp
from mjlab_microban.robot.microban_climb_constants import BALL_GEOM_NAMES, BALL_SITE_NAMES
from mjlab_microban.tasks.climbing.microban_climbing_env_cfg import KP, make_microban_climbing_env_cfg
from mjlab_microban.tasks.climbing.wall import SIM_TO_ENV_QUAT, hold_positions, resolve_wall, sim_to_env, start_positions

DEFAULT_MOTION = Path(__file__).parents[1] / "motion" / "l1s0_q.npz"

# Joint order of the motion file's columns 7.. (rock-climbing-workshop: rcr.types.MICROBAN_JOINTS).
FILE_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
    "head",
)  # fmt: skip

# Extra env-frame translation of the motion, on top of the wall's frame change (wall.sim_to_env).
DEFAULT_OFFSET = (0.0, 0.0, 0.0)

GHOST_RGBA = (0.3, 0.7, 1.0, 0.4)

COLLISION_GEOM_GROUP = 3  # climbing_sim's hold collision geoms (wall.py), and the robot's
MAX_LISTED_CONTACTS = 20
HOLD_GEOM_RE = re.compile(r"hold(\d+)_c(\d+)_r(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION, help="Motion .npz with q (frames, 26)")
    p.add_argument("--level", type=int, choices=(1, 2, 3, 4), help="Wall level (default: the motion's wall)")
    p.add_argument("--seed", type=int, help="Wall seed (default: the motion's wall)")
    p.add_argument("--wall", type=str, default=None,
                   help="A bundled wall (N1, N2, N3) or a wall JSON file, instead of --level/--seed")
    p.add_argument("--fps", type=float, default=50.0, help="Frame rate of the motion (default: 50 Hz)")
    p.add_argument("--offset", type=float, nargs=3, default=DEFAULT_OFFSET, metavar=("X", "Y", "Z"),
                   help="Extra env-frame translation of the motion (default: %(default)s)")
    p.add_argument("--kp", type=float, default=KP, help="BAM firmware P gain (default: %(default)s)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--port", type=int, default=8080, help="Viser server port (default: 8080)")
    return p.parse_args()  # fmt: skip


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions, b batched (..., 4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


class Motion:
    """The reference, in the env frame and the robot's joint order."""

    def __init__(self, path: Path, joint_names: list[str], offset: np.ndarray, fps: float):
        q = np.load(path)["q"]
        if q.ndim != 2 or q.shape[1] != 7 + len(FILE_JOINTS):
            raise SystemExit(f"{path}: expected q of shape (frames, {7 + len(FILE_JOINTS)}), got {q.shape}")
        self.num_frames = q.shape[0]
        self.dt = 1.0 / fps
        self.root_pos = sim_to_env(q[:, 0:3]) + offset
        quat = quat_mul(SIM_TO_ENV_QUAT, q[:, 3:7])
        self.root_quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
        self.joint_pos = q[:, 7:][:, [FILE_JOINTS.index(j) for j in joint_names]]
        self.joint_vel = np.gradient(self.joint_pos, self.dt, axis=0)
        self.root_lin_vel = np.gradient(self.root_pos, self.dt, axis=0)


class OpenLoopPlayer:
    def __init__(self, args: argparse.Namespace):
        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARNING] CUDA is not available, falling back to CPU.")
            device = "cpu"
        self.device = device

        self.wall = resolve_wall(args.motion_file, args.level, args.seed, args.wall)
        print(f"[INFO] wall {self.wall}: {len(hold_positions(self.wall))} holds")
        # The holds of the motion's used_holds collide as 5-box cups, the others as 1-box blocks.
        cfg = make_microban_climbing_env_cfg(play=True, wall=self.wall, motion_file=str(args.motion_file))
        cfg.scene.num_envs = 1
        self.decimation = cfg.decimation
        self.physics_dt = cfg.sim.mujoco.timestep
        self.scene = Scene(cfg.scene, device=device)
        self.sim = Simulation(num_envs=1, cfg=cfg.sim, model=self.scene.compile(), device=device)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        if self.scene.sensor_context is not None:
            self.sim.set_sensor_context(self.scene.sensor_context)

        self.robot: Entity = self.scene["robot"]
        set_bam_kp(self.robot, args.kp)
        self.motion = Motion(args.motion_file, list(self.robot.joint_names), np.array(args.offset), args.fps)
        print(f"[INFO] {self.motion.num_frames} frames ({self.motion.num_frames * self.motion.dt:.1f} s) "
              f"from {args.motion_file}, joints: {list(self.robot.joint_names)}")  # fmt: skip

        idx = self.robot.indexing
        self._free_q_adr = idx.free_joint_q_adr.cpu().numpy()
        self._joint_q_adr = idx.joint_q_adr.cpu().numpy()
        self._ghost_model = self._make_ghost_model()
        self._init_collision_check()

        self.frame = 0
        self.playing = False
        self.lock = threading.Lock()
        self._setup_viser(args.port)
        self.reset_to_frame(0)
        self._check_start()

    def _check_start(self) -> None:
        """Print how far the first frame's balls are from their start sockets (is it the motion's wall?)."""
        ids = self.robot.find_sites(BALL_SITE_NAMES, preserve_order=True)[0]
        balls = (self.robot.data.site_pos_w[0, ids] - self.scene.env_origins[0]).cpu().numpy()
        start = start_positions(self.wall)
        errors = [
            f"{limb} {np.linalg.norm(ball - start[limb]) * 1000:.1f} mm"
            for name, ball in zip(BALL_SITE_NAMES, balls, strict=True)
            for limb in start
            if name.startswith(limb)
        ]
        print("[INFO] first frame, balls from their start sockets: " + ", ".join(errors))

    # ------------------------------------------------------------------ sim

    def _t(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)[None]

    def reset_to_frame(self, k: int) -> None:
        """Put the robot kinematically on frame k of the motion (with its velocities) and target it."""
        m = self.motion
        self.frame = k
        self.sim.reset()
        self.scene.reset()  # actuator delays, wall placement
        root = torch.zeros(1, 13, device=self.device)
        root[:, 0:3] = self._t(m.root_pos[k]) + self.scene.env_origins
        root[:, 3:7] = self._t(m.root_quat[k])
        root[:, 7:10] = self._t(m.root_lin_vel[k])
        self.robot.write_root_state_to_sim(root)
        self.robot.write_joint_state_to_sim(self._t(m.joint_pos[k]), self._t(m.joint_vel[k]))
        self.robot.set_joint_position_target(self._t(m.joint_pos[k]))
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self) -> None:
        """One policy step (decimation physics steps) targeting the next frame of the motion."""
        if self.frame >= self.motion.num_frames - 1:
            self.playing = False
            return
        self.frame += 1
        self.robot.set_joint_position_target(self._t(self.motion.joint_pos[self.frame]))
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.physics_dt)
        self.sim.forward()

    # ----------------------------------------------------------- collisions

    def _init_collision_check(self) -> None:
        """A CPU copy of the model to evaluate contacts of the simulated or the reference state."""
        m = self.sim.mj_model
        self._check_data = mujoco.MjData(m)
        trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk")
        self._robot_root = int(m.body_rootid[trunk])
        wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wall/wall")
        self._wall_root = int(m.body_rootid[wall])
        self._ball_geoms = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{n}") for n in BALL_GEOM_NAMES
        } - {-1}
        self._issue_frames: np.ndarray = np.zeros(0, dtype=int)

    def _geom_kind(self, g: int) -> str:
        """'ball', 'robot', 'floor' or 'wall'."""
        m = self.sim.mj_model
        if g in self._ball_geoms:
            return "ball"
        body = m.geom_bodyid[g]
        if m.body_rootid[body] == self._robot_root:
            return "robot"
        return "wall" if m.body_rootid[body] == self._wall_root else "floor"

    def _geom_label(self, g: int) -> str:
        m = self.sim.mj_model
        name = m.geom(g).name or m.body(m.geom_bodyid[g]).name
        name = name.removeprefix("robot/").removeprefix("wall/")
        hold = HOLD_GEOM_RE.search(name)
        if hold:
            part = name[hold.end() :].lstrip("_") or "?"
            return f"hold{hold[1]} (c{hold[2]} r{hold[3]}) {part}"
        return name

    def _contacts(self, which: str, with_forces: bool, frame: int | None = None) -> list[dict]:
        """Contacts of the simulated robot ("Simulation") or of the reference ("Reference") on ``frame``
        (default: the current one)."""
        m, d = self.sim.mj_model, self._check_data
        if which == "Reference":
            d.qpos[:] = self._ghost_qpos(self.frame if frame is None else frame)
            d.qvel[:] = 0.0
        else:
            d.qpos[:] = self.sim.data.qpos[0].cpu().numpy()
            d.qvel[:] = self.sim.data.qvel[0].cpu().numpy()
        if m.nmocap:
            d.mocap_pos[:] = self.sim.data.mocap_pos[0].cpu().numpy()
            d.mocap_quat[:] = self.sim.data.mocap_quat[0].cpu().numpy()
        if with_forces:
            mujoco.mj_forward(m, d)
        else:  # positions only: no constraint solve
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
        out, force = [], np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            k1, k2 = self._geom_kind(g1), self._geom_kind(g2)
            if "ball" not in (k1, k2) and "robot" not in (k1, k2):
                continue  # wall on floor
            if k2 in ("ball", "robot") and k1 not in ("ball", "robot"):
                g1, g2, k1, k2 = g2, g1, k2, k1  # the robot's geom first
            other = self._geom_label(g2)
            expected = k1 == "ball" and k2 == "wall" and other.startswith("hold")
            if k1 in ("ball", "robot") and k2 in ("ball", "robot"):
                kind = "self collision"
            elif k2 == "floor":
                kind = "on the floor"
            elif expected:
                kind = "ball in hold"
            elif k1 == "ball":
                kind = "ball on " + ("the board" if other == "board" else "a hold's outside")
            else:
                kind = "body on the wall"
            f = None
            if with_forces:
                mujoco.mj_contactForce(m, d, i, force)
                f = float(force[0])
            out.append(dict(a=self._geom_label(g1), b=other, dist=float(c.dist), force=f, kind=kind, ok=expected))
        return out

    def _contacts_markdown(self, contacts: list[dict], which: str) -> str:
        issues = [c for c in contacts if not c["ok"]]
        head = f"**{which}, frame {self.frame}**: {len(contacts)} contacts, " + (
            f"**{len(issues)} unexpected**" if issues else "none unexpected"
        )
        if not contacts:
            return head
        rows = sorted(contacts, key=lambda c: (c["ok"], c["dist"]))[:MAX_LISTED_CONTACTS]
        lines = [head, "", "| | robot | other | depth mm | N |", "|-|-|-|-|-|"]
        for c in rows:
            f = "" if c["force"] is None else f"{c['force']:.2f}"
            mark = "ok" if c["ok"] else "⚠ " + c["kind"]
            lines.append(f"| {mark} | {c['a']} | {c['b']} | {-c['dist'] * 1000:.1f} | {f} |")
        if len(contacts) > len(rows):
            lines.append(f"\n… {len(contacts) - len(rows)} more")
        return "\n".join(lines)

    def scan_reference(self) -> str:
        """Unexpected contacts of the reference on every frame: the frames, grouped into intervals."""
        issues: dict[int, list[dict]] = {}
        for k in range(self.motion.num_frames):
            bad = [c for c in self._contacts("Reference", with_forces=False, frame=k) if not c["ok"]]
            if bad:
                issues[k] = bad
        self._issue_frames = np.array(sorted(issues), dtype=int)
        if not issues:
            return f"Reference: no unexpected contact in {self.motion.num_frames} frames."
        intervals, start = [], None
        frames = list(self._issue_frames)
        for i, k in enumerate(frames):
            start = k if start is None else start
            if i + 1 == len(frames) or frames[i + 1] != k + 1:
                intervals.append((start, k))
                start = None
        lines = [f"Reference: **{len(frames)} frames** with unexpected contacts, in {len(intervals)} intervals:", ""]
        for a, b in intervals[:MAX_LISTED_CONTACTS]:
            worst = min((c for k in range(a, b + 1) for c in issues[k]), key=lambda c: c["dist"])
            lines.append(
                f"- frames {a}-{b} ({a * self.motion.dt:.2f}-{b * self.motion.dt:.2f} s): {worst['kind']}, "
                f"{worst['a']} / {worst['b']}, up to {-worst['dist'] * 1000:.1f} mm"
            )
        if len(intervals) > MAX_LISTED_CONTACTS:
            lines.append(f"- … {len(intervals) - MAX_LISTED_CONTACTS} more")
        return "\n".join(lines)

    # ---------------------------------------------------------------- viser

    def _make_ghost_model(self) -> mujoco.MjModel:
        """The model with only the robot's visual geoms, coloured, as the reference ghost."""
        model = copy.deepcopy(self.sim.mj_model)
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk")
        root = model.body_rootid[trunk] if trunk >= 0 else -1
        for g in range(model.ngeom):
            is_robot = root >= 0 and model.body_rootid[model.geom_bodyid[g]] == root
            if is_robot and model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                model.geom_rgba[g] = GHOST_RGBA
            else:
                model.geom_rgba[g, 3] = 0.0
        return model

    def _ghost_qpos(self, k: int) -> np.ndarray:
        m = self.motion
        qpos = self.sim.mj_model.qpos0.copy()
        qpos[self._free_q_adr[0:3]] = m.root_pos[k] + self.scene.env_origins[0].cpu().numpy()
        qpos[self._free_q_adr[3:7]] = m.root_quat[k]
        qpos[self._joint_q_adr] = m.joint_pos[k]
        return qpos

    def _setup_viser(self, port: int) -> None:
        self.server = viser.ViserServer(port=port, label="open loop")
        self.vscene = MjlabViserScene(self.server, self.sim.mj_model, num_envs=1)
        self.vscene.debug_visualization_enabled = True
        self.vscene.camera_tracking_enabled = False  # keep the wall still

        gui = self.server.gui
        with gui.add_folder("Motion"):
            self.play_button = gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
            self.frame_slider = gui.add_slider(
                "Frame", min=0, max=self.motion.num_frames - 1, step=1, initial_value=0
            )
            self.time_text = gui.add_text("Time", initial_value="", disabled=True)
            self.speed_slider = gui.add_slider("Speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
            reset_button = gui.add_button("Reset to frame", icon=viser.Icon.REFRESH)
            self.ghost_checkbox = gui.add_checkbox("Show reference ghost", initial_value=True)
        with gui.add_folder("Collisions"):
            collision_geoms = gui.add_checkbox("Show collision geoms", initial_value=False)
            contact_points = gui.add_checkbox("Show contact points", initial_value=False)
            contact_forces = gui.add_checkbox("Show contact forces", initial_value=False)
            self.contacts_checkbox = gui.add_checkbox("List contacts", initial_value=True)
            self.contacts_source = gui.add_dropdown("Of", ("Simulation", "Reference"), initial_value="Simulation")
            self.contacts_md = gui.add_markdown("")
            scan_button = gui.add_button("Scan reference", icon=viser.Icon.SEARCH)
            next_issue_button = gui.add_button("Next issue", icon=viser.Icon.ARROW_RIGHT)
            self.scan_md = gui.add_markdown("")
        self.vscene.create_visualization_gui(camera_distance=1.8, camera_azimuth=180.0, camera_elevation=15.0)

        @self.play_button.on_click
        def _(_) -> None:
            with self.lock:
                if not self.playing and self.frame >= self.motion.num_frames - 1:
                    self.reset_to_frame(0)
                self.playing = not self.playing
            self._refresh_play_button()

        @self.frame_slider.on_update
        def _(event: viser.GuiEvent) -> None:
            if event.client is None:  # set by render(), not by the user
                return
            with self.lock:
                self.playing = False
                self.reset_to_frame(int(self.frame_slider.value))
            self._refresh_play_button()
            self.render()

        @reset_button.on_click
        def _(_) -> None:
            with self.lock:
                self.reset_to_frame(int(self.frame_slider.value))
            self.render()

        @self.ghost_checkbox.on_update
        def _(_) -> None:
            self.render()

        @collision_geoms.on_update
        def _(_) -> None:
            def mutate() -> None:
                self.vscene.geom_groups_visible[COLLISION_GEOM_GROUP] = collision_geoms.value
                self.vscene._sync_visibilities()

            self.vscene._apply_visualization_change(mutate)
            self.render()

        @contact_points.on_update
        def _(_) -> None:
            self.vscene.show_contact_points = contact_points.value
            self.render()

        @contact_forces.on_update
        def _(_) -> None:
            self.vscene.show_contact_forces = contact_forces.value
            self.render()

        @self.contacts_checkbox.on_update
        def _(_) -> None:
            self.render()

        @self.contacts_source.on_update
        def _(_) -> None:
            self.render()

        @scan_button.on_click
        def _(_) -> None:
            self.scan_md.content = "Scanning…"
            with self.lock:
                self.scan_md.content = self.scan_reference()

        @next_issue_button.on_click
        def _(_) -> None:
            with self.lock:
                if not len(self._issue_frames):
                    self.scan_md.content = self.scan_reference()
                later = self._issue_frames[self._issue_frames > self.frame]
                if not len(later):
                    return
                self.playing = False
                self.reset_to_frame(int(later[0]))
                self.contacts_source.value = "Reference"
            self._refresh_play_button()
            self.render()

    def _refresh_play_button(self) -> None:
        self.play_button.label = "Pause" if self.playing else "Play"
        self.play_button.icon = viser.Icon.PLAYER_PAUSE if self.playing else viser.Icon.PLAYER_PLAY

    def render(self) -> None:
        with self.lock:
            k = self.frame
            self.vscene.clear()
            if self.ghost_checkbox.value:
                self.vscene.add_ghost_mesh(self._ghost_qpos(k), model=self._ghost_model, label="reference")
            self.vscene.update(self.sim.data)
            if self.contacts_checkbox.value:
                which = self.contacts_source.value
                contacts = self._contacts(which, with_forces=which == "Simulation")
                self.contacts_md.content = self._contacts_markdown(contacts, which)
        self.frame_slider.value = k
        self.time_text.value = f"{k * self.motion.dt:.2f} / {(self.motion.num_frames - 1) * self.motion.dt:.2f} s"

    def run(self) -> None:
        print(f"[INFO] Viser running at http://localhost:{self.server.get_port()} (Ctrl+C to quit)")
        self.render()
        next_time = time.perf_counter()
        while True:
            if not self.playing:
                time.sleep(0.02)
                next_time = time.perf_counter()
                continue
            with self.lock:
                if self.playing:
                    self.step()
            if not self.playing:
                self._refresh_play_button()
            self.render()
            next_time += self.motion.dt / self.speed_slider.value
            delay = next_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_time = time.perf_counter()  # slower than real time: don't try to catch up


def main() -> None:
    OpenLoopPlayer(parse_args()).run()


if __name__ == "__main__":
    main()
