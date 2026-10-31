"""Convert a reference climb of rock-climbing-workshop/scripts/plan_sim_wall.py (--save-npz) into the motion
.npz of the climbing task.

Input: an .npz with ``q`` (frames, 26) every 0.02 s: the trunk position (simulator world frame), its
orientation (w, x, y, z) and the 19 joints in rcr.types.MICROBAN_JOINTS order. It is mapped into the env
frame with the transform of tasks/climbing/wall.py (the one the env's wall is built with), replayed
kinematically on the climbing robot to record the body poses and velocities (as planning_to_npz.py), and
checked against the env's wall: the balls of the first frame have to sit in the start sockets.

The wall is the one the climb was planned on, stored in the planner's .npz as ``wall`` (plan_sim_wall.py
--save-npz) and copied into the motion, where the env reads it (tasks/climbing/wall.py). --level/--seed or
--wall only check it, or give it for an older planner .npz without it.

The holds the plan uses (``used_holds``, planner ids r<row>_c<col>, which plan_sim_wall.py --save-npz stores in
its .npz) are copied into the motion, and the env's wall makes those holds 5-box cups and the others 1-box
blocks. An older planner .npz without them: give its contact plan (plan_sim_wall.py --save) with --contact-plan.

Usage:
    cd ../rock-climbing-workshop && uv run scripts/plan_sim_wall.py --level 2 --seed 5 --save-npz build/l2s5.npz
    uv run python src/mjlab_microban/scripts/workshop_to_npz.py \
        --input-file ../rock-climbing-workshop/build/l2s5.npz --output-file src/mjlab_microban/motion/l2s5.npz
    uv run train Mjlab-Climbing-Microban --env.commands.motion.motion-file src/mjlab_microban/motion/l2s5.npz
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg

from mjlab_microban.robot.microban_climb_constants import BALL_SITE_NAMES
from mjlab_microban.tasks.climbing.microban_climbing_env_cfg import make_microban_climbing_env_cfg
from mjlab_microban.tasks.climbing.wall import (
    SIM_TO_ENV_QUAT,
    contact_plan_hold_ids,
    hold_indices,
    motion_used_hold_ids,
    resolve_wall,
    hold_positions,
    sim_to_env,
    start_positions,
)

LOG_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")

# Joint order of the columns 7.. of q (rock-climbing-workshop: rcr.types.MICROBAN_JOINTS).
FILE_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
    "head",
)  # fmt: skip
INPUT_FPS = 50.0  # rcr.kinematics.ik.DT = 0.02 s

START_TOLERANCE = 0.005  # [m] first-frame ball to its start socket


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-file", type=Path, required=True, help="plan_sim_wall.py --save-npz output (q: N x 26)")
    p.add_argument("--output-file", type=Path, required=True)
    p.add_argument("--contact-plan", type=Path,
                   help="plan_sim_wall.py --save output (the contact plan), for an input .npz without used_holds")
    p.add_argument("--level", type=int, help="wall level, as plan_sim_wall.py --level (default: the input's wall)")
    p.add_argument("--seed", type=int, help="wall seed, as plan_sim_wall.py --seed (default: the input's wall)")
    p.add_argument("--wall", type=str, help="a bundled wall (N1, N2, N3) or a wall JSON file, as plan_sim_wall.py --wall")
    p.add_argument("--input-fps", type=float, default=INPUT_FPS)
    p.add_argument("--output-fps", type=float, default=50.0, help="Frame rate of the output motion (policy rate)")
    p.add_argument("--device", type=str, default="cuda:0")
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


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA is not available, falling back to CPU.")
        device = "cpu"
    try:
        wall = resolve_wall(args.input_file, args.level, args.seed, args.wall)
    except ValueError as e:
        raise SystemExit(f"[ERROR] {e}") from e

    try:
        used_ids = contact_plan_hold_ids(args.contact_plan) if args.contact_plan else motion_used_hold_ids(args.input_file)
        hold_indices(wall, used_ids)
    except ValueError as e:
        raise SystemExit(f"[ERROR] {e}") from e
    print(f"[INFO] {len(used_ids)} holds used: {', '.join(used_ids)}")

    q = np.load(args.input_file)["q"]
    if q.ndim != 2 or q.shape[1] != 7 + len(FILE_JOINTS):
        raise SystemExit(f"{args.input_file}: expected q of shape (frames, {7 + len(FILE_JOINTS)}), got {q.shape}")
    base_pos = sim_to_env(q[:, 0:3])
    quat = quat_mul(SIM_TO_ENV_QUAT, q[:, 3:7])
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    rows = np.hstack([base_pos, quat[:, [1, 2, 3, 0]], q[:, 7:]])  # csv: pos, quat xyzw, joints

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / args.output_fps
    scene = Scene(make_microban_climbing_env_cfg(wall=wall).scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    with tempfile.NamedTemporaryFile("w", suffix=".csv") as f:
        np.savetxt(f, rows, delimiter=",")
        f.flush()
        motion = MotionLoader(motion_file=f.name, input_fps=args.input_fps, output_fps=args.output_fps, device=device)

    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(list(FILE_JOINTS), preserve_order=True)[0]
    ball_ids = robot.find_sites(BALL_SITE_NAMES, preserve_order=True)[0]

    log: dict[str, list] = {k: [] for k in LOG_KEYS}
    balls = []
    scene.reset()
    for _ in range(motion.output_frames):
        (pos, rot, lin_vel, ang_vel, dof_pos, dof_vel), _ = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = rot
        root_states[:, 7:10] = lin_vel
        root_states[:, 10:] = ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = dof_pos
        joint_vel[:, robot_joint_indexes] = dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())
        origin = scene.env_origins[0].cpu().numpy()
        balls.append(robot.data.site_pos_w[0, ball_ids].cpu().numpy() - origin)

    # The motion has to be on this wall: the first frame's balls in their start sockets.
    start = start_positions(wall)
    errors = {}
    for name, ball in zip(BALL_SITE_NAMES, balls[0], strict=True):
        limb = next((limb for limb in start if name.startswith(limb)), None)
        if limb is not None:
            errors[limb] = float(np.linalg.norm(ball - start[limb]))
    print(f"[INFO] wall {wall}: {len(hold_positions(wall))} holds; first-frame balls from their start sockets: "
          + ", ".join(f"{limb} {e * 1000:.1f} mm" for limb, e in errors.items()))  # fmt: skip
    if not errors or max(errors.values()) > START_TOLERANCE:
        raise SystemExit(f"[ERROR] the motion does not start on the start holds of wall {wall}: wrong wall?")

    data = {k: np.stack(v, axis=0) for k, v in log.items()}
    data["fps"] = np.array([args.output_fps])
    data["used_holds"] = np.array(used_ids)  # planner hold ids (r<row>_c<col>), see tasks/climbing/wall.py
    data["wall"] = np.array(wall.text())  # the wall it was planned on (WallSpec.parse)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_file, **data)
    print(f"[INFO] Saved {motion.output_frames} frames to {args.output_file}")


if __name__ == "__main__":
    main()
