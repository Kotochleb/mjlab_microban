"""Convert a planned climb (../planning/trajecoty.py --export) into the motion .npz of the climbing task.

The planner's frame is shifted by wall.HOLDS_OFFSET so that the motion matches the wall of the climbing
env. The motion is replayed kinematically on the climbing robot in the climbing scene to record the body
poses and velocities (same as csv_to_npz.py).

Usage:
    uv run python src/mjlab_microban/scripts/planning_to_npz.py \
        --input-file ../planning/trajectory.npz --output-file src/mjlab_microban/motion/climb_planning.npz
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

from mjlab_microban.tasks.climbing.microban_climbing_env_cfg import make_microban_climbing_env_cfg
from mjlab_microban.tasks.climbing.wall import HOLDS_OFFSET

LOG_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-file", type=Path, required=True, help="Planned trajectory .npz")
    p.add_argument("--output-file", type=Path, default=Path("src/mjlab_microban/motion/climb_planning.npz"))
    p.add_argument("--output-fps", type=float, default=50.0, help="Frame rate of the output motion (policy rate)")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA is not available, falling back to CPU.")
        device = "cpu"

    plan = np.load(args.input_file)
    joint_names = [str(n) for n in plan["joint_names"]]
    input_fps = 1.0 / float(plan["dt"])
    base_pos = plan["trunk_pos"] + HOLDS_OFFSET
    quat_wxyz = plan["trunk_quat"]
    rows = np.hstack([base_pos, quat_wxyz[:, [1, 2, 3, 0]], plan["q_joints"]])  # csv: pos, quat xyzw, joints

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / args.output_fps
    scene = Scene(make_microban_climbing_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    with tempfile.NamedTemporaryFile("w", suffix=".csv") as f:
        np.savetxt(f, rows, delimiter=",")
        f.flush()
        motion = MotionLoader(motion_file=f.name, input_fps=input_fps, output_fps=args.output_fps, device=device)

    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    log: dict[str, list] = {k: [] for k in LOG_KEYS}
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

    data = {k: np.stack(v, axis=0) for k, v in log.items()}
    data["fps"] = np.array([args.output_fps])
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_file, **data)
    print(f"[INFO] Saved {motion.output_frames} frames to {args.output_file}")


if __name__ == "__main__":
    main()
