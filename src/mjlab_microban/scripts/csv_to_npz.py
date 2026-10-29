# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Convert a Microban motion CSV into the .npz format used by the tracking task.

Each CSV row is one frame: base position (x, y, z), base quaternion (qx, qy, qz, qw),
followed by the joint positions in the order of JOINT_NAMES below.

Usage:
    uv run python src/mjlab_microban/scripts/csv_to_npz.py --input-file motion.csv --output-file motion.npz

    # Also upload to the wandb "motions" registry (usable with --registry-name):
    uv run python src/mjlab_microban/scripts/csv_to_npz.py --input-file motion.csv --output-file motion.npz --wandb-name microban_dance
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg

from mjlab_microban.tasks.tracking.microban_tracking_env_cfg import make_microban_tracking_env_cfg

# Model joint order, as written by microban_retarget.
JOINT_NAMES = [
    "head",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
]

LOG_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-file", type=Path, required=True, help="Path to the input CSV file")
    p.add_argument("--output-file", type=Path, default=Path("motion.npz"), help="Path to the output .npz file")
    p.add_argument("--input-fps", type=float, default=30.0, help="Frame rate of the CSV file")
    p.add_argument("--output-fps", type=float, default=50.0, help="Frame rate of the output motion (policy rate)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--wandb-name", type=str, default=None, help="If set, upload the motion to the wandb registry under this name")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA is not available, falling back to CPU.")
        device = "cpu"

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / args.output_fps

    scene = Scene(make_microban_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    motion = MotionLoader(
        motion_file=str(args.input_file),
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        device=device,
    )

    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]

    log: dict[str, list] = {k: [] for k in LOG_KEYS}
    scene.reset()

    for _ in range(motion.output_frames):
        (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel), _ = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = base_rot
        root_states[:, 7:10] = base_lin_vel
        root_states[:, 10:] = base_ang_vel
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

    if args.wandb_name is not None:
        import wandb

        registry = "motions"
        run = wandb.init(project="csv_to_npz", name=args.wandb_name)
        # mjlab's train/play scripts expect the file to be named motion.npz in the artifact.
        artifact = wandb.Artifact(name=args.wandb_name, type=registry)
        artifact.add_file(str(args.output_file), name="motion.npz")
        artifact = run.log_artifact(artifact)
        run.link_artifact(artifact=artifact, target_path=f"wandb-registry-{registry}/{args.wandb_name}")
        print(f"[INFO] Motion uploaded to wandb registry: {registry}/{args.wandb_name}")
        wandb.finish()


if __name__ == "__main__":
    main()
