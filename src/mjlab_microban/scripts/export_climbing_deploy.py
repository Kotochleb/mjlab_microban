# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Export a trained climbing policy for the real robot: the deployment bundle of rock-climbing-workshop's
rcr.controller.mjlab_policy (run at 50 Hz on the robot by its Microban move, src/rcr/robot/climb_policy_move.py),
checked against the env. That controller is imported from $RCR_SRC (default ../rock-climbing-workshop/src).

The env is the task's play env (as ``uv run play``), on the CPU, with the checkpoint and the motion; from it come:
  - <out>/policy.onnx         the actor (obs normaliser + network), obs -> actions;
  - <out>/climb_bundle.npz    the reference, observation layout, action scale, adaptive phase parameters, kinematics;
  - <out>/wall.json           the wall, for the organisers' simulator (make sim WALL=<out>/wall.json);
  - <out>/parity.npz          the check's trace (env vs move, every tick).

Check (--check-steps, default 500 ticks = 10 s): the play env runs the policy (torch); the move's controller is fed
the robot-like measurements of every tick (encoders, IMU), and has to give the same observation, action and joint
targets as the env (the env's v_ref given, then its own estimated v_ref is compared to the env's).

Usage (train_climbing.sh --deploy does this, with the run / motion resolution of --play):
    uv run python src/mjlab_microban/scripts/export_climbing_deploy.py --task Mjlab-Climbing-Microban-Adaptive-Phase \
        --wandb-run-path entity/project/run_id --motion-file src/mjlab_microban/motion/l1s0.npz --out deploy/l1s0
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

SUPPORTED_TERMS = ("base_ang_vel", "projected_gravity", "base_yaw", "joint_pos", "joint_vel", "actions", "command",
                   "phase")
# The robot's code: rock-climbing-workshop's rcr.controller.mjlab_policy ($RCR_SRC: its src/ folder).
RCR_SRC = Path(os.environ.get("RCR_SRC", Path(__file__).resolve().parents[4] / "rock-climbing-workshop" / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint-file", type=Path)
    src.add_argument("--wandb-run-path", type=str)
    p.add_argument("--wandb-checkpoint-name", type=str, default=None)
    # The env reads its wall from this flag on the command line (tasks/climbing/wall.py, motion_file_from_argv).
    p.add_argument("--motion-file", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="bundle folder (default: logs/deploy/<run>_<checkpoint>)")
    p.add_argument("--contact-threshold", type=float, default=0.005,
                   help="m: a ball is on a hold (for the trunk position estimate) where the plan has it this close "
                        "to a hold socket (default 0.005)")
    p.add_argument("--check-steps", type=int, default=500, help="ticks of the parity check (0: none)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--offline", action="store_true",
                   help="no W&B: the checkpoint downloaded before (logs/rsl_rl/<experiment>/wandb_checkpoints/<run>)")
    p.add_argument("--force", action="store_true",
                   help="rebuild the bundle even if one of this checkpoint and motion passed its check already")
    return p.parse_args()


def resolve_checkpoint(args, task: str) -> Path:
    if args.checkpoint_file is not None:
        if not args.checkpoint_file.is_file():
            sys.exit(f"checkpoint not found: {args.checkpoint_file}")
        return args.checkpoint_file.resolve()
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab.utils.os import get_wandb_checkpoint_path

    log_root = (Path("logs/rsl_rl") / load_rl_cfg(task).experiment_name).resolve()
    if not args.offline:
        os.environ.setdefault("WANDB_HTTP_TIMEOUT", "20")  # no internet: fail fast, then the cache
        try:
            path, cached = get_wandb_checkpoint_path(log_root, Path(args.wandb_run_path), args.wandb_checkpoint_name)
            print(f"[INFO] checkpoint {path.name} of W&B run {args.wandb_run_path} "
                  f"({'cached' if cached else 'downloaded'})")
            return path
        except Exception as e:  # noqa: BLE001  (no network, no API key, W&B down: the cache)
            print(f"[WARN] W&B unreachable ({type(e).__name__}: {str(e)[:120]}): the cached checkpoint")
    path = cached_checkpoint(log_root, args.wandb_run_path, args.wandb_checkpoint_name)
    if path is None:
        sys.exit(f"no cached checkpoint of W&B run {args.wandb_run_path} in {log_root / 'wandb_checkpoints'}: "
                 "run once with internet (it is downloaded and kept there)")
    newest = "" if args.wandb_checkpoint_name else ": the newest one downloaded, maybe not the run's latest"
    print(f"[INFO] checkpoint {path.name} of W&B run {args.wandb_run_path} (cached, offline{newest})")
    return path


def cached_checkpoint(log_root: Path, run_path: str, name: str | None) -> Path | None:
    """A checkpoint of a W&B run downloaded before (mjlab keeps them in wandb_checkpoints/<run id>)."""
    folder = log_root / "wandb_checkpoints" / str(run_path).rstrip("/").split("/")[-1]
    if name:
        return folder / name if (folder / name).is_file() else None

    def iteration(p: Path) -> int:
        it = p.stem.removeprefix("model_")
        return int(it) if it.isdigit() else -1

    return max(folder.glob("model_*.pt"), key=iteration, default=None)


def cached_bundle(out: Path, task: str, checkpoint: Path, motion_file: Path, fmt: int) -> bool:
    """A bundle of this checkpoint and motion is in out, of this format, and passed its check: kept."""
    try:
        info = json.loads((out / "info.json").read_text())
    except (OSError, ValueError):
        return False
    files = all((out / f).is_file() for f in ("climb_bundle.npz", "policy.onnx", "wall.json"))
    same = (info.get("task") == task and info.get("checkpoint") == str(checkpoint)
            and info.get("motion_file") == str(motion_file) and info.get("format") == fmt)
    if files and same and info.get("parity") == "passed":
        print(f"[INFO] cached bundle: same checkpoint, motion and format, its check passed "
              f"({info.get('parity_ticks')} ticks); --force rebuilds it")
        return True
    return False


def actor_layout(env) -> list[tuple[str, int, int]]:
    """(term, per-step dim, history length) of the actor's observation, in order; unsupported terms: an error."""
    om = env.observation_manager
    names = om.active_terms["actor"]
    dims = om.group_obs_term_dim["actor"]
    cfgs = om._group_obs_term_cfgs["actor"]
    layout = []
    for name, dim, cfg in zip(names, dims, cfgs, strict=True):
        if name not in SUPPORTED_TERMS:
            sys.exit(f"actor observation term {name!r} is not implemented on the robot (rcr.controller.mjlab_policy "
                     f"implements {SUPPORTED_TERMS}): the conv1d variants are not supported yet")
        if cfg.clip is not None or cfg.scale is not None or getattr(cfg, "delay_max_lag", 0) > 0:
            sys.exit(f"actor term {name!r} has clip / scale / delay: not implemented on the robot")
        hist = int(cfg.history_length or 0)
        total = int(np.prod(dim))
        layout.append((name, total // max(hist, 1), hist))
    return layout


def kinematics(env, joint_names: list[str], ball_sites: tuple[str, ...]) -> dict:
    """The robot's bodies under the trunk, parents first, in the trunk frame (MuJoCo's model of the env)."""
    import mujoco

    m = env.sim.mj_model
    trunk = m.body("robot/trunk").id
    bodies = [trunk]
    for b in range(trunk + 1, m.nbody):
        if int(m.body_parentid[b]) in bodies:
            bodies.append(b)
    index = {b: i for i, b in enumerate(bodies)}
    out = {k: [] for k in ("kin_parent", "kin_body_pos", "kin_body_quat", "kin_joint", "kin_joint_axis",
                           "kin_joint_pos", "kin_joint_qpos0")}
    for b in bodies:
        out["kin_parent"].append(-1 if b == trunk else index[int(m.body_parentid[b])])
        out["kin_body_pos"].append(np.zeros(3) if b == trunk else m.body_pos[b].copy())
        out["kin_body_quat"].append(np.array([1.0, 0, 0, 0]) if b == trunk else m.body_quat[b].copy())
        joints = [j for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])] if m.body_jntnum[b] else []
        hinges = [j for j in joints if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        if b != trunk and len(joints) != len(hinges) or len(hinges) > 1:
            sys.exit(f"body {m.body(b).name}: only one hinge per body is supported")
        if b != trunk and hinges:
            j = hinges[0]
            out["kin_joint"].append(joint_names.index(m.joint(j).name.split("/", 1)[-1]))
            out["kin_joint_axis"].append(m.jnt_axis[j].copy())
            out["kin_joint_pos"].append(m.jnt_pos[j].copy())
            out["kin_joint_qpos0"].append(float(m.qpos0[m.jnt_qposadr[j]]))
        else:
            out["kin_joint"].append(-1)
            out["kin_joint_axis"].append(np.zeros(3))
            out["kin_joint_pos"].append(np.zeros(3))
            out["kin_joint_qpos0"].append(0.0)
    out = {k: np.array(v) for k, v in out.items()}
    sites = [m.site(f"robot/{s}") for s in ball_sites]
    out["ball_site_body"] = np.array([index[int(m.site_bodyid[s.id])] for s in sites])
    out["ball_site_pos"] = np.array([m.site_pos[s.id].copy() for s in sites])
    imu = m.site("robot/imu")
    if index.get(int(m.site_bodyid[imu.id])) != 0:
        sys.exit("the imu site is not on the trunk")
    out["imu_quat"] = m.site_quat[imu.id].copy()
    return out


def check_kinematics(env, bundle: dict, joint_names: list[str], ball_sites, move_mod, n: int = 50) -> float:
    """Largest ball position difference (m) between the move's forward kinematics and MuJoCo's, random poses."""
    import mujoco

    m = env.sim.mj_model
    d = mujoco.MjData(m)
    kin = move_mod.Kinematics(bundle)
    rng = np.random.default_rng(0)
    adr = [m.jnt_qposadr[m.joint(f"robot/{n}").id] for n in joint_names]
    site_ids = [m.site(f"robot/{s}").id for s in ball_sites]
    trunk = m.body("robot/trunk").id
    worst = 0.0
    for _ in range(n):
        q = rng.uniform(-1.0, 1.0, len(joint_names))
        d.qpos[:] = m.qpos0
        d.qpos[adr] = q
        mujoco.mj_kinematics(m, d)
        R, p = d.xmat[trunk].reshape(3, 3), d.xpos[trunk]
        ref = np.array([R.T @ (d.site_xpos[s] - p) for s in site_ids])
        worst = max(worst, float(np.abs(kin.balls(q) - ref).max()))
    return worst


def export_onnx(runner, path: Path) -> None:
    import torch

    model = runner.alg.get_policy().as_onnx(verbose=False)
    model.to("cpu").eval()
    torch.onnx.export(model, (torch.zeros(1, model.input_size),), str(path), export_params=True, opset_version=18,
                      input_names=["obs"], output_names=["actions"], dynamic_axes={}, dynamo=False)


def main() -> None:
    args = parse_args()
    motion_file = args.motion_file.resolve()
    if not motion_file.is_file():
        sys.exit(f"motion not found: {motion_file}")

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import mjlab_microban.tasks  # noqa: F401  (registers the tasks; the wall comes from --motion-file)
    from mjlab_microban.robot.microban_climb_constants import BALL_SITE_NAMES
    from mjlab_microban.tasks.climbing import mdp as climb_mdp
    from mjlab_microban.tasks.climbing.microban_climbing_env_cfg import KP
    from mjlab_microban.tasks.climbing.wall import hold_positions, motion_wall, sim_wall

    sys.path.insert(0, str(RCR_SRC))
    from rcr.controller import mjlab_policy as move_mod  # the robot's code, run here against the env

    task = args.task
    if "Multi-Board" in task:
        sys.exit("multi-board tasks: export one board at a time is not supported yet (give the single-board task)")
    checkpoint = resolve_checkpoint(args, task)
    out = args.out or Path("logs/deploy") / f"{checkpoint.parent.name}_{checkpoint.stem}"
    out.mkdir(parents=True, exist_ok=True)
    if not args.force and cached_bundle(out, task, checkpoint, motion_file, move_mod.BUNDLE_FORMAT):
        print(f"[INFO] bundle: {out.resolve()}")
        return

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.commands["motion"].motion_file = str(motion_file)
    if "base_yaw" in env_cfg.observations["actor"].terms:  # the robot starts its heading at 0
        env_cfg.observations["actor"].terms["base_yaw"].params["offset_range"] = (0.0, 0.0)
    agent_cfg = load_rl_cfg(task)
    if tuple(agent_cfg.obs_groups.get("actor", ("actor",))) != ("actor",):
        sys.exit(f"{task}: the actor reads the groups {agent_cfg.obs_groups['actor']}, only 'actor' is supported")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(task))(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)
    command = env.command_manager.get_term("motion")
    joint_term = env.action_manager.get_term("joint_pos")
    if joint_term.cfg.clip is not None:
        sys.exit("the joint_pos action has a clip: not implemented on the robot")
    if [joint_names[i] for i in joint_term._target_ids.tolist()] != joint_names:
        sys.exit("the joint_pos action does not drive every joint in the robot's order")
    has_phase = "phase_rate" in env.action_manager.active_terms
    if env.action_manager.active_terms[-1:] != (["phase_rate"] if has_phase else env.action_manager.active_terms[-1:]):
        sys.exit("the phase_rate action has to be the last one")
    layout = actor_layout(env)

    # -- the reference (motion frame = env frame without the env origin) -----------------------------------------
    motion = command.motion
    anchor = command.cfg.body_names.index(command.cfg.anchor_body_name)
    ball_idx, ball_off = climb_mdp._ball_site_offsets(env, command, BALL_SITE_NAMES, "robot")
    from mjlab.utils.lab_api.math import quat_apply

    bp, bq = motion.body_pos_w[:, ball_idx], motion.body_quat_w[:, ball_idx]
    ref_balls = (bp + quat_apply(bq, ball_off.expand_as(bp))).cpu().numpy()
    wall = motion_wall(motion_file)
    holds = np.asarray(hold_positions(wall))
    dist = np.linalg.norm(ref_balls[:, :, None, :] - holds[None, None], axis=-1).min(-1)
    ref_contact = dist < args.contact_threshold
    print(f"[INFO] {len(ref_contact)} frames; balls on holds (estimate): "
          f"{dict(zip(BALL_SITE_NAMES, ref_contact.mean(0).round(2)))}; frames with none: {(~ref_contact.any(1)).sum()}")

    # One step to get the residual action scale (computed from the BAM gain in process_actions).
    wrapped.reset()
    wrapped.step(torch.zeros(1, env.action_manager.total_action_dim))
    scale = joint_term._scale.reshape(-1).cpu().numpy() * np.ones(len(joint_names))

    cmd_cfg = next(c for n, c in zip(env.observation_manager.active_terms["actor"],
                                     env.observation_manager._group_obs_term_cfgs["actor"]) if n == "command")
    num_steps, stride = int(cmd_cfg.params.get("num_steps", 20)), int(cmd_cfg.params.get("stride", 1))
    future_offsets = cmd_cfg.params.get("offsets")
    future_offsets = (np.arange(1, num_steps + 1) * stride if future_offsets is None
                      else np.asarray(future_offsets, dtype=np.int64))
    phase_cfg = env.action_manager.get_term("phase_rate").cfg if has_phase else None
    bundle = {
        "format": move_mod.BUNDLE_FORMAT,
        "task": task, "checkpoint": str(checkpoint), "motion_file": str(motion_file), "wall": str(wall),
        "kp_fw": float(KP), "step_dt": float(env.step_dt),
        "joint_names": np.array(joint_names),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().numpy(),
        "action_scale": scale,
        "clip_actions": float("nan") if agent_cfg.clip_actions is None else float(agent_cfg.clip_actions),
        "num_actions": env.action_manager.total_action_dim,
        "obs_terms": np.array([t for t, _, _ in layout]),
        "obs_dims": np.array([d for _, d, _ in layout]),
        "obs_history": np.array([h for _, _, h in layout]),
        "future_offsets": future_offsets,
        "ref_joint_pos": motion.joint_pos.cpu().numpy(),
        "ref_trunk_pos": motion.body_pos_w[:, anchor].cpu().numpy(),
        "ref_trunk_quat": motion.body_quat_w[:, anchor].cpu().numpy(),
        "ref_ball_pos": ref_balls,
        "ref_contact": ref_contact,
        "ball_site_names": np.array(BALL_SITE_NAMES),
        "phase_enabled": has_phase,
        "phase_sigma": float(phase_cfg.sigma) if has_phase else 0.0,
        "phase_k": float(phase_cfg.k) if has_phase else 1.0,
        "phase_delta_clip": float("nan") if not has_phase or getattr(phase_cfg, "delta_clip", None) is None
        else float(phase_cfg.delta_clip),
        "phase_rate_terms": np.array(list(phase_cfg.rate_terms) if has_phase else [], dtype=str),
        "phase_tolerances": np.array([phase_cfg.tolerances[t] for t in phase_cfg.rate_terms] if has_phase else []),
        "phase_min_rate": float("nan") if not has_phase or phase_cfg.min_rate is None else float(phase_cfg.min_rate),
        **kinematics(env, joint_names, BALL_SITE_NAMES),
    }
    if has_phase and not torch.allclose(command.nominal_speed, torch.ones_like(command.nominal_speed)):
        sys.exit("the play env's nominal motion speed is not 1")
    np.savez(out / "climb_bundle.npz", **bundle)
    export_onnx(runner, out / "policy.onnx")
    sim_wall(wall).to_json(out / "wall.json")
    info = {k: bundle[k] for k in ("task", "checkpoint", "motion_file", "wall", "kp_fw", "format")}
    info["parity"] = "pending"  # not reusable until the check below passes
    (out / "info.json").write_text(json.dumps(info, indent=2))
    fk_err = check_kinematics(env, bundle, joint_names, BALL_SITE_NAMES, move_mod)
    print(f"[INFO] forward kinematics of the move vs MuJoCo: {fk_err * 1e3:.4f} mm max")
    if fk_err > 1e-5:
        sys.exit("the move's forward kinematics do not match the model")

    ok = True
    if args.check_steps > 0:
        ok = parity_check(args, env, wrapped, policy, command, robot, move_mod, out, layout, has_phase)
    info["parity"] = ("passed" if ok else "failed") if args.check_steps > 0 else "skipped"
    info["parity_ticks"] = args.check_steps
    (out / "info.json").write_text(json.dumps(info, indent=2))
    env.close()
    print(f"[INFO] bundle: {out.resolve()}")
    if not ok:
        sys.exit("[ERROR] the move does not reproduce the env: do not deploy this bundle")


def parity_check(args, env, wrapped, policy, command, robot, move_mod, out: Path, layout, has_phase) -> bool:
    """The env runs the policy; the move's controller, fed the same measurements, has to give the same observation,
    action and targets (with the env's v_ref). Its own estimated v_ref is compared to the env's."""
    import torch

    from mjlab_microban.robot.microban_climb_constants import BALL_SITE_NAMES

    om = env.observation_manager
    gyro_cfg = next(c for n, c in zip(om.active_terms["actor"], om._group_obs_term_cfgs["actor"])
                    if n == "base_ang_vel")
    site_ids = robot.find_sites(BALL_SITE_NAMES, preserve_order=True)[0]
    origin = env.scene.env_origins[0]

    wrapped.reset()
    obs = wrapped.get_observations()
    ctrl = move_mod.ClimbController(out)
    # The phase integrates the policy's dv: onnxruntime's and torch's actions differ by ~1e-6, which, integrated over
    # minutes, ends up putting a floor() on the other side of a frame. ctrl is re-synced to the env's phase every
    # tick (each tick checked exactly); free runs on its own, for how long the two stay identical.
    free = move_mod.ClimbController(out)
    free_diverged = None
    env_state = (0.0, 0, 1.0)  # the env's (phase, frame, speed) of the last tick
    est = move_mod.ClimbController(out)  # for the estimate only (reset for its yaw alignment)

    def measure() -> tuple[dict, dict]:
        d = robot.data
        meas = {"q": d.joint_pos_biased[0].cpu().numpy().astype(float),
                "dq": d.joint_vel[0].cpu().numpy().astype(float),
                "gyro": gyro_cfg.func(env, **gyro_cfg.params)[0].cpu().numpy().astype(float),
                "gravity": d.projected_gravity_b[0].cpu().numpy().astype(float),
                "quat": d.root_link_quat_w[0].cpu().numpy().astype(float)}
        truth = {"trunk_pos": (d.root_link_pos_w[0] - origin).cpu().numpy().astype(float),
                 "trunk_R": move_mod.quat_mat(meas["quat"]),
                 "balls": (d.site_pos_w[0, site_ids] - origin).cpu().numpy().astype(float)}
        return meas, truth

    rec: dict[str, list] = {k: [] for k in (
        "obs_env", "obs_move", "act_env", "act_move", "target_env", "target_move", "v_ref_env", "v_ref_truth",
        "v_ref_est", "frame_env", "frame_move", "trunk_pos", "trunk_pos_est", "trunk_ori_err_est")}
    steps = min(args.check_steps, ctrl.T - 2)
    for k in range(steps):
        meas, truth = measure()
        v_env = float(command.phase_rate_ref[0]) if has_phase else 1.0
        if k == 0:  # the climb starts (the env's reset): its first update used the v_ref of the reset state
            ctrl.reset(meas, v_ref=v_env)
            free.reset(meas, v_ref=v_env)
            est.reset(meas)
        frame = ctrl.t  # the env's v_ref of this tick was computed against this frame
        if k == 0:
            v_truth = v_est = 1.0
        else:
            v_truth = ctrl.rate_ref(meas, frame, state=truth)  # the move's formula on the true state
            v_est = est.rate_ref(meas, frame)  # the move's estimate
        e = est.estimate(meas, frame)
        if k > 0:  # the env's phase before this tick's update
            ctrl.phase, ctrl.t, ctrl.speed = env_state
        target_move = ctrl.tick(meas, v_ref=v_env if k > 0 else None)
        free.tick(meas, v_ref=v_env if k > 0 else None)

        obs_env = obs["actor"][0].cpu().numpy()
        if free_diverged is None and np.abs(free.obs - obs_env).max() > 1e-4:
            free_diverged = k
        env_state = (float(command.phase[0]), int(command.time_steps[0]), float(command.speed[0])) if has_phase \
            else (float(command.time_steps[0]), int(command.time_steps[0]), 1.0)
        with torch.inference_mode():
            action = policy(obs)
        obs, _, dones, _ = wrapped.step(action)
        rec["obs_env"].append(obs_env)
        rec["obs_move"].append(ctrl.obs.astype(float))
        rec["act_env"].append(action[0].cpu().numpy())
        rec["act_move"].append(ctrl.last_action.copy())
        # the env subtracts its encoder bias from the targets; the robot's encoders are what they are
        rec["target_env"].append((robot.data.joint_pos_target[0] + robot.data.encoder_bias[0]).cpu().numpy())
        rec["target_move"].append(target_move)
        rec["v_ref_env"].append(v_env)
        rec["v_ref_truth"].append(v_truth)
        rec["v_ref_est"].append(v_est)
        rec["frame_env"].append(int(command.time_steps[0]))
        rec["frame_move"].append(ctrl.t)
        rec["trunk_pos"].append(truth["trunk_pos"])
        rec["trunk_pos_est"].append(e["trunk_pos"])
        rec["trunk_ori_err_est"].append(move_mod.rotation_angle(truth["trunk_R"], e["trunk_R"]))
        if bool(dones[0]):
            print(f"[WARN] the env terminated (the robot fell) at tick {k}: the check stops there")
            break
    r = {k: np.array(v) for k, v in rec.items()}
    np.savez(out / "parity.npz", **r)

    n = len(r["obs_env"])
    d_obs = np.abs(r["obs_env"] - r["obs_move"])
    d_act = np.abs(r["act_env"] - r["act_move"]).max()
    d_tgt = np.abs(r["target_env"] - r["target_move"]).max()
    d_vf = np.abs(r["v_ref_env"] - r["v_ref_truth"]).max()
    # the env's frame after its step k is the one the move advances to at its tick k + 1
    frames_ok = np.array_equal(r["frame_env"][:-1], r["frame_move"][1:])
    worst = int(d_obs.max(0).argmax())
    bounds = np.cumsum([0] + [d * max(h, 1) for _, d, h in layout])
    term = layout[int(np.searchsorted(bounds, worst, side="right")) - 1][0]
    print(f"[CHECK] {n} ticks ({n * env.step_dt:.1f} s), phase reached frame {r['frame_env'][-1]}")
    print(f"[CHECK] observation   max |env - move| = {d_obs.max():.2e}  (worst: {term}[{worst}])")
    print(f"[CHECK] action        max |env - move| = {d_act:.2e}")
    print(f"[CHECK] joint targets max |env - move| = {d_tgt:.2e} rad")
    print(f"[CHECK] reference frames identical: {frames_ok}")
    print(f"[CHECK] (free-running, not re-synced: identical for "
          f"{'all ' + str(n) if free_diverged is None else free_diverged} ticks; past that, float rounding of the phase)")
    print(f"[CHECK] v_ref formula on the true state max |env - move| = {d_vf:.2e}")
    pos_err = np.linalg.norm(r["trunk_pos"] - r["trunk_pos_est"], axis=-1)
    print(f"[CHECK] estimate: trunk position error mean {pos_err.mean() * 1e3:.1f} mm, max {pos_err.max() * 1e3:.1f} mm;"
          f" orientation error max {np.degrees(r['trunk_ori_err_est'].max()):.2f} deg")
    if has_phase:
        dv = np.abs(r["v_ref_env"] - r["v_ref_est"])
        print(f"[CHECK] estimate: v_ref |env - estimated| mean {dv.mean():.3f}, max {dv.max():.3f} "
              f"(env v_ref mean {r['v_ref_env'].mean():.3f}, estimated {r['v_ref_est'].mean():.3f})")
    ok = d_obs.max() < 1e-4 and d_act < 1e-3 and d_tgt < 1e-3 and frames_ok and d_vf < 1e-3
    print(f"[CHECK] {'PASSED' if ok else 'FAILED'}: the move {'reproduces' if ok else 'does NOT reproduce'} the env")
    return ok


if __name__ == "__main__":
    main()
