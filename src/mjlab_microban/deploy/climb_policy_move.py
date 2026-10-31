# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""A trained Mjlab-Climbing-Microban[-Adaptive-Phase] policy as a Microban move: the 50 Hz loop on the robot.

The same file runs on the real robot (Microban's program, branch dev_v2: ``make run MOVES=climb_policy_move.py``) and in
the organisers' simulator (``climbing_sim.run_move``, ``make sim WALL=... MOVES=...``). It imports numpy and onnxruntime
only, which Microban's Pi environment has; nothing of mjlab.

It reads a deployment bundle made on the laptop by ``src/mjlab_microban/scripts/export_climbing_deploy.py``
(``train_climbing.sh --deploy ...``), a folder with:
  - ``policy.onnx``: the actor (obs normaliser + MLP), obs -> actions;
  - ``climb_bundle.npz``: the planned climb (reference joint positions, trunk poses, ball positions, expected ball
    contacts), the actor's observation layout, the action scale, the adaptive phase parameters and the robot's
    kinematic tree (for the forward kinematics of the balls);
found at $MICROBAN_CLIMB_BUNDLE, else next to this file, else in ``climb_policy/`` next to it, else ``~/climb_policy``.

What the env computed from the simulator, this file computes from what the robot measures (encoders, IMU):
  - the actor's observation, term by term, with the same proprioceptive history (oldest first, backfilled at the
    start), in the env's joint order;
  - the residual targets: reference joint positions of the frame the step ends on + action scale * action;
  - the adaptive phase (Sleiman et al., CoRL 2024): phase += v, v = max(v_ref + sigma * dv, min_rate), where v_ref
    measures how well the robot tracks the reference (balls, trunk position, trunk orientation). The env had the
    true trunk pose; here it is **estimated** [ours]: the trunk orientation is the IMU's, its yaw aligned at the start
    so that the four balls sit on their start holds, and the trunk position is the one that puts the balls the plan
    has on holds at this frame (``ref_contact``) where the plan has them (forward kinematics from the encoders,
    least squares over those balls). A ball that slips off its hold is not seen by this estimate.

Sequence: on start, the joints ramp from where they are to the climb's first frame (``ramp_s``), hold it
(``start_delay_s``), then the policy climbs. At the end of the climb the policy keeps tracking the last frame until the
move is stopped. Stopped, it holds its last targets ``stop_hold_s`` then ends (Microban turns the torque off: the
robot hangs on its rope). If something goes wrong (non-finite measurement or action, the trunk tilted past
``max_tilt_deg``), the climb is aborted: the robot holds the joint positions it measured then, until stopped.
Every tick of the climb is logged to ``~/climb_logs/<time>.npz`` when the move stops (``save_log``).
"""

import math
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

BUNDLE_FORMAT = 1
PHASE_EPS = 1e-4  # as mjlab_microban.tasks.climbing.mdp._PHASE_EPS
G_DOWN = np.array([0.0, 0.0, -1.0])


# --------------------------------------------------------------------------------------------------------------------
# Rotations (quaternions w x y z)
# --------------------------------------------------------------------------------------------------------------------


def quat_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def mat_quat(R) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w >= 0)."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12))
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q * (1.0 if q[0] >= 0 else -1.0) / np.linalg.norm(q)


def axis_angle_mat(axis, angle) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])  # fmt: skip


def rot_z(angle) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def rotation_angle(Ra, Rb) -> float:
    """Angle of the rotation between two orientations (mjlab's quat_error_magnitude)."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.acos(min(1.0, max(-1.0, c)))


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# --------------------------------------------------------------------------------------------------------------------
# The bundle and the controller (no Microban, no simulator: also run against the env by the export's parity check)
# --------------------------------------------------------------------------------------------------------------------


def find_bundle(explicit=None) -> Path | None:
    here = Path(__file__).resolve().parent
    candidates = [explicit, os.environ.get("MICROBAN_CLIMB_BUNDLE"), here, here / "climb_policy",
                  Path.home() / "climb_policy"]
    for c in candidates:
        if c and (Path(c) / "climb_bundle.npz").is_file() and (Path(c) / "policy.onnx").is_file():
            return Path(c)
    return None


class Kinematics:
    """Forward kinematics of the robot's bodies in the trunk frame (MuJoCo's convention, from the env's model)."""

    def __init__(self, b) -> None:
        self.parent = b["kin_parent"].astype(int)
        self.pos, self.quat = b["kin_body_pos"], b["kin_body_quat"]
        self.joint = b["kin_joint"].astype(int)          # index in the env's joint order, -1: fixed to its parent
        self.axis, self.jpos, self.qpos0 = b["kin_joint_axis"], b["kin_joint_pos"], b["kin_joint_qpos0"]
        self.R_local = [quat_mat(q) for q in self.quat]
        self.site_body = b["ball_site_body"].astype(int)
        self.site_pos = b["ball_site_pos"]

    def balls(self, q) -> np.ndarray:
        """Ball centres in the trunk frame, [N, 3]."""
        n = len(self.parent)
        Rs, ps = [None] * n, [None] * n
        for i in range(n):  # parents first
            if self.parent[i] < 0:
                Rs[i], ps[i] = np.eye(3), np.zeros(3)
                continue
            Rp, pp = Rs[self.parent[i]], ps[self.parent[i]]
            R, p = Rp @ self.R_local[i], pp + Rp @ self.pos[i]
            j = self.joint[i]
            if j >= 0:
                anchor = p + R @ self.jpos[i]
                R = R @ axis_angle_mat(self.axis[i], q[j] - self.qpos0[i])
                p = anchor - R @ self.jpos[i]
            Rs[i], ps[i] = R, p
        return np.array([ps[b] + Rs[b] @ s for b, s in zip(self.site_body, self.site_pos)])


class ClimbController:
    """The policy, its observation and the reference phase, from the robot's measurements; one ``tick`` every 20 ms.

    ``meas``: q, dq (env joint order, rad, rad/s), gyro (the IMU's angular velocity, IMU frame), gravity (unit, trunk
    frame), quat (trunk orientation w x y z, any yaw reference). ``tick`` returns the 19 joint targets (env order).
    """

    def __init__(self, bundle_dir, min_phase_rate=None, adaptive=True) -> None:
        bundle_dir = Path(bundle_dir)
        b = dict(np.load(bundle_dir / "climb_bundle.npz", allow_pickle=False))
        if int(b["format"]) != BUNDLE_FORMAT:
            raise ValueError(f"bundle format {int(b['format'])}, this move reads {BUNDLE_FORMAT}: export it again")
        self.b = b
        self.session = ort.InferenceSession(str(bundle_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.joint_names = [str(n) for n in b["joint_names"]]
        self.default = b["default_joint_pos"]
        self.scale = b["action_scale"]
        self.clip = float(b["clip_actions"])  # NaN: none
        self.num_actions = int(b["num_actions"])
        self.terms = [str(t) for t in b["obs_terms"]]
        self.term_dims = b["obs_dims"].astype(int)
        self.term_hist = b["obs_history"].astype(int)
        self.future_offsets = b["future_offsets"].astype(int)
        self.ref_q = b["ref_joint_pos"]
        self.ref_trunk_pos, self.ref_trunk_R = b["ref_trunk_pos"], np.array([quat_mat(q) for q in b["ref_trunk_quat"]])
        self.ref_balls, self.ref_contact = b["ref_ball_pos"], b["ref_contact"].astype(bool)
        self.T = len(self.ref_q)
        self.R_imu = quat_mat(b["imu_quat"])  # IMU site in the trunk
        self.kin = Kinematics(b)
        self.has_phase = bool(b["phase_enabled"])
        self.sigma, self.k = float(b["phase_sigma"]), float(b["phase_k"])
        self.rate_terms = [str(t) for t in b["phase_rate_terms"]]
        self.tolerances = dict(zip(self.rate_terms, b["phase_tolerances"].astype(float)))
        bundle_min = float(b["phase_min_rate"])
        self.min_rate = bundle_min if min_phase_rate is None else min_phase_rate
        self.min_rate = None if self.min_rate is None or math.isnan(self.min_rate) else self.min_rate
        self.adaptive = adaptive
        self.obs_size = int(sum(self.term_dims * np.maximum(self.term_hist, 1)))
        expected = self.session.get_inputs()[0].shape[-1]
        if isinstance(expected, int) and expected != self.obs_size:
            raise ValueError(f"policy.onnx takes {expected} inputs, the bundle's layout makes {self.obs_size}")

    # -- the reference phase ---------------------------------------------------------------------------------------

    def reset(self, meas) -> None:
        """The start of the climb (the env's reset): frame 0, history backfilled, last action 0."""
        self.t, self.phase, self.speed, self.v_ref = 0, 0.0, 1.0, 1.0
        self.last_action = np.zeros(self.num_actions)
        self.history: dict[str, list] = {}
        self.ticks = 0
        self.finished = False
        R = quat_mat(meas["quat"])
        self.yaw_align = self._yaw_alignment(meas["q"], R)
        self.yaw_windup, self.prev_yaw = 0.0, yaw_of(R)
        self.est = None

    def frames_ahead(self, offsets) -> np.ndarray:
        frac = self.phase - self.t
        if not (-PHASE_EPS <= frac < 1.0):
            frac = 0.0
        idx = self.t + np.floor(frac + self.speed * np.asarray(offsets, float) + PHASE_EPS).astype(int)
        return np.clip(idx, 0, self.T - 1)

    def _advance(self, dv: float, v_ref: float) -> None:
        """The motion command's update after a step: v from v_ref and the policy's dv, then the phase."""
        self.v_ref = v_ref
        if self.has_phase:
            v = v_ref + self.sigma * dv
            if self.min_rate is not None:
                v = max(v, self.min_rate)
            self.speed = max(v, -self.phase)  # not before the first frame
        frac = self.phase - self.t
        if not (-PHASE_EPS <= frac < 1.0):
            self.phase = float(self.t)
        self.phase += self.speed
        self.t = int(math.floor(self.phase + PHASE_EPS))
        if self.t >= self.T - 1:  # the end of the climb: stay on the last frame (the env restarts it)
            self.t, self.phase, self.finished = self.T - 1, float(self.T - 1), True

    # -- the state estimate and v_ref ------------------------------------------------------------------------------

    def _yaw_alignment(self, q, R_imu_w) -> float:
        """Yaw to add to the IMU's orientation so that the four balls best match the climb's first frame (Procrustes
        about z), where the robot starts: hanging on the start holds."""
        a = (R_imu_w @ self.kin.balls(q).T).T
        b = self.ref_balls[0]
        a, b = a - a.mean(0), b - b.mean(0)
        return math.atan2(float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])),
                          float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))

    def estimate(self, meas, frame: int) -> dict:
        """Trunk pose and ball positions in the motion frame: IMU orientation (yaw aligned), position from the balls
        the plan has on holds at ``frame``."""
        R = rot_z(self.yaw_align) @ quat_mat(meas["quat"])
        balls_b = self.kin.balls(meas["q"])
        stance = self.ref_contact[frame]
        if stance.any():
            pos = np.mean(self.ref_balls[frame][stance] - (R @ balls_b[stance].T).T, axis=0)
        else:
            pos = self.ref_trunk_pos[frame].copy()
        return {"trunk_pos": pos, "trunk_R": R, "balls": pos + (R @ balls_b.T).T, "stance": stance}

    def rate_ref(self, meas, frame: int, state=None) -> float:
        """v_ref = prod_i min(k ** (1 - e_i / tol_i^2), 1) against the reference ``frame``. ``state``: the trunk pose
        and balls (estimate() by default; the parity check gives the simulator's)."""
        if not self.has_phase:
            return 1.0
        if not self.adaptive:
            return 1.0
        s = self.estimate(meas, frame) if state is None else state
        self.est = s
        errors = {}
        if "balls" in self.rate_terms:
            errors["balls"] = float(np.mean(np.sum((self.ref_balls[frame] - s["balls"]) ** 2, axis=-1)))
        if "trunk_pos" in self.rate_terms:
            errors["trunk_pos"] = float(np.sum((self.ref_trunk_pos[frame] - s["trunk_pos"]) ** 2))
        if "trunk_ori" in self.rate_terms:
            errors["trunk_ori"] = rotation_angle(self.ref_trunk_R[frame], s["trunk_R"]) ** 2
        if "joint_pos" in self.rate_terms:
            errors["joint_pos"] = float(np.mean((self.ref_q[frame] - meas["q"]) ** 2))
        v, log_k = 1.0, math.log(self.k)
        for name, e in errors.items():
            v *= min(math.exp(log_k * (1.0 - e / self.tolerances[name] ** 2)), 1.0)
        return 0.0 if math.isnan(v) else v

    # -- the observation and the policy ----------------------------------------------------------------------------

    def _term(self, name: str, meas) -> np.ndarray:
        if name == "base_ang_vel":
            return meas["gyro"]
        if name == "projected_gravity":
            return meas["gravity"]
        if name == "base_yaw":
            return np.array([math.cos(self.yaw_windup), math.sin(self.yaw_windup)])
        if name == "joint_pos":
            return meas["q"] - self.default
        if name == "joint_vel":
            return meas["dq"]
        if name == "actions":
            return self.last_action
        if name == "command":
            return (self.ref_q[self.frames_ahead(self.future_offsets)] - self.default).reshape(-1)
        if name == "phase":
            return np.array([self.phase / max(self.T - 1.0, 1.0), self.v_ref])
        raise ValueError(f"unknown observation term {name!r}")

    def observation(self, meas) -> np.ndarray:
        parts = []
        for name, dim, hist in zip(self.terms, self.term_dims, self.term_hist):
            x = np.asarray(self._term(name, meas), float).reshape(-1)
            assert x.shape == (dim,), f"{name}: {x.shape} != ({dim},)"
            if hist > 0:
                buf = self.history.setdefault(name, [x] * hist)  # backfilled on the first append
                if self.ticks > 0:
                    buf.append(x)
                    del buf[0]
                parts.extend(buf)
            else:
                parts.append(x)
        return np.concatenate(parts)

    def tick(self, meas, v_ref=None, state=None) -> np.ndarray:
        """One 20 ms step: the reference advances (from the second tick on, with the last action's dv), the
        observation is built, the policy runs, and the joint targets are returned. ``v_ref`` / ``state``: given by
        the parity check instead of the estimate."""
        if self.ticks > 0:
            if v_ref is None:
                v_ref = self.rate_ref(meas, self.t, state)
            dv = float(self.last_action[-1]) if self.has_phase else 0.0
            self._advance(dv, v_ref)
        yaw = yaw_of(quat_mat(meas["quat"]))
        if math.isfinite(yaw):
            self.yaw_windup += wrap(yaw - self.prev_yaw)
            self.prev_yaw = yaw
        obs = self.observation(meas).astype(np.float32)
        action = self.session.run(None, {self.input_name: obs[None]})[0][0].astype(float)
        if not math.isnan(self.clip):
            action = np.clip(action, -self.clip, self.clip)
        self.last_action = action
        self.obs = obs
        self.ticks += 1
        ref = self.ref_q[self.frames_ahead([1])[0]]
        return ref + self.scale * action[: len(self.scale)]


# --------------------------------------------------------------------------------------------------------------------
# The Microban move
# --------------------------------------------------------------------------------------------------------------------


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class ClimbPolicyMove:
    """Climb with the trained RL policy (Mjlab-Climbing-Microban, adaptive phase), from the climb's start holds."""

    description = "Climb (learned policy): ramp to the climb's first pose, then the policy climbs"
    variants = ("climber",)
    start_pose = "current"  # torque on where the robot is: it is on the wall
    kp = 120  # the BAM firmware P gain the policy was trained with (overwritten by the bundle's kp_fw)
    ki = 0
    kd = 0
    options = {
        "ramp_s": 2.0,
        "start_delay_s": 2.0,
        "stop_hold_s": 5.0,
        "adaptive_phase": True,
        "min_phase_rate": 0.1,
        "max_tilt_deg": 60.0,
        "save_log": True,
    }
    option_docs = {
        "ramp_s": "s, from the current pose to the climb's first frame",
        "start_delay_s": "s, holding the first frame before the policy starts",
        "stop_hold_s": "s, holding the last targets once stopped (then the torque goes off)",
        "adaptive_phase": "the reference waits for the robot (v_ref estimated); off: nominal speed",
        "min_phase_rate": "the reference advances at least at this fraction of its speed",
        "max_tilt_deg": "trunk tilt from the vertical that aborts the climb (holds where it is)",
        "save_log": "every tick to ~/climb_logs/<time>.npz when it stops",
    }

    def __init__(self) -> None:
        self.bundle_dir = find_bundle()
        if self.bundle_dir is None:
            print("[climb_policy] no bundle (climb_bundle.npz + policy.onnx): set MICROBAN_CLIMB_BUNDLE, or copy it to "
                  "~/climb_policy (train_climbing.sh --deploy --host ...). The move is not listed.")
            self.variants = ()
            return
        b = np.load(self.bundle_dir / "climb_bundle.npz", allow_pickle=False)
        self.kp = int(round(float(b["kp_fw"])))
        self.options = dict(self.options, min_phase_rate=float(b["phase_min_rate"])
                            if not math.isnan(float(b["phase_min_rate"])) else 0.0)
        self._joints = [str(n) for n in b["joint_names"]]
        self.description += f" [{str(b['task'])}, {Path(str(b['checkpoint'])).name}, wall {str(b['wall'])}]"
        self._ctrl = ClimbController(self.bundle_dir)  # loads the network once, at the program's start
        R_imu = quat_mat(b["imu_quat"])
        self._trunk_to_imu = R_imu.T
        print(f"[climb_policy] bundle {self.bundle_dir}: {self.description}")

    # -- Microban's observation -> the controller's measurements (env joint order) ---------------------------------

    def _meas(self, obs) -> dict:
        from_joints = self._index
        gravity = np.asarray(obs["gravity"], float)
        quat = np.asarray(obs["quat"], float)
        if not np.isfinite(quat).all():
            quat = self._last_quat
        if not np.isfinite(gravity).all():
            gravity = quat_mat(quat).T @ G_DOWN
        self._last_quat = quat
        return {"q": np.asarray(obs["q"], float)[from_joints], "dq": np.asarray(obs["dq"], float)[from_joints],
                "gyro": self._trunk_to_imu @ np.asarray(obs["gyro"], float),  # Microban gives it in the trunk frame
                "gravity": gravity, "quat": quat}

    def _targets(self, targets_env) -> dict:
        return dict(zip(self._joints, np.asarray(targets_env, float).tolist()))

    # -- the move --------------------------------------------------------------------------------------------------

    def start(self, obs) -> None:
        if self.bundle_dir is None:
            raise RuntimeError("no climb bundle on this machine (~/climb_policy): train_climbing.sh --deploy --host")
        self._robot_joints = list(getattr(self, "_robot_joints", None) or _microban_joints(obs))
        self._index = [self._robot_joints.index(n) for n in self._joints]
        self._last_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._t0 = obs["time"]
        self._q0 = np.asarray(obs["q"], float)[self._index]
        self._first = self._ctrl.ref_q[0]
        self._stage, self._stage_t0 = "ramp", obs["time"]
        self._last = self._q0.copy()
        self._log: list[dict] = []
        self._reason = ""
        self._stop_t0 = None

    def step(self, obs):
        now = obs["time"]
        meas = self._meas(obs)
        opt = self.options
        if self._stage in ("ramp", "wait"):
            if self._stage == "ramp":
                a = _smoothstep((now - self._stage_t0) / max(float(opt["ramp_s"]), 1e-3))
                self._last = (1.0 - a) * self._q0 + a * self._first
                if a >= 1.0:
                    self._stage, self._stage_t0 = "wait", now
            elif now - self._stage_t0 >= float(opt["start_delay_s"]):
                self._ctrl.adaptive = bool(opt["adaptive_phase"])
                self._ctrl.min_rate = float(opt["min_phase_rate"])
                self._ctrl.reset(meas)
                self._stage, self._stage_t0 = "climb", now
                print(f"[climb_policy] climbing ({self._ctrl.T} frames, {self._ctrl.T * 0.02:.1f} s at nominal speed)")
            if self._stage != "climb":
                return self._targets(self._last)

        if self._stage == "climb":
            reason = self._check(meas)
            if not reason:
                t_start = time.perf_counter()
                try:
                    targets = self._ctrl.tick(meas)
                except Exception as e:  # noqa: BLE001
                    targets, reason = None, f"policy failed: {e!r}"
                dt = time.perf_counter() - t_start
                if targets is not None and not np.isfinite(targets).all():
                    reason = "non-finite targets"
                if not reason:
                    self._last = targets
                    self._record(now, meas, targets, dt)
                    if dt > 0.015:
                        print(f"[climb_policy] slow tick: {dt * 1e3:.1f} ms")
            if reason:
                self._abort(reason, meas)
        return self._targets(self._last)

    def _check(self, meas) -> str:
        if not (np.isfinite(meas["q"]).all() and np.isfinite(meas["dq"]).all()):
            return "non-finite joint measurement"
        g = meas["gravity"]
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, -float(g[2]) / max(np.linalg.norm(g), 1e-9)))))
        if tilt > float(self.options["max_tilt_deg"]):
            return f"trunk tilted {tilt:.0f} deg"
        return ""

    def _abort(self, reason: str, meas) -> None:
        self._reason = reason
        self._stage = "aborted"
        if np.isfinite(meas["q"]).all():
            self._last = meas["q"].copy()
        print(f"[climb_policy] climb aborted at frame {self._ctrl.t}: {reason}. Holding; stop the move to end.")

    def _record(self, now, meas, targets, dt) -> None:
        c = self._ctrl
        if c.finished and not getattr(self, "_said_end", False):
            self._said_end = True
            print("[climb_policy] end of the climb: tracking its last frame until stopped")
        if not self.options["save_log"]:
            return
        est = c.est or {}
        self._log.append({"time": now, "q": meas["q"], "dq": meas["dq"], "gyro": meas["gyro"],
                          "gravity": meas["gravity"], "quat": meas["quat"], "targets": targets,
                          "action": c.last_action, "frame": c.t, "phase": c.phase, "v_ref": c.v_ref,
                          "speed": c.speed, "tick_s": dt,
                          "trunk_pos_est": est.get("trunk_pos", np.full(3, np.nan))})

    def stopping(self, obs):
        if self._stop_t0 is None:
            self._stop_t0 = obs["time"]
            self._save_log()
        if obs["time"] - self._stop_t0 < float(self.options["stop_hold_s"]):
            return self._targets(self._last)
        return None

    def _save_log(self) -> None:
        if not self._log:
            return
        out = Path.home() / "climb_logs"
        try:
            out.mkdir(exist_ok=True)
            path = out / time.strftime("climb_%Y%m%d_%H%M%S.npz")
            np.savez(path, joint_names=np.array(self._joints), reason=self._reason,
                     **{k: np.array([r[k] for r in self._log]) for k in self._log[0]})
            print(f"[climb_policy] log: {path} ({len(self._log)} ticks)")
        except OSError as e:
            print(f"[climb_policy] log not saved: {e!r}")
        self._log = []


def _microban_joints(obs) -> list[str]:
    """The joint order of Microban's observation (microban_model.JOINTS, on the robot and in the simulators)."""
    import microban_model

    joints = list(microban_model.JOINTS)
    assert len(joints) == len(obs["q"]), f"{len(joints)} joints in microban_model, {len(obs['q'])} in obs['q']"
    return joints
