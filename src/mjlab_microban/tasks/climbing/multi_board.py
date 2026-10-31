"""Multi-board climbing: several climbing_sim walls at once, each with its own planned climb, and a curriculum over
their difficulty levels in the style of the locomotion tasks' rough-terrain curriculum (mjlab: terrain_levels_vel,
TerrainEntity.update_env_origins).

Board set
---------
A JSON file (``boards/multi_board.json`` by default, or the file in $MICROBAN_BOARDS_FILE) lists the boards:

    {
      "spacing": 1.2,
      "boards": [
        {"level": 1, "seed": 0, "motion": "l1s0.npz"},
        {"level": 2, "seed": 13},
        ...
      ]
    }

Each board is a climbing_sim route ``generate_wall(level, seed)`` with its planned climb, a motion .npz of the
climbing task (rock-climbing-workshop plan_sim_wall.py --save-npz, then scripts/workshop_to_npz.py;
scripts/make_multi_board_motions.py does both for every board of the file). ``motion`` is relative to
src/mjlab_microban/motion (default: ``multi_board/l<level>s<seed>.npz``). A board's difficulty is its wall level
(``difficulty`` overrides it); the curriculum levels are the distinct difficulties, easiest first.

Scene
-----
Every env (MuJoCo world) holds all the boards, side by side along the env y axis (``spacing`` apart, board i at
y = i * spacing), in one wall entity at the world origin. An env climbs one board: its env origin is put on that
board (as the terrain curriculum puts env origins on sub-terrains), so the motion, observations and rewards,
all relative to the env origin, are those of the single-board env.

Motion
------
The boards' motions are concatenated into one motion, each padded at its end with copies of its last frame
(``segment_padding`` frames). ``time_steps`` index this concatenation, so every term that reads frames t + k of
``command.motion`` (future windows, residual actions) works unchanged, the padding standing in for the clamp at the
end of the motion, as long as k <= segment_padding. The command keeps each env on its board's segment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from mjlab_microban.tasks.climbing.mdp import motion_contact_schedule
from mjlab_microban.tasks.climbing.wall import (
    WallSpec,
    get_multi_board_spec,
    hold_positions,
    motion_wall,
    used_holds,
)

BOARDS_DIR = Path(__file__).parent / "boards"
DEFAULT_BOARDS_FILE = BOARDS_DIR / "multi_board.json"
BOARDS_FILE_ENV = "MICROBAN_BOARDS_FILE"
MOTION_DIR = Path(__file__).resolve().parents[2] / "motion"  # src/mjlab_microban/motion
DEFAULT_SPACING = 1.2  # [m] between board centres (the walls are 0.8 m wide)


# ---------------------------------------------------------------------------------------------------------
# Board set
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Board:
    wall: WallSpec
    motion_file: str
    difficulty: int  # the wall level, unless the board file says otherwise

    def default_motion_name(self) -> str:
        return f"l{self.wall.level}s{self.wall.seed}"


@dataclass(frozen=True)
class BoardSet:
    boards: tuple[Board, ...]
    spacing: float
    source: str

    @property
    def difficulties(self) -> tuple[int, ...]:
        """The distinct difficulties, easiest first: curriculum level i is difficulties[i]."""
        return tuple(sorted({b.difficulty for b in self.boards}))

    @property
    def board_levels(self) -> tuple[int, ...]:
        """Curriculum level (0 = easiest) of each board."""
        return tuple(self.difficulties.index(b.difficulty) for b in self.boards)

    @property
    def offsets(self) -> tuple[tuple[float, float, float], ...]:
        """Position of each board's origin in the world (env frame axes): along y, ``spacing`` apart."""
        return tuple((0.0, i * self.spacing, 0.0) for i in range(len(self.boards)))


def boards_file() -> Path:
    """$MICROBAN_BOARDS_FILE, or boards/multi_board.json."""
    return Path(os.environ.get(BOARDS_FILE_ENV) or DEFAULT_BOARDS_FILE)


def load_board_set(path: str | Path | None = None) -> BoardSet:
    """The boards of a board file (see the module docstring). The motions are not read here (they are checked
    when the scene / motion command is built), so a board file whose motions are not made yet still loads."""
    path = Path(path) if path is not None else boards_file()
    data = json.loads(path.read_text())
    boards = []
    for i, entry in enumerate(data["boards"]):
        if "level" not in entry or "seed" not in entry:
            raise ValueError(f"{path}: board {i} needs a 'level' and a 'seed', got {entry}")
        wall = WallSpec(level=int(entry["level"]), seed=int(entry["seed"]))
        motion = entry.get("motion", f"multi_board/l{wall.level}s{wall.seed}.npz")
        motion_path = Path(motion) if Path(motion).is_absolute() else MOTION_DIR / motion
        boards.append(Board(wall=wall, motion_file=str(motion_path), difficulty=int(entry.get("difficulty", wall.level))))
    if not boards:
        raise ValueError(f"{path}: no boards")
    return BoardSet(boards=tuple(boards), spacing=float(data.get("spacing", DEFAULT_SPACING)), source=str(path))


def check_board_motion(board: Board) -> None:
    """The board's motion exists and was planned on the board's wall."""
    if not Path(board.motion_file).exists():
        raise FileNotFoundError(
            f"{board.motion_file}: no motion for the board {board.wall}; make the board set's motions with "
            "src/mjlab_microban/scripts/make_multi_board_motions.py"
        )
    planned_on = motion_wall(board.motion_file)
    if planned_on != board.wall:
        raise ValueError(f"{board.motion_file} was planned on wall {planned_on}, the board is {board.wall}")


def make_multi_board_wall_cfg(board_set: BoardSet) -> EntityCfg:
    """All the boards in one fixed (mocap) entity "wall", at the world origin: board i translated by
    board_set.offsets[i]; the holds each board's motion uses are 5-box cups, the others 1-box blocks."""

    def spec_fn():
        boards = []
        for b in board_set.boards:
            check_board_motion(b)
            boards.append((b.wall, used_holds(b.wall, str(Path(b.motion_file).resolve()))))
        summary = ", ".join(f"{b.wall.text()} ({len(u)} used)" for b, (_, u) in zip(board_set.boards, boards))
        print(f"[wall] {len(boards)} boards from {board_set.source}: {summary}")
        return get_multi_board_spec(boards, board_set.offsets)

    return EntityCfg(spec_fn=spec_fn)


def board_hold_positions(board_set: BoardSet) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """Per board, its hold sockets relative to its own origin (env frame)."""
    return tuple(tuple(tuple(float(v) for v in p) for p in hold_positions(b.wall)) for b in board_set.boards)


def reset_wall_to_world_origin(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg = SceneEntityCfg("wall")
) -> None:
    """Event: the multi-board wall entity (a mocap body) at the world origin, unrotated, in every env (the env
    origins are on the boards, so it is not moved with them)."""
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, 3] = 1.0
    asset.write_mocap_pose_to_sim(pose, env_ids=env_ids)


# ---------------------------------------------------------------------------------------------------------
# Motion command
# ---------------------------------------------------------------------------------------------------------


class _ConcatenatedMotion:
    """The attributes of MotionLoader the terms use, for the boards' motions end to end."""

    KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")

    def __init__(self, motions: list[MotionLoader], padding: int):
        for key in self.KEYS:
            parts = []
            for m in motions:
                x = getattr(m, key)
                parts += [x, x[-1:].expand(padding, *x.shape[1:])]
            setattr(self, key, torch.cat(parts, dim=0).contiguous())
        self.time_step_total = self.joint_pos.shape[0]


class MultiBoardMotionCommand(MotionCommand):
    """MotionCommand over several boards: each env tracks the motion of its board (``board_ids``, set by
    ``set_boards``, which also moves the env origin onto the board). ``time_steps`` index the concatenated
    motion (see the module docstring); motion ends, resampling and adaptive sampling are per board."""

    cfg: MultiBoardMotionCommandCfg

    def __init__(self, cfg: MultiBoardMotionCommandCfg, env: ManagerBasedRlEnv):
        if not cfg.motion_files:
            raise ValueError("MultiBoardMotionCommandCfg needs motion_files")
        nb = len(cfg.motion_files)
        if not (len(cfg.board_offsets) == len(cfg.board_levels) == nb):
            raise ValueError("motion_files, board_offsets and board_levels need one entry per board")
        cfg.motion_file = cfg.motion_files[0]  # the base class loads it; replaced by the concatenation below
        super().__init__(cfg, env)

        motions = [MotionLoader(f, self.body_indexes, device=self.device) for f in cfg.motion_files]
        lengths = [m.time_step_total for m in motions]
        self.motion = _ConcatenatedMotion(motions, cfg.segment_padding)
        starts = np.concatenate([[0], np.cumsum(np.array(lengths) + cfg.segment_padding)[:-1]])
        self.segment_start = torch.tensor(starts, dtype=torch.long, device=self.device)  # [nb]
        self.segment_length = torch.tensor(lengths, dtype=torch.long, device=self.device)  # [nb], unpadded

        self.board_origins = torch.tensor(cfg.board_offsets, dtype=torch.float32, device=self.device)  # [nb, 3]
        self.board_levels = torch.tensor(cfg.board_levels, dtype=torch.long, device=self.device)  # [nb]
        self.num_levels = int(self.board_levels.max()) + 1
        if set(cfg.board_levels) != set(range(self.num_levels)):
            raise ValueError(f"board_levels must cover 0..{self.num_levels - 1}, got {cfg.board_levels}")
        # Boards of each level, padded: level_boards[l, :level_count[l]].
        per_level = [[b for b in range(nb) if cfg.board_levels[b] == lv] for lv in range(self.num_levels)]
        self.level_count = torch.tensor([len(p) for p in per_level], device=self.device)
        width = max(len(p) for p in per_level)
        self.level_boards = torch.tensor([p + [p[0]] * (width - len(p)) for p in per_level], device=self.device)

        # Adaptive sampling, per board: bins of 1 s of its motion.
        steps_per_s = 1.0 / env.step_dt
        self.board_bin_count = (self.segment_length.float() // steps_per_s).long() + 1  # [nb]
        self.bin_count = int(self.board_bin_count.max())
        self.bin_valid = torch.arange(self.bin_count, device=self.device)[None] < self.board_bin_count[:, None]
        self.bin_failed_count = torch.zeros(nb, self.bin_count, device=self.device)
        self._current_bin_failed = torch.zeros(nb, self.bin_count, device=self.device)

        self.board_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["board"] = torch.zeros(self.num_envs, device=self.device)
        # Without a curriculum (e.g. play): every env on a random board. The curriculum term overrides this.
        all_envs = torch.arange(self.num_envs, device=self.device)
        self.set_boards(all_envs, torch.randint(0, nb, (self.num_envs,), device=self.device))
        self.time_steps[:] = self.segment_start[self.board_ids]

    @property
    def num_boards(self) -> int:
        return len(self.cfg.motion_files)

    # Boards.

    def sample_boards(self, levels: torch.Tensor) -> torch.Tensor:
        """A random board of each given curriculum level."""
        k = (torch.rand(levels.shape, device=self.device) * self.level_count[levels]).long()
        return self.level_boards[levels, k]

    def set_boards(self, env_ids: torch.Tensor, board_ids: torch.Tensor) -> None:
        """Put the envs on these boards (their env origins on the boards); their motion is resampled on the next
        reset of the command (the curriculum runs just before it)."""
        self.board_ids[env_ids] = board_ids
        self._env.scene.env_origins[env_ids] = self.board_origins[board_ids]

    def _segment_of(self, t: torch.Tensor) -> torch.Tensor:
        """The board whose (padded) segment holds the frames t of the concatenated motion."""
        return torch.searchsorted(self.segment_start, t, right=True) - 1

    # Sampling.

    def _bin_probabilities(self) -> torch.Tensor:
        """[nb, bins] adaptive sampling probabilities of each board's bins (0 past its last bin), as
        MotionCommand._adaptive_sampling computes them for one motion."""
        probs = torch.zeros_like(self.bin_failed_count)
        for b in range(self.num_boards):
            n = int(self.board_bin_count[b])
            p = self.bin_failed_count[b, :n] + self.cfg.adaptive_uniform_ratio / float(n)
            p = torch.nn.functional.pad(p.view(1, 1, -1), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate")
            p = torch.nn.functional.conv1d(p, self.kernel.view(1, 1, -1)).view(-1)
            probs[b, :n] = p / p.sum()
        return probs

    def _record_failures(self, env_ids: torch.Tensor) -> None:
        """Bins (of the board each env was on: the one its time step is in) where the episodes failed."""
        failed = self._env.termination_manager.terminated[env_ids]
        if not torch.any(failed):
            return
        t = self.time_steps[env_ids][failed]
        board = self._segment_of(t)
        local = (t - self.segment_start[board]).clamp(min=0)
        n = self.board_bin_count[board]
        bins = torch.minimum(local * n // self.segment_length[board].clamp(min=1), n - 1)
        self._current_bin_failed.zero_()
        self._current_bin_failed.index_put_((board, bins), torch.ones_like(bins, dtype=torch.float), accumulate=True)

    def _sample_local_steps(self, env_ids: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
        length = self.segment_length[board]
        if self.cfg.sampling_mode == "start":
            return torch.zeros_like(length)
        if self.cfg.sampling_mode == "uniform":
            self.metrics["sampling_entropy"][env_ids] = 1.0
            return (torch.rand(length.shape, device=self.device) * length).long().clamp(max=length - 1)
        assert self.cfg.sampling_mode == "adaptive"
        self._record_failures(env_ids)
        probs = self._bin_probabilities()
        bins = torch.multinomial(probs[board], 1).squeeze(-1)
        n = self.board_bin_count[board]
        u = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        local = ((bins + u) / n * (length - 1)).long()

        valid = self.bin_valid.float()
        entropy = -(probs * (probs + 1e-12).log() * valid).sum(dim=-1)
        entropy = entropy / torch.log(self.board_bin_count.float()).clamp(min=1e-12)
        entropy = torch.where(self.board_bin_count > 1, entropy, torch.ones_like(entropy))
        pmax, imax = probs.max(dim=-1)
        self.metrics["sampling_entropy"][env_ids] = entropy[board]
        self.metrics["sampling_top1_prob"][env_ids] = pmax[board]
        self.metrics["sampling_top1_bin"][env_ids] = imax[board].float() / n.float()
        return local

    def _resample_command(self, env_ids: torch.Tensor):
        # MotionCommand._resample_command, with the time step sampled on each env's board.
        board = self.board_ids[env_ids]
        self.time_steps[env_ids] = self.segment_start[board] + self._sample_local_steps(env_ids, board)
        self.metrics["board"][env_ids] = board.float()

        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        ranges = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand[:, 0:3]
        root_ori = quat_mul(quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5]), root_ori)
        ranges = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand[:, :3]
        root_ang_vel += rand[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids]
        joint_pos += sample_uniform(
            lower=self.cfg.joint_position_range[0],
            upper=self.cfg.joint_position_range[1],
            size=joint_pos.shape,
            device=joint_pos.device,  # type: ignore
        )
        self._write_reference_state_to_sim(
            env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
        )

    def _update_command(self):
        self.time_steps += 1
        end = self.segment_start[self.board_ids] + self.segment_length[self.board_ids]
        env_ids = torch.where(self.time_steps >= end)[0]
        if env_ids.numel() > 0:
            self._resample_command(env_ids)

        self.update_relative_body_poses()

        if self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()

    def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
        """Frame ``frame`` of each env's own board motion (clamped to its last frame), no perturbation."""
        board = self.board_ids[env_ids]
        local = torch.clamp(torch.full_like(board, int(frame)), max=self.segment_length[board] - 1)
        self.time_steps[env_ids] = self.segment_start[board] + local
        self._write_reference_state_to_sim(
            env_ids,
            self.body_pos_w[env_ids, 0],
            self.body_quat_w[env_ids, 0],
            self.body_lin_vel_w[env_ids, 0],
            self.body_ang_vel_w[env_ids, 0],
            self.joint_pos[env_ids],
            self.joint_vel[env_ids],
        )


@dataclass(kw_only=True)
class MultiBoardMotionCommandCfg(MotionCommandCfg):
    """motion_file is set to the first board's motion (the train / play scripts check that it exists)."""

    motion_file: str = ""
    motion_files: tuple[str, ...] = ()
    board_offsets: tuple[tuple[float, float, float], ...] = ()  # board origins in the world
    board_levels: tuple[int, ...] = ()  # curriculum level of each board, 0 = easiest
    level_names: tuple[str, ...] = ()  # name of each curriculum level, for the logs
    # Copies of each motion's last frame after it: the furthest frame ahead any term reads (t + k, k <= padding).
    segment_padding: int = 100

    def build(self, env: ManagerBasedRlEnv) -> MultiBoardMotionCommand:
        return MultiBoardMotionCommand(self, env)


# ---------------------------------------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------------------------------------


class board_levels:
    """Board curriculum, as the rough-terrain curriculum of the locomotion tasks (terrain_levels_vel): each env has
    a level (0 = the easiest boards); at its reset, an env that lasted at least ``up_fraction`` of the maximum
    episode length (it tracked its climb without failing) moves one level up, one that lasted less than
    ``down_fraction`` of it moves one level down. An env moving up from the top level goes to a random level (as
    TerrainEntity.update_env_origins). Its next board is a random board of its level.

    The envs start at a random level in 0..max_init_level.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        command = env.command_manager.get_term(cfg.params["command_name"])
        assert isinstance(command, MultiBoardMotionCommand), "board_levels needs a MultiBoardMotionCommand"
        max_init = min(int(cfg.params.get("max_init_level", 0)), command.num_levels - 1)
        self.levels = torch.randint(0, max_init + 1, (env.num_envs,), device=env.device)
        command.set_boards(torch.arange(env.num_envs, device=env.device), command.sample_boards(self.levels))
        names = command.cfg.level_names
        self.level_names = names if len(names) == command.num_levels else tuple(map(str, range(command.num_levels)))

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        command_name: str,
        up_fraction: float = 0.9,
        down_fraction: float = 0.5,
        max_init_level: int = 0,
    ) -> dict[str, torch.Tensor]:
        del max_init_level
        command = cast(MultiBoardMotionCommand, env.command_manager.get_term(command_name))
        steps = env.episode_length_buf[env_ids]
        ran = steps > 0  # not the first reset of the env (nothing to judge)
        fraction = steps.float() / env.max_episode_length
        move_up = ran & (fraction >= up_fraction)
        move_down = ran & (fraction < down_fraction) & ~move_up

        levels = self.levels[env_ids] + move_up.long() - move_down.long()
        levels = torch.where(
            levels >= command.num_levels,
            torch.randint_like(levels, command.num_levels),
            levels.clamp(min=0),
        )
        self.levels[env_ids] = levels
        command.set_boards(env_ids, command.sample_boards(levels))

        all_levels = self.levels.float()
        result = {"mean": all_levels.mean(), "max": all_levels.max()}
        for lv, name in enumerate(self.level_names):
            result[f"frac_level_{name}"] = (self.levels == lv).float().mean()
        return result


# ---------------------------------------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------------------------------------


class multi_board_motion_contact_schedule(motion_contact_schedule):
    """motion_contact_schedule over the boards of a MultiBoardMotionCommand: each board's (padded) motion segment
    against its own holds (``board_hold_positions``: per board, its hold sockets relative to its origin)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        command = env.command_manager.get_term(cfg.params["command_name"])
        assert isinstance(command, MultiBoardMotionCommand)
        ball_ref = self._ball_reference(cfg, env, command)  # [T, N, 3], each board's frame
        threshold = cfg.params.get("threshold", 0.003)
        holds_per_board = cfg.params["board_hold_positions"]
        assert len(holds_per_board) == command.num_boards
        ends = list(command.segment_start[1:].tolist()) + [command.motion.time_step_total]
        contact, next_frame, next_pos = [], [], []
        for b, (start, end) in enumerate(zip(command.segment_start.tolist(), ends, strict=True)):
            holds = torch.tensor(holds_per_board[b], dtype=torch.float32, device=env.device)
            c, f, p = self._schedule(ball_ref[start:end], holds, threshold)
            contact.append(c)
            next_frame.append(f + start)
            next_pos.append(p)
        self.contact = torch.cat(contact)
        self.next_frame = torch.cat(next_frame)
        self.next_pos = torch.cat(next_pos)
        self.future = torch.tensor(cfg.params["offsets"], device=env.device)
        assert int(self.future.max()) <= command.cfg.segment_padding
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        site_names: tuple[str, ...],
        offsets: tuple[int, ...],
        board_hold_positions: tuple,
        threshold: float = 0.003,
        max_time: float = 1.0,
        asset_name: str = "robot",
    ) -> torch.Tensor:
        del board_hold_positions
        return super().__call__(env, command_name, site_names, offsets, (), threshold, max_time, asset_name)


__all__ = [
    "Board",
    "BoardSet",
    "MultiBoardMotionCommand",
    "MultiBoardMotionCommandCfg",
    "board_hold_positions",
    "board_levels",
    "boards_file",
    "check_board_motion",
    "load_board_set",
    "make_multi_board_wall_cfg",
    "multi_board_motion_contact_schedule",
    "reset_wall_to_world_origin",
]

