"""Make the motions of a multi-board board file (tasks/climbing/multi_board.py): for each board, plan its climb with
rock-climbing-workshop's planner and trajectory optimiser (scripts/plan_board.py, run with the workshop's python:
plan_sim_wall.py's planning, --level L --seed S or --wall <this project's wall JSON>), then convert it into the
climbing task's motion (scripts/workshop_to_npz.py), at the board's ``motion`` path.

A level 2+ plan usually stops before the top of the wall (the planner gets stuck), and the optimiser cannot do
some moves: the climb is cut before the first move it cannot do, and the part that works is trained on. A board
with no move left is a failure: reported, its motion files removed (failures are never in the training set);
take it out of the board file or give it another seed.

The boards are planned in parallel (--jobs, one planner process each), their conversions run one at a time.
Boards whose motion already exists are skipped (--force: redo them).

Usage:
    uv run python src/mjlab_microban/scripts/make_multi_board_motions.py
    uv run python src/mjlab_microban/scripts/make_multi_board_motions.py --boards-file my_boards.json --force
    uv run python src/mjlab_microban/scripts/make_multi_board_motions.py --force -j 8   # 8 boards at a time
"""

import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mjlab_microban.tasks.climbing.multi_board import boards_file, check_board_motion, load_board_set
from mjlab_microban.tasks.climbing.wall import wall_file

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSHOP_TO_NPZ = Path(__file__).resolve().parent / "workshop_to_npz.py"
PLAN_BOARD = Path(__file__).resolve().parent / "plan_board.py"  # runs with the workshop's python, in its checkout
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boards-file", type=Path, default=None, help=f"board file (default: {boards_file()})")
    p.add_argument("--workshop-dir", type=Path, default=REPO_ROOT.parent / "rock-climbing-workshop",
                   help="rock-climbing-workshop checkout (its .venv runs the planner)")
    p.add_argument("--force", action="store_true", help="redo the motions that exist")
    p.add_argument("--device", default="cuda:0", help="workshop_to_npz.py --device")
    p.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1,
                   help="boards planned in parallel (default: the number of cores); the conversions (GPU) run one "
                        "at a time")
    return p.parse_args()  # fmt: skip


def planner_cmd(workshop: Path) -> list[str]:
    python = workshop / ".venv" / "bin" / "python"
    return [str(python)] if python.exists() else ["uv", "run", "python"]


def main() -> None:
    args = parse_args()
    board_set = load_board_set(args.boards_file)
    workshop = args.workshop_dir.resolve()
    if not (workshop / "scripts" / "plan_sim_wall.py").exists():
        raise SystemExit(f"[ERROR] no rock-climbing-workshop checkout at {workshop} (--workshop-dir)")

    todo = []
    for board in board_set.boards:
        if Path(board.motion_file).exists() and not args.force:
            print(f"[INFO] {board.wall}: {board.motion_file} exists, skipped")
        else:
            todo.append(board)
    jobs = max(1, min(args.jobs, len(todo)))
    # Each planner gets its share of the cores (its numpy / BLAS threads), so the jobs do not oversubscribe them.
    threads = str(max(1, (os.cpu_count() or 1) // jobs))
    planner_env = {**os.environ, **{v: threads for v in THREAD_VARS}}
    print(f"[INFO] {len(todo)} boards to make, {jobs} in parallel ({threads} threads each)")

    convert_lock = threading.Lock()  # one conversion at a time: they share the GPU
    print_lock = threading.Lock()

    def log(*lines: str) -> None:
        with print_lock:
            print("\n".join(lines), flush=True)

    def fail(board, why: str) -> bool:
        """A failed board: reported, and its stale motion files removed (a failure is never in the training set)."""
        log(f"[ERROR] {board.wall}: {why}")
        motion = Path(board.motion_file)
        for f in (motion, motion.with_name(motion.stem + "_q.npz"), motion.with_name(motion.stem + "_plan.json")):
            f.unlink(missing_ok=True)
        return False

    def make(board) -> bool:
        motion = Path(board.motion_file)
        plan = motion.with_name(motion.stem + "_q.npz")
        contact_plan = motion.with_name(motion.stem + "_plan.json")
        plan.parent.mkdir(parents=True, exist_ok=True)
        if board.wall.name:  # a wall JSON file of this project, given to the planner by its absolute path
            wall_args = ["--wall", str(wall_file(board.wall.name).resolve()), "--level", str(board.difficulty)]
            convert_wall_args = ["--wall", board.wall.name]
        else:
            wall_args = ["--level", str(board.wall.level), "--seed", str(board.wall.seed)]
            convert_wall_args = wall_args
        plan.unlink(missing_ok=True)  # the planner writes no .npz when it finds no move: no stale one
        log(f"[INFO] {board.wall}: planning -> {plan}")
        r = subprocess.run(
            [*planner_cmd(workshop), str(PLAN_BOARD), *wall_args, "--wall-text", board.wall.text(),
             "--save", str(contact_plan), "--save-npz", str(plan)],
            cwd=workshop, capture_output=True, text=True, env=planner_env, check=False,
        )  # fmt: skip
        summary = [line for line in r.stdout.splitlines() if line.startswith(("trajectory", "cut ")) or " moves in " in line]
        log(f"[INFO] {board.wall}: planned", *(f"        {line}" for line in summary))
        if r.returncode != 0 or not plan.exists():
            return fail(board, f"planning failed: {(r.stderr.strip().splitlines() or ['no move from the start'])[-1]}")
        with convert_lock:
            log(f"[INFO] {board.wall}: converting -> {motion}")
            r = subprocess.run(
                [sys.executable, str(WORKSHOP_TO_NPZ), "--input-file", str(plan), "--output-file", str(motion),
                 "--contact-plan", str(contact_plan), *convert_wall_args, "--device", args.device],
                cwd=REPO_ROOT, capture_output=True, text=True, check=False,
            )  # fmt: skip
        if r.returncode != 0:
            return fail(board, "conversion failed:\n" + "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:]))
        check_board_motion(board)
        log(f"[INFO] {board.wall}: done -> {motion}")
        return True

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        ok = list(pool.map(make, todo))
    failed = [b for b, good in zip(todo, ok, strict=True) if not good]

    if failed:
        raise SystemExit(
            f"[ERROR] no motion for {', '.join(str(b.wall) for b in failed)}: remove these boards from "
            f"{board_set.source} (failures are not trained on) or try other seeds"
        )
    print(f"[INFO] all {len(board_set.boards)} boards of {board_set.source} have their motion")


if __name__ == "__main__":
    main()
