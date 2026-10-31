"""Make the motions of a multi-board board file (tasks/climbing/multi_board.py): for each board, plan its climb with
rock-climbing-workshop (scripts/plan_sim_wall.py --level L --seed S --save-npz), then convert it into the
climbing task's motion (scripts/workshop_to_npz.py), at the board's ``motion`` path.

Boards whose motion already exists are skipped (--force: redo them). A level 2+ plan usually stops before the
top of the wall (the planner gets stuck): its partial climb is used as is. A board whose trajectory cannot be
optimised (the planner's IK fails on a move) is reported; give it another seed in the board file.

Usage:
    uv run python src/mjlab_microban/scripts/make_multi_board_motions.py
    uv run python src/mjlab_microban/scripts/make_multi_board_motions.py --boards-file my_boards.json --force
"""

import argparse
import subprocess
import sys
from pathlib import Path

from mjlab_microban.tasks.climbing.multi_board import boards_file, check_board_motion, load_board_set

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSHOP_TO_NPZ = Path(__file__).resolve().parent / "workshop_to_npz.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boards-file", type=Path, default=None, help=f"board file (default: {boards_file()})")
    p.add_argument("--workshop-dir", type=Path, default=REPO_ROOT.parent / "rock-climbing-workshop",
                   help="rock-climbing-workshop checkout (its .venv runs the planner)")
    p.add_argument("--force", action="store_true", help="redo the motions that exist")
    p.add_argument("--device", default="cuda:0", help="workshop_to_npz.py --device")
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

    failed = []
    for board in board_set.boards:
        motion = Path(board.motion_file)
        if motion.exists() and not args.force:
            print(f"[INFO] {board.wall}: {motion} exists, skipped")
            continue
        plan = motion.with_name(motion.stem + "_q.npz")
        plan.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] {board.wall}: planning -> {plan}")
        r = subprocess.run(
            [*planner_cmd(workshop), "scripts/plan_sim_wall.py", "--level", str(board.wall.level),
             "--seed", str(board.wall.seed), "--save-npz", str(plan)],
            cwd=workshop, capture_output=True, text=True,
        )  # fmt: skip
        summary = [line for line in r.stdout.splitlines() if " moves in " in line or line.startswith("trajectory")]
        print("\n".join(f"        {line}" for line in summary))
        if r.returncode != 0 or not plan.exists():
            error = (r.stderr.strip().splitlines() or ["?"])[-1]
            print(f"[ERROR] {board.wall}: planning failed: {error}")
            failed.append(board)
            continue
        print(f"[INFO] {board.wall}: converting -> {motion}")
        r = subprocess.run(
            [sys.executable, str(WORKSHOP_TO_NPZ), "--input-file", str(plan), "--output-file", str(motion),
             "--device", args.device],
            cwd=REPO_ROOT,
        )  # fmt: skip
        if r.returncode != 0:
            failed.append(board)
            continue
        check_board_motion(board)

    if failed:
        raise SystemExit(f"[ERROR] no motion for {', '.join(str(b.wall) for b in failed)}: try other seeds")
    print(f"[INFO] all {len(board_set.boards)} boards of {board_set.source} have their motion")


if __name__ == "__main__":
    main()
