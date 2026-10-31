"""Plan a board's climb with rock-climbing-workshop's planner and trajectory optimiser, keeping the part that works.

Run with rock-climbing-workshop's python, in its checkout (make_multi_board_motions.py does):

    cd ../rock-climbing-workshop && .venv/bin/python <this file> --level 2 --seed 5 --save plan.json --save-npz q.npz

As scripts/plan_sim_wall.py --save --save-npz (its A* contact plan, then rcr.kinematics' move-by-move IK), except:

- A move the optimiser cannot do (a held ball MAX_ERROR from its socket: ValueError) ends the climb: the plan is cut
  just before it and the moves before it are kept, rather than the whole board failing. Failed moves are never
  in the output.
- The .npz has the ``used_holds`` (planner ids r<row>_c<col>, of the kept stances) and the ``wall``
  (mjlab_microban's WallSpec text: "level:seed", or --wall-text) next to ``q``, as the climbing task reads them,
  and --save writes the kept contact plan.

Fails (exit 1, no output written) when not one move is left.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "scripts"))  # the workshop's plan_sim_wall.py helpers

from climbing_sim import Wall  # noqa: E402
from climbing_sim.routes import generate_wall  # noqa: E402
from plan_sim_wall import to_wall_model  # noqa: E402
from rcr import io, kinematics  # noqa: E402
from rcr.kinematics import ik  # noqa: E402
from rcr.planner import astar, feasible, plan  # noqa: E402
from rcr.types import MICROBAN_JOINTS, ContactPlan, Trajectory  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--level", type=int, default=1, choices=(1, 2, 3, 4))
    p.add_argument("--seed", type=int, help="random if not given")
    p.add_argument("--wall", type=Path, help="a wall JSON file (climbing_sim form) instead of generate_wall")
    p.add_argument("--wall-text", help="the .npz's 'wall' (default: 'level:seed', or --wall)")
    p.add_argument("--save", type=Path, required=True, help="the kept contact plan (JSON)")
    p.add_argument("--save-npz", type=Path, required=True, help="the kept trajectory (q: N x 26, every ik.DT)")
    return p.parse_args()


def optimise_prefix(wall, contact_plan: ContactPlan) -> tuple[Trajectory, int, str | None]:
    """kinematics.optimise, move by move, stopping before the first move it cannot do:
    (trajectory of the moves done, number of moves done, why the next one failed or None)."""
    kinematics.validate_plan(wall, contact_plan)
    qs = [ik.start_pose(wall, contact_plan.stances[0])]
    done, why = 0, None
    for a, b in zip(contact_plan.stances, contact_plan.stances[1:], strict=False):
        try:
            qs += ik.track_move(wall, a, b, qs[-1])[0]
        except ValueError as e:
            why = str(e)
            break
        done += 1
    joints = [ik.model.idx_qs[ik.model.getJointId(name)] for name in MICROBAN_JOINTS]
    positions = [[float(q[j]) for j in joints] for q in qs]
    trajectory = Trajectory(
        joint_names=list(MICROBAN_JOINTS),
        times=[round(k * ik.DT, 9) for k in range(len(qs))],
        positions=positions,
        planned_positions=[row.copy() for row in positions],
        trunk_poses=[[float(v) for v in q[[0, 1, 2, 6, 3, 4, 5]]] for q in qs],  # Pinocchio (x,y,z,w) -> (w,x,y,z)
    )
    return trajectory, done, why


def main() -> None:
    args = parse_args()
    seed = random.randrange(10_000) if args.seed is None else args.seed
    sim_wall = Wall.load(str(args.wall)) if args.wall else generate_wall(args.level, seed)
    wall = to_wall_model(sim_wall, args.level, seed)
    wall_text = args.wall_text or (str(args.wall) if args.wall else f"{args.level}:{seed}")
    print(f"wall: {wall_text}, {len(wall.holds)} holds")

    clock = time.perf_counter()
    contact_plan = plan(wall, feasible, HEURISTIC_WEIGHT=astar.WEIGHT)
    moves = len(contact_plan.stances) - 1
    z = {h.id: h.position[2] for h in wall.holds}
    hand = max(z[contact_plan.stances[-1].contacts[h]] for h in ("right_hand", "left_hand"))
    top = max(z.values())
    print(
        f"{moves} moves in {time.perf_counter() - clock:.1f} s; highest hand {hand:.3f} m, top {top:.3f} m: "
        f"{'reached' if hand == top else 'NOT reached'}"
    )
    if moves < 1:
        raise SystemExit("no move from the start stance")

    clock = time.perf_counter()
    trajectory, done, why = optimise_prefix(wall, contact_plan)
    if why:
        print(f"cut before move {done + 1}/{moves}, which the optimiser cannot do: {why}")
    if done < 1:
        raise SystemExit(f"the optimiser cannot do the first move: {why}")
    kept = ContactPlan(wall_id=contact_plan.wall_id, stances=contact_plan.stances[: done + 1])
    print(f"trajectory: {trajectory.times[-1]:.1f} s of motion, {done} moves, solved in {time.perf_counter() - clock:.1f} s")

    args.save.parent.mkdir(parents=True, exist_ok=True)
    io.save_contact_plan(kept, args.save)
    args.save_npz.parent.mkdir(parents=True, exist_ok=True)
    kinematics.convert_trajectory_to_npz(trajectory, args.save_npz)
    used = sorted({h for s in kept.stances for h in s.contacts.values() if h not in (None, "ground")})
    data = dict(np.load(args.save_npz))
    np.savez(args.save_npz, **data, used_holds=np.array(used), wall=np.array(wall_text))
    print(f"saved {args.save_npz} ({len(used)} holds used), {args.save}")


if __name__ == "__main__":
    main()
