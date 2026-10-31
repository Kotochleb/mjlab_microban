#!/usr/bin/env bash
# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Train the Microban climbing task: tracking of a climb planned by rock-climbing-workshop on a climbing_sim wall.
#
# Usage:
#   src/mjlab_microban/scripts/train_climbing.sh [--policy mlp|conv1d] [motion.npz|planner.npz] [extra train args...]
#
# --policy selects the policy's view of the future reference (default: mlp, or $POLICY):
#   mlp     the TASK as is: the future reference frames are flat inputs of the MLP;
#   conv1d  TASK's Conv1d variant (TASK-Conv1d): dense 1 s windows of the future reference, encoded by 1D
#           convolutions before the MLP (tasks/climbing/microban_climbing_conv1d_env_cfg.py).
#
# Without a file, the level 1 (seed 0) climb src/mjlab_microban/motion/l1s0.npz is used.
# The wall (level / seed) and the holds used (5-box cups, the others 1-box blocks) are read by the env from
# the motion's "wall" and "used_holds" (tasks/climbing/wall.py), so the motion file is all there is to give.
# A planner output (rock-climbing-workshop/scripts/plan_sim_wall.py --save-npz, with q: N x 26) is first
# converted to a motion with workshop_to_npz.py: <name>_q.npz -> <name>.npz, other names -> <name>_motion.npz.
# Environment variables:
#   TASK       Task id (default: Mjlab-Climbing-Microban; also Mjlab-Climbing-Microban-Residual,
#              Mjlab-Climbing-Microban-No-State-Estimation, Mjlab-Climbing-Microban-Residual-No-State-Estimation)
#   NUM_ENVS   Number of parallel environments (default: 4096)
#   VIDEO      Record training videos (default: True)
#   POLICY     Default of --policy (mlp)
#
# Examples:
#   src/mjlab_microban/scripts/train_climbing.sh
#   src/mjlab_microban/scripts/train_climbing.sh src/mjlab_microban/motion/l1s0_q.npz --agent.max-iterations 10000
#   TASK=Mjlab-Climbing-Microban-Residual src/mjlab_microban/scripts/train_climbing.sh
#   src/mjlab_microban/scripts/train_climbing.sh --policy conv1d src/mjlab_microban/motion/l1s0.npz

set -euo pipefail

TASK="${TASK:-Mjlab-Climbing-Microban}"
NUM_ENVS="${NUM_ENVS:-4096}"
VIDEO="${VIDEO:-True}"
POLICY="${POLICY:-mlp}"

# Take --policy out of the arguments (anywhere on the line); the rest goes to the motion file / train.
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) [[ $# -ge 2 ]] || { echo "--policy needs a value: mlp or conv1d" >&2; exit 1; }; POLICY="$2"; shift 2 ;;
    --policy=*) POLICY="${1#--policy=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

case "$POLICY" in
  mlp) ;;
  conv1d)
    case "$TASK" in
      Mjlab-Climbing-Microban|Mjlab-Climbing-Microban-Residual) TASK="$TASK-Conv1d" ;;
      *-Conv1d) ;;
      *) echo "No Conv1d variant of $TASK (there are Mjlab-Climbing-Microban[-Residual]-Conv1d)" >&2; exit 1 ;;
    esac
    ;;
  *) echo "Unknown --policy $POLICY (expected mlp or conv1d)" >&2; exit 1 ;;
esac

export MUJOCO_GL=egl

# Run from the repository root so logs end up in logs/rsl_rl/.
REPO_ROOT="$(cd "$(dirname "$(realpath "$0")")/../../.." && pwd)"

if [[ $# -ge 1 && "$1" != -* ]]; then
  MOTION_FILE="$(realpath "$1")"
  shift
else
  MOTION_FILE="$REPO_ROOT/src/mjlab_microban/motion/l1s0.npz"
fi

cd "$REPO_ROOT"

if [[ ! -f "$MOTION_FILE" ]]; then
  echo "Motion file not found: $MOTION_FILE" >&2
  exit 1
fi
if [[ "$MOTION_FILE" != *.npz ]]; then
  echo "Unsupported motion file (expected .npz): $MOTION_FILE" >&2
  exit 1
fi

# The kind of .npz: a planner output has q, an env motion has body_pos_w.
npz_kind() {
  uv run python -c "
import sys, numpy as np
f = np.load(sys.argv[1]).files
print('motion' if 'body_pos_w' in f else 'planner' if 'q' in f else 'unknown')
" "$1"
}

KIND="$(npz_kind "$MOTION_FILE")"
if [[ "$KIND" == "planner" ]]; then
  PLAN_FILE="$MOTION_FILE"
  if [[ "$PLAN_FILE" == *_q.npz ]]; then
    MOTION_FILE="${PLAN_FILE%_q.npz}.npz"
  else
    MOTION_FILE="${PLAN_FILE%.npz}_motion.npz"
  fi
  echo "[INFO] Converting planner output $PLAN_FILE -> $MOTION_FILE"
  uv run python src/mjlab_microban/scripts/workshop_to_npz.py \
    --input-file "$PLAN_FILE" \
    --output-file "$MOTION_FILE"
elif [[ "$KIND" != "motion" ]]; then
  echo "Not a climbing motion (body_pos_w) nor a planner output (q): $MOTION_FILE" >&2
  exit 1
fi

# The env builds its wall from the motion: it needs the motion's wall and used holds.
WALL="$(uv run python -c "
import sys, numpy as np
d = np.load(sys.argv[1])
missing = [k for k in ('wall', 'used_holds') if k not in d.files]
if missing:
    sys.exit(f'{sys.argv[1]} has no {missing}: re-plan it with rock-climbing-workshop/scripts/plan_sim_wall.py '
             '--save-npz, then convert it with src/mjlab_microban/scripts/workshop_to_npz.py')
print(f\"{d['wall']} ({len(d['used_holds'])} holds used)\")
" "$MOTION_FILE")"

echo "[INFO] Training $TASK ($POLICY policy) with $NUM_ENVS envs on $MOTION_FILE, wall $WALL"
exec uv run train "$TASK" \
  --env.scene.num-envs "$NUM_ENVS" \
  --env.commands.motion.motion-file "$MOTION_FILE" \
  --video "$VIDEO" \
  "$@"
