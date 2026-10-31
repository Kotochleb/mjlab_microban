#!/usr/bin/env bash
# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Train the Microban climbing task: tracking of a climb planned by rock-climbing-workshop on a climbing_sim wall.
#
# Usage:
#   src/mjlab_microban/scripts/train_climbing.sh [variant flags] [motion.npz|planner.npz] [extra train args...]
#
# All climbing tasks use residual actions (target = ref + 0.25 * effort_limit / kp * a) and no state estimation.
#
# Variant flags (anywhere on the line; they pick the task id Mjlab-Climbing-Microban[-Conv1d|-Conv1d-History]):
#   --policy P              the policy's view of the reference / proprioception (default: mlp, or $POLICY):
#       mlp             flat inputs of the MLP (future reference frames, 5-step proprioceptive history);
#       conv1d          dense 1 s windows of the future reference, encoded by 1D convolutions before the MLP
#                       (tasks/climbing/microban_climbing_conv1d_env_cfg.py);
#       conv1d-history  as conv1d, plus a 0.5 s history of the actor's proprioception, also encoded by 1D
#                       convolutions (tasks/climbing/microban_climbing_conv1d_history_env_cfg.py).
#   --conv1d, --conv1d-history   shorthands for --policy conv1d / --policy conv1d-history
#   --multi-board           train on several boards at once with a curriculum over their levels
#                           (Mjlab-Climbing-Microban-Multi-Board[-Conv1d|-Conv1d-History]): the boards and their
#                           motions come from a board file, not from a motion file argument
#   --boards-file F         the board file of --multi-board (default: tasks/climbing/boards/multi_board.json);
#                           make its motions first with src/mjlab_microban/scripts/make_multi_board_motions.py
# Run flags:
#   --num-envs N            number of parallel environments (default: 4096, or $NUM_ENVS)
#   --no-video              do not record training videos
#   --dry-run               print the train command instead of running it
#   -h, --help              this help
#
# Without a file, the level 1 (seed 0) climb src/mjlab_microban/motion/l1s0.npz is used.
# The wall (level / seed) and the holds used (5-box cups, the others 1-box blocks) are read by the env from
# the motion's "wall" and "used_holds" (tasks/climbing/wall.py), so the motion file is all there is to give.
# A planner output (rock-climbing-workshop/scripts/plan_sim_wall.py --save-npz, with q: N x 26) is first
# converted to a motion with workshop_to_npz.py: <name>_q.npz -> <name>.npz, other names -> <name>_motion.npz.
# Environment variables:
#   TASK       Task id, used as is instead of --policy
#   NUM_ENVS   Default of --num-envs (4096)
#   VIDEO      Record training videos (default: True)
#   POLICY     Default of --policy (mlp)
#
# Examples:
#   src/mjlab_microban/scripts/train_climbing.sh
#   src/mjlab_microban/scripts/train_climbing.sh src/mjlab_microban/motion/l1s0_q.npz --agent.max-iterations 10000
#   src/mjlab_microban/scripts/train_climbing.sh --conv1d-history src/mjlab_microban/motion/l1s0.npz
#   src/mjlab_microban/scripts/train_climbing.sh --conv1d --num-envs 2048 --no-video
#   src/mjlab_microban/scripts/train_climbing.sh --multi-board --conv1d-history

set -euo pipefail

usage() { sed -n '/^# Usage:/,/^# Examples:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; }

NUM_ENVS="${NUM_ENVS:-4096}"
VIDEO="${VIDEO:-True}"
POLICY="${POLICY:-mlp}"
DRY_RUN=false
MULTI_BOARD=false
POLICY_FLAG=false  # --policy given explicitly

need_value() { [[ $2 -ge 2 ]] || { echo "$1 needs a value" >&2; exit 1; }; }

# Take the script's flags out of the arguments (anywhere on the line); the rest goes to the motion file / train.
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --policy) need_value "$1" $#; POLICY="$2"; POLICY_FLAG=true; shift 2 ;;
    --policy=*) POLICY="${1#--policy=}"; POLICY_FLAG=true; shift ;;
    --conv1d) POLICY=conv1d; POLICY_FLAG=true; shift ;;
    --conv1d-history) POLICY=conv1d-history; POLICY_FLAG=true; shift ;;
    --num-envs) need_value "$1" $#; NUM_ENVS="$2"; shift 2 ;;
    --num-envs=*) NUM_ENVS="${1#--num-envs=}"; shift ;;
    --no-video) VIDEO=False; shift ;;
    --multi-board) MULTI_BOARD=true; shift ;;
    --boards-file) need_value "$1" $#; MULTI_BOARD=true; export MICROBAN_BOARDS_FILE="$(realpath "$2")"; shift 2 ;;
    --boards-file=*) MULTI_BOARD=true; export MICROBAN_BOARDS_FILE="$(realpath "${1#--boards-file=}")"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

[[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]] || { echo "--num-envs must be a positive integer, got $NUM_ENVS" >&2; exit 1; }

case "$POLICY" in
  mlp) POLICY_SUFFIX="" ;;
  conv1d) POLICY_SUFFIX="-Conv1d" ;;
  conv1d-history) POLICY_SUFFIX="-Conv1d-History" ;;
  *) echo "Unknown --policy $POLICY (expected mlp, conv1d or conv1d-history)" >&2; exit 1 ;;
esac

if [[ -n "${TASK:-}" ]]; then
  if $POLICY_FLAG; then
    echo "TASK=$TASK is set: drop it or --policy" >&2
    exit 1
  fi
else
  if $MULTI_BOARD; then
    TASK="Mjlab-Climbing-Microban-Multi-Board$POLICY_SUFFIX"
  else
    TASK="Mjlab-Climbing-Microban$POLICY_SUFFIX"
  fi
fi

export MUJOCO_GL=egl

# Run from the repository root so logs end up in logs/rsl_rl/.
REPO_ROOT="$(cd "$(dirname "$(realpath "$0")")/../../.." && pwd)"

if $MULTI_BOARD; then
  if [[ $# -ge 1 && "$1" != -* ]]; then
    echo "--multi-board takes its motions from the board file, not $1 (use --boards-file)" >&2
    exit 1
  fi
  cd "$REPO_ROOT"
  CMD=(uv run train "$TASK" --env.scene.num-envs "$NUM_ENVS" --video "$VIDEO" "$@")
  echo "[INFO] Training $TASK with $NUM_ENVS envs on the boards of ${MICROBAN_BOARDS_FILE:-src/mjlab_microban/tasks/climbing/boards/multi_board.json}"
  if $DRY_RUN; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
  fi
  exec "${CMD[@]}"
fi

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

CMD=(uv run train "$TASK"
  --env.scene.num-envs "$NUM_ENVS"
  --env.commands.motion.motion-file "$MOTION_FILE"
  --video "$VIDEO"
  "$@")

echo "[INFO] Training $TASK with $NUM_ENVS envs on $MOTION_FILE, wall $WALL"
if $DRY_RUN; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
exec "${CMD[@]}"
