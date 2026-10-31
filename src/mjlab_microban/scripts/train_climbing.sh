#!/usr/bin/env bash
# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Train the Microban climbing task (tracking of a planned climb on the simple wall).
#
# Usage:
#   src/mjlab_microban/scripts/train_climbing.sh [motion.npz|planned_trajectory.npz] [extra train args...]
#
# Without a file, the default motion src/mjlab_microban/motion/climb_planning.npz is used.
# A planned trajectory (../planning/trajecoty.py --export) is first converted to a motion
# with planning_to_npz.py (written next to it as <name>_motion.npz).
# Environment variables:
#   TASK       Task id (default: Mjlab-Climbing-Microban, or Mjlab-Climbing-Microban-No-State-Estimation)
#   NUM_ENVS   Number of parallel environments (default: 4096)
#
# Example:
#   src/mjlab_microban/scripts/train_climbing.sh ../planning/trajectory.npz --agent.max-iterations 10000

set -euo pipefail

TASK="${TASK:-Mjlab-Climbing-Microban}"
NUM_ENVS="${NUM_ENVS:-4096}"

# Run from the repository root so logs end up in logs/rsl_rl/.
REPO_ROOT="$(cd "$(dirname "$(realpath "$0")")/../../.." && pwd)"

if [[ $# -ge 1 && "$1" != -* ]]; then
    MOTION_FILE="$(realpath "$1")"
    shift
else
    MOTION_FILE="$REPO_ROOT/src/mjlab_microban/motion/climb_planning.npz"
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

# A planner export has trunk_pos / q_joints, a motion has body_pos_w.
IS_PLAN="$(uv run python -c "import numpy as np, sys; print(int('trunk_pos' in np.load(sys.argv[1]).files))" "$MOTION_FILE")"
if [[ "$IS_PLAN" == "1" ]]; then
    PLAN_FILE="$MOTION_FILE"
    MOTION_FILE="${PLAN_FILE%.npz}_motion.npz"
    echo "[INFO] Converting planned trajectory $PLAN_FILE -> $MOTION_FILE"
    uv run python src/mjlab_microban/scripts/planning_to_npz.py \
        --input-file "$PLAN_FILE" \
        --output-file "$MOTION_FILE"
fi

echo "[INFO] Training $TASK with $NUM_ENVS envs on $MOTION_FILE"
exec uv run train "$TASK" \
    --env.scene.num-envs "$NUM_ENVS" \
    --env.commands.motion.motion-file "$MOTION_FILE" \
    --video True \
    "$@"
