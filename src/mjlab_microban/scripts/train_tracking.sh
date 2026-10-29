#!/usr/bin/env bash
# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Train the Microban tracking task on a retargeted motion.
#
# Usage:
#   src/mjlab_microban/scripts/train_tracking.sh <motion.npz|motion.csv> [extra train args...]
#
# A .csv motion is first converted to .npz with csv_to_npz.py (next to the CSV).
# Environment variables:
#   TASK       Task id (default: Mjlab-Tracking-Microban, or e.g. Mjlab-Tracking-Microban-No-State-Estimation)
#   NUM_ENVS   Number of parallel environments (default: 4096)
#   INPUT_FPS  Frame rate of the CSV motion (default: 30)
#
# Example:
#   TASK=Mjlab-Tracking-Microban-No-State-Estimation src/mjlab_microban/scripts/train_tracking.sh dance.csv --agent.max-iterations 10000

set -euo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '11,23p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

TASK="${TASK:-Mjlab-Tracking-Microban}"
NUM_ENVS="${NUM_ENVS:-4096}"
INPUT_FPS="${INPUT_FPS:-30}"

MOTION_FILE="$(realpath "$1")"
shift

if [[ ! -f "$MOTION_FILE" ]]; then
    echo "Motion file not found: $MOTION_FILE" >&2
    exit 1
fi

# Run from the repository root so logs end up in logs/rsl_rl/.
cd "$(dirname "$(realpath "$0")")/../../.."

case "$MOTION_FILE" in
    *.npz)
        ;;
    *.csv)
        NPZ_FILE="${MOTION_FILE%.csv}.npz"
        echo "[INFO] Converting $MOTION_FILE -> $NPZ_FILE"
        uv run python src/mjlab_microban/scripts/csv_to_npz.py \
            --input-file "$MOTION_FILE" \
            --output-file "$NPZ_FILE" \
            --input-fps "$INPUT_FPS"
        MOTION_FILE="$NPZ_FILE"
        ;;
    *)
        echo "Unsupported motion file (expected .npz or .csv): $MOTION_FILE" >&2
        exit 1
        ;;
esac

echo "[INFO] Training $TASK with $NUM_ENVS envs on $MOTION_FILE"
exec uv run train "$TASK" \
    --env.scene.num-envs "$NUM_ENVS" \
    --env.commands.motion.motion-file "$MOTION_FILE" \
    "$@"
