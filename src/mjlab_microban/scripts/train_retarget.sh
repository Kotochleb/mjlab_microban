#!/usr/bin/env bash
# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Train the Microban tracking task on a retargeted motion, with 4096 parallel
# environments and wandb logging, using the motion in src/mjlab_microban/motion/.
#
# Usage:
#   src/mjlab_microban/scripts/train_retarget.sh [extra train args...]

set -euo pipefail

# Run from the repository root so logs end up in logs/rsl_rl/.
cd "$(dirname "$(realpath "$0")")/../../.."

MOTION_FILE="src/mjlab_microban/motion/walk1_subject1_first500.npz"

uv run train Mjlab-Tracking-Microban \
    --env.scene.num-envs 4096 \
    --env.commands.motion.motion-file "$MOTION_FILE" \
    --agent.logger wandb \
    "$@"
