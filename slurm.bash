#!/bin/bash
#SBATCH --job-name=mjlab_microban_climbing
#SBATCH --output=logs/%A_%a/log.out
#SBATCH --error=logs/%A_%a/log.err
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --constraint=h100
#SBATCH --account=mny@h100
#SBATCH --qos=qos_gpu_h100-t3

module purge
module load arch/h100
module load uv
module load git

echo "Using GPU arch: ${JZ_GPU_ARCH}"

# Run training
set -euxo pipefail

export MUJOCO_GL=egl
export CUDA_CACHE_PATH=$WORK/.cuda_cache
export CUDA_CACHE_DISABLE=1
export TMPDIR=$WORK/.tmp

# Single-board climbing task: the env builds its wall from the motion file's "wall" field, so each array
# task trains one policy on one wall. Override with TASK=... sbatch slurm.bash.
TASK="${TASK:-Mjlab-Climbing-Microban-Adaptive-Phase}"

# One array task per wall; keep --array above equal to 0-(N-1). Each wall has its own JSON file in
# src/mjlab_microban/tasks/climbing/boards/walls/<wall>.json and a planned climb in
# src/mjlab_microban/motion/multi_board/<wall>.npz (scripts/make_multi_board_motions.py).
# l2_s3.json has no planned motion yet; add it here once it has one.
WALLS=(
  "gen_l3_s0"
  "gen_l3_s1"
)

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${#WALLS[@]}" ]]; then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} but only ${#WALLS[@]} walls; fix --array" >&2
  exit 1
fi

WALL="${WALLS[$SLURM_ARRAY_TASK_ID]}"
WALL_FILE="src/mjlab_microban/tasks/climbing/boards/walls/${WALL}.json"
MOTION_FILE="$(realpath "src/mjlab_microban/motion/multi_board/${WALL}.npz")"
for f in "${WALL_FILE}" "${MOTION_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing ${f} for wall ${WALL}" >&2
    exit 1
  fi
done
RUN_NAME="${WALL}"
# Training is relaunched immediately, at most TRAIN_RETRIES times, but only after an attempt that
# failed within TRAIN_RETRY_WINDOW seconds of its start; a later failure fails the whole script.
TRAIN_RETRIES=5
TRAIN_RETRY_WINDOW=$((20 * 60))
echo "Launching training for task: ${TASK} on wall: ${WALL} (motion: ${MOTION_FILE})"

# Every attempt starts afresh; retries get their own run name (and directory), suffixed with the
# attempt number.
for ((attempt = 0; attempt <= TRAIN_RETRIES; attempt++)); do
  run="${RUN_NAME}"
  if ((attempt > 0)); then
    run="${RUN_NAME}-Retry${attempt}"
  fi
  echo "Training attempt ${attempt} (run name: ${run})"
  start=${SECONDS}
  if uv run train "${TASK}" \
    --env.scene.num-envs 4096 \
    --env.commands.motion.motion-file "${MOTION_FILE}" \
    --agent.save-interval 100 \
    --agent.max-iterations 24_000 \
    --agent.run-name "${run}" \
    --agent.logger tensorboard \
    --log-root "${WORK}/tensorboard_logs" \
    --agent.experiment-name "climbing-single-wall" \
    --env.sim.graph-conditional False \
    --video True; then
    exit 0
  fi
  elapsed=$((SECONDS - start))
  echo "Training attempt ${attempt} (${run}) failed after ${elapsed} s" >&2
  if ((elapsed >= TRAIN_RETRY_WINDOW)); then
    echo "Training failed after the ${TRAIN_RETRY_WINDOW} s retry window; not retrying" >&2
    exit 1
  fi
done
echo "Training failed after ${TRAIN_RETRIES} retries" >&2
exit 1
