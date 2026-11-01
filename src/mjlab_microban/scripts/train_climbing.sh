#!/usr/bin/env bash
# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Train the Microban climbing task: tracking of a climb planned by rock-climbing-workshop on a climbing_sim wall,
# or play back a trained policy (--play).
#
# Usage:
#   src/mjlab_microban/scripts/train_climbing.sh [variant flags] [motion.npz|planner.npz] [extra train args...]
#   src/mjlab_microban/scripts/train_climbing.sh --play [variant flags] [play flags] [motion.npz] [extra play args...]
#   src/mjlab_microban/scripts/train_climbing.sh --replay [variant flags] [motion.npz] [extra play args...]
#   src/mjlab_microban/scripts/train_climbing.sh --deploy [variant flags] [play flags] [deploy flags] [motion.npz]
#   src/mjlab_microban/scripts/train_climbing.sh --finetune [fine-tune flags] [extra train args...]
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
#   --adaptive-phase        adaptive phase dynamics (Sleiman et al., CoRL 2024, tasks/climbing/adaptive_phase.py):
#                           the reference waits for the robot when it falls behind, plus a policy residual on its rate
#                           (Mjlab-Climbing-Microban-Multi-Board[-Conv1d|-Conv1d-History]): the boards and their
#                           motions come from a board file, not from a motion file argument
#   --boards-file F         the board file of --multi-board (default: tasks/climbing/boards/multi_board.json);
#                           make its motions first with src/mjlab_microban/scripts/make_multi_board_motions.py
# Fine-tune flags (resume a trained policy and keep training it; tasks/climbing/microban_climbing_finetune_env_cfg.py):
#   --finetune              train the -Finetune variant of the task: the used holds collide as the legacy 5-box cup
#                           (five upright boxes, no trims / tilts / posts, before git 20a0573), every curriculum
#                           starts at its final value (full DR ranges, realistic motor power, all board levels,
#                           maximum board horizons), the adaptive phase level (regularisation penalties) starts at
#                           0.5 and rises slowly, the policy std is reset to 0.45, and the firmware P gain is drawn
#                           per env and joint in 120 * [0.9, 1.1]. Implies --adaptive-phase; single board (the
#                           networks are the same as the multi-board ones), on the given motion, by default
#                           src/mjlab_microban/motion/l1s0_cycle_settle2.npz (the level 1 wall climbed with a fixed
#                           gait, 2 s still after each move); --multi-board for the board file instead, or set
#                           TASK=...-Finetune. Without --play: training resumes from
#                           the last checkpoint of --finetune-from (default: $FINETUNE_FROM, or
#                           kotochleb-cnrs-laas/mjlab_microban_climbing/al2qgf37, itself a fine-tune of cp4zxscz on
#                           Mjlab-Climbing-Microban-Adaptive-Phase-Multi-Board-Finetune). With --play: play a fine-tuned run
#                           (--wandb-run-path / --run / --checkpoint) in the fine-tuning env
#   --finetune-from R       the W&B run to fine-tune (implies --finetune): entity/project/run_id
#   --finetune-checkpoint NAME     its checkpoint to start from, e.g. model_7000.pt (implies --finetune; default:
#                           its latest)
#   The run's networks have to fit the task: give the same variant flags (--policy) as the run was trained with.
# Run flags:
#   --num-envs N            number of parallel environments (default: 4096, or $NUM_ENVS)
#   --no-video              do not record training videos
#   --dry-run               print the train / play command instead of running it
#   -h, --help              this help
# Play flags (uv run play: the env's play config, a trained policy in the viewer):
#   --play                  play back a trained policy instead of training
#   --run DIR               the training run to play (implies --play; default: the latest run with a checkpoint
#                           in logs/rsl_rl/<experiment of the task>)
#   --checkpoint F          the checkpoint to play (implies --play; default: the run's latest model_<it>.pt)
#   --wandb-run-path R      play a W&B run instead of a local one (implies --play): entity/project/run_id or the
#                           run's URL; its checkpoints are downloaded to logs/rsl_rl/<experiment>/wandb_checkpoints
#   --wandb-checkpoint-name NAME   the W&B run's checkpoint to play, e.g. model_5000.pt (default: its latest)
#   --replay                replay the motion, without a policy (implies --play): play's zero agent, i.e. the
#                           reference joint positions as PD targets (residual actions), and no terminations.
#                           --run / --wandb-run-path only pick its motion (no checkpoint is loaded); for the open
#                           loop viewer with a frame slider, see scripts/play_open_loop.py
#   --video                 also record a clip to <run>/videos/play (play's --video-length, default 200 steps)
#   --viser                 view in the Viser web viewer (implies --play; http://localhost:8080) instead of the
#                           native MuJoCo one: its Rewards tab plots each reward term live (bars of the ~1 s mean,
#                           and their histories), its Metrics tab the env's metrics; the env shown is picked in
#                           Controls > Scene
# Deploy flags (the policy --play would play, on the real robot):
#   --deploy                export the policy --play would play (same --run / --checkpoint / --wandb-run-path and
#                           motion selection) to a deployment bundle for the robot's 50 Hz loop, checked against the
#                           env (scripts/export_climbing_deploy.py: the move's observation, actions and targets have
#                           to match the env's, tick by tick); single-board MLP tasks (the adaptive phase or not)
#   --deploy-out DIR        the bundle's folder (default: logs/deploy/<run>_<checkpoint>)
#   --check-steps N         ticks of the parity check (default 500, 10 s; 0: none)
#                           The robot's side is in rock-climbing-workshop ($RCR_REPO, default ../rock-climbing-workshop):
#                           src/rcr/controller/mjlab_policy.py runs the policy, src/rcr/robot/climb_policy_move.py is
#                           its Microban move; the export checks that controller against the env
#   --sim-check             then run the move closed loop in the organisers' simulator, hanging on the wall's start
#                           holds (rock-climbing-workshop's scripts/sim_check_climb_policy.py, with its Python,
#                           $RCR_PYTHON, default <workshop>/.venv/bin/python: uv sync --group sim --group rl there);
#                           --sim-seconds S of climbing (default 30)
#   --host H                then copy our Pi code (scripts/link_pi.py deploy H: ~/rcr/src) and the bundle
#                           (H:~/climb_policy) to the robot, and run the robot's program with the move (Microban's
#                           make run HOST=H MOVES=src/rcr/robot/climb_policy_move.py, in $MICROBAN, default
#                           <workshop>/third_party/microban): the robot on the wall, its balls in the
#                           start holds, rope on. Then, in the dashboard, start "climb_policy": the joints ramp to the
#                           climb's first pose (2 s), hold it (2 s), and the policy climbs; stop it to end
#                           (it holds 5 s, then the torque goes off). Its log: H:~/climb_logs/
#                           Done by the workshop's scripts/robot_climb_policy.py (as sim_check_climb_policy.py
#                           in the sim): it asks before starting, then stops the move after --seconds S of
#                           climbing (default 60); --view shows the robot live in MuJoCo's viewer
#   --offline               no W&B (no internet): the checkpoint downloaded before, in
#                           logs/rsl_rl/<experiment>/wandb_checkpoints/<run_id>, and the run's motion cached
#                           there (W&B unreachable: the same, by itself). A bundle already exported from the
#                           same checkpoint and motion, its check passed, is reused (no export)
#   In play, --num-envs defaults to 1 and the variant flags have to match the run's (--policy, --multi-board /
#   --boards-file). The wall's motion .npz is selected automatically when not given: the one the run was trained
#   on (its params/env.yaml, or the W&B run's config), found by its path, else by its file name under
#   src/mjlab_microban/motion; when that fails, the level 1 wall's src/mjlab_microban/motion/l1s0.npz.
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
#   src/mjlab_microban/scripts/train_climbing.sh --play --conv1d-history
#   src/mjlab_microban/scripts/train_climbing.sh --finetune --agent.max-iterations 5000
#   src/mjlab_microban/scripts/train_climbing.sh --finetune-checkpoint model_7000.pt
#   src/mjlab_microban/scripts/train_climbing.sh --replay src/mjlab_microban/motion/multi_board/l2s19.npz
#   src/mjlab_microban/scripts/train_climbing.sh --play --conv1d --wandb-run-path my-team/mjlab_microban_climbing/abc123xy
#   src/mjlab_microban/scripts/train_climbing.sh --play --viser --adaptive-phase --wandb-run-path my-team/mjlab_microban_climbing/abc123xy
#   src/mjlab_microban/scripts/train_climbing.sh --play --checkpoint logs/rsl_rl/mjlab_microban_climbing_residual/<run>/model_5000.pt
#   src/mjlab_microban/scripts/train_climbing.sh --deploy --adaptive-phase --wandb-run-path my-team/mjlab_microban_climbing/abc123xy --sim-check
#   src/mjlab_microban/scripts/train_climbing.sh --deploy --adaptive-phase --wandb-run-path my-team/mjlab_microban_climbing/abc123xy --host microban

set -euo pipefail

usage() { sed -n '/^# Usage:/,/^# Examples:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; }

NUM_ENVS_SET=false  # --num-envs or $NUM_ENVS given explicitly (play defaults to 1 env)
[[ -n "${NUM_ENVS:-}" ]] && NUM_ENVS_SET=true
NUM_ENVS="${NUM_ENVS:-4096}"
VIDEO="${VIDEO:-True}"
POLICY="${POLICY:-mlp}"
DRY_RUN=false
MULTI_BOARD=false
ADAPTIVE_PHASE=false
POLICY_FLAG=false  # --policy given explicitly
PLAY=false
PLAY_RUN=""
PLAY_CHECKPOINT=""
PLAY_VIDEO=false
PLAY_VISER=false
REPLAY=false
WANDB_RUN=""
WANDB_CHECKPOINT=""
DEPLOY=false
DEPLOY_OUT=""
CHECK_STEPS=500
SIM_CHECK=false
SIM_SECONDS=30
ROBOT_HOST=""
ROBOT_VIEW=false
ROBOT_SECONDS=60
OFFLINE=false
FINETUNE=false
FINETUNE_FROM="${FINETUNE_FROM:-kotochleb-cnrs-laas/mjlab_microban_climbing/al2qgf37}"
FINETUNE_CHECKPOINT=""

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
    --num-envs) need_value "$1" $#; NUM_ENVS="$2"; NUM_ENVS_SET=true; shift 2 ;;
    --num-envs=*) NUM_ENVS="${1#--num-envs=}"; NUM_ENVS_SET=true; shift ;;
    --no-video) VIDEO=False; shift ;;
    --multi-board) MULTI_BOARD=true; shift ;;
    --adaptive-phase) ADAPTIVE_PHASE=true; shift ;;
    --boards-file) need_value "$1" $#; MULTI_BOARD=true; export MICROBAN_BOARDS_FILE="$(realpath "$2")"; shift 2 ;;
    --boards-file=*) MULTI_BOARD=true; export MICROBAN_BOARDS_FILE="$(realpath "${1#--boards-file=}")"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --play) PLAY=true; shift ;;
    --run) need_value "$1" $#; PLAY=true; PLAY_RUN="$2"; shift 2 ;;
    --run=*) PLAY=true; PLAY_RUN="${1#--run=}"; shift ;;
    --checkpoint) need_value "$1" $#; PLAY=true; PLAY_CHECKPOINT="$2"; shift 2 ;;
    --checkpoint=*) PLAY=true; PLAY_CHECKPOINT="${1#--checkpoint=}"; shift ;;
    --video) PLAY_VIDEO=true; shift ;;
    --viser) PLAY=true; PLAY_VISER=true; shift ;;
    --replay) PLAY=true; REPLAY=true; shift ;;
    --wandb-run-path|--wandb-run) need_value "$1" $#; PLAY=true; WANDB_RUN="$2"; shift 2 ;;
    --wandb-run-path=*|--wandb-run=*) PLAY=true; WANDB_RUN="${1#*=}"; shift ;;
    --wandb-checkpoint-name|--wandb-checkpoint) need_value "$1" $#; PLAY=true; WANDB_CHECKPOINT="$2"; shift 2 ;;
    --wandb-checkpoint-name=*|--wandb-checkpoint=*) PLAY=true; WANDB_CHECKPOINT="${1#*=}"; shift ;;
    --deploy) PLAY=true; DEPLOY=true; shift ;;
    --deploy-out) need_value "$1" $#; DEPLOY=true; PLAY=true; DEPLOY_OUT="$2"; shift 2 ;;
    --deploy-out=*) DEPLOY=true; PLAY=true; DEPLOY_OUT="${1#*=}"; shift ;;
    --check-steps) need_value "$1" $#; CHECK_STEPS="$2"; shift 2 ;;
    --check-steps=*) CHECK_STEPS="${1#*=}"; shift ;;
    --sim-check) DEPLOY=true; PLAY=true; SIM_CHECK=true; shift ;;
    --sim-seconds) need_value "$1" $#; SIM_SECONDS="$2"; shift 2 ;;
    --sim-seconds=*) SIM_SECONDS="${1#*=}"; shift ;;
    --view) ROBOT_VIEW=true; shift ;;
    --seconds) need_value "$1" $#; ROBOT_SECONDS="$2"; shift 2 ;;
    --seconds=*) ROBOT_SECONDS="${1#*=}"; shift ;;
    --offline) OFFLINE=true; shift ;;
    --finetune) FINETUNE=true; shift ;;
    --finetune-from) need_value "$1" $#; FINETUNE=true; FINETUNE_FROM="$2"; shift 2 ;;
    --finetune-from=*) FINETUNE=true; FINETUNE_FROM="${1#*=}"; shift ;;
    --finetune-checkpoint) need_value "$1" $#; FINETUNE=true; FINETUNE_CHECKPOINT="$2"; shift 2 ;;
    --finetune-checkpoint=*) FINETUNE=true; FINETUNE_CHECKPOINT="${1#*=}"; shift ;;
    --host) need_value "$1" $#; DEPLOY=true; PLAY=true; ROBOT_HOST="$2"; shift 2 ;;
    --host=*) DEPLOY=true; PLAY=true; ROBOT_HOST="${1#*=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

$PLAY && ! $NUM_ENVS_SET && NUM_ENVS=1
if $FINETUNE; then
  $DEPLOY && { echo "--finetune is a training env: drop --deploy (deploy the fine-tuned run without it)" >&2; exit 1; }
  if [[ -n "${TASK:-}" ]]; then
    [[ "$TASK" == *-Finetune ]] || { echo "--finetune needs a -Finetune task, got TASK=$TASK" >&2; exit 1; }
    [[ "$TASK" == *-Multi-Board* ]] && MULTI_BOARD=true
  else
    ADAPTIVE_PHASE=true
  fi
fi
if $DEPLOY; then
  $REPLAY && { echo "--deploy exports a policy: drop --replay" >&2; exit 1; }
  $MULTI_BOARD && { echo "--deploy takes one board: drop --multi-board (give the board's motion file)" >&2; exit 1; }
  $PLAY_VIDEO && { echo "--deploy records no video: drop --video" >&2; exit 1; }
  $PLAY_VISER && { echo "--deploy opens no viewer: drop --viser" >&2; exit 1; }
fi
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
  TASK_PREFIX="Mjlab-Climbing-Microban"
  $ADAPTIVE_PHASE && TASK_PREFIX+="-Adaptive-Phase"
  if $MULTI_BOARD; then
    TASK="$TASK_PREFIX-Multi-Board$POLICY_SUFFIX"
  else
    TASK="$TASK_PREFIX$POLICY_SUFFIX"
  fi
  $FINETUNE && TASK+="-Finetune"
fi

# Run from the repository root so logs end up in logs/rsl_rl/.
REPO_ROOT="$(cd "$(dirname "$(realpath "$0")")/../../.." && pwd)"

# ---------------------------------------------------------------------------------------------------
# Play
# ---------------------------------------------------------------------------------------------------

# The rsl_rl experiment (logs/rsl_rl/<name>) of a task id, as in tasks/climbing/rl_cfg.py and __init__.py.
experiment_name() {
  local rest="${1#Mjlab-Climbing-Microban}" name="mjlab_microban_climbing_residual" multi="" adaptive="" finetune=""
  [[ "$rest" != "$1" ]] || return 1
  if [[ "$rest" == *-Finetune ]]; then
    finetune="_finetune"
    rest="${rest%-Finetune}"
  fi
  if [[ "$rest" == -Adaptive-Phase* ]]; then
    adaptive="_adaptive_phase"
    rest="${rest#-Adaptive-Phase}"
  fi
  if [[ "$rest" == -Multi-Board* ]]; then
    multi="_multi_board"
    rest="${rest#-Multi-Board}"
  fi
  case "$rest" in
    "") ;;
    -Conv1d) name+="_conv1d" ;;
    -Conv1d-History) name+="_conv1d_history" ;;
    *) return 1 ;;
  esac
  echo "$name$adaptive$multi$finetune"
}

# The latest model_<iteration>.pt of a run directory (by iteration), or nothing.
latest_checkpoint() {
  local f it best="" best_it=-1
  for f in "$1"/model_*.pt; do
    [[ -f "$f" ]] || continue
    it="${f##*/model_}"; it="${it%.pt}"
    [[ "$it" =~ ^[0-9]+$ ]] || continue
    if (( it > best_it )); then best_it=$it; best="$f"; fi
  done
  echo "$best"
}

# A motion path recorded by a run (maybe on another machine) -> the local file, or nothing: the path itself,
# the same path under this repo's src/mjlab_microban/, else the one file of that name in src/mjlab_microban/motion.
local_motion() {
  local p="$1" rel found
  [[ -n "$p" ]] || return 0
  if [[ -f "$p" ]]; then realpath "$p"; return 0; fi
  rel="${p##*src/mjlab_microban/}"
  if [[ "$p" == *src/mjlab_microban/* && -f "$REPO_ROOT/src/mjlab_microban/$rel" ]]; then
    echo "$REPO_ROOT/src/mjlab_microban/$rel"; return 0
  fi
  found="$(find "$REPO_ROOT/src/mjlab_microban/motion" -type f -name "$(basename "$p")" 2>/dev/null)"
  if [[ -n "$found" && "$(wc -l <<< "$found")" -eq 1 ]]; then
    echo "$found"
  fi
}

# The policy play would play, exported for the robot ($CHECKPOINT / $WANDB_RUN, $MOTION_FILE set by play below),
# then checked in the organisers' simulator (--sim-check) and run on the robot (--host).
deploy() {
  [[ $# -eq 0 ]] || { echo "--deploy takes no extra arguments, got: $*" >&2; exit 1; }
  # The robot's side lives in rock-climbing-workshop: the controller (src/rcr/controller/mjlab_policy.py), its
  # Microban move (src/rcr/robot/climb_policy_move.py), the closed-loop sim check and Microban itself.
  local ws="${RCR_REPO:-$REPO_ROOT/../rock-climbing-workshop}"
  ws="$(cd "$ws" 2>/dev/null && pwd)" || { echo "rock-climbing-workshop not found (set RCR_REPO)" >&2; exit 1; }
  local ws_python="${RCR_PYTHON:-${SIM_PYTHON:-$ws/.venv/bin/python}}" microban="${MICROBAN:-$ws/third_party/microban}"
  local move="$ws/src/rcr/robot/climb_policy_move.py" src=() out="$DEPLOY_OUT"
  [[ -f "$move" ]] || { echo "No $move: rock-climbing-workshop without the mjlab policy controller" >&2; exit 1; }
  if [[ -n "$WANDB_RUN" ]]; then
    src=(--wandb-run-path "$WANDB_RUN")
    [[ -n "$WANDB_CHECKPOINT" ]] && src+=(--wandb-checkpoint-name "$WANDB_CHECKPOINT")
  else
    src=(--checkpoint-file "$CHECKPOINT")
  fi
  local export_cmd=(uv run python src/mjlab_microban/scripts/export_climbing_deploy.py --task "$TASK" "${src[@]}"
    --motion-file "$MOTION_FILE" --check-steps "$CHECK_STEPS")
  [[ -n "$out" ]] && export_cmd+=(--out "$out")
  $OFFLINE && export_cmd+=(--offline)
  local robot_cmd=("$ws_python" "$ws/scripts/robot_climb_policy.py" "<bundle>" --host "$ROBOT_HOST"
    --seconds "$ROBOT_SECONDS" --microban "$microban")
  $ROBOT_VIEW && robot_cmd+=(--view)
  if $DRY_RUN; then
    printf 'RCR_SRC=%q ' "$ws/src"; printf '%q ' "${export_cmd[@]}"; echo
    $SIM_CHECK && echo "$ws_python $ws/scripts/sim_check_climb_policy.py <bundle> --seconds $SIM_SECONDS"
    [[ -n "$ROBOT_HOST" ]] && { printf '%q ' "${robot_cmd[@]}"; echo; }
    return 0
  fi
  echo "[INFO] Exporting $CHECKPOINT ($TASK) on $MOTION_FILE for the robot"
  local log
  log="$(mktemp)"
  RCR_SRC="$ws/src" "${export_cmd[@]}" 2>&1 | tee "$log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { rm -f "$log"; echo "[ERROR] export failed: nothing deployed" >&2; exit 1; }
  out="$(sed -n 's/^\[INFO\] bundle: //p' "$log" | tail -1)"
  rm -f "$log"
  [[ -n "$out" && -f "$out/climb_bundle.npz" ]] || { echo "[ERROR] no bundle written" >&2; exit 1; }

  if $SIM_CHECK || [[ -n "$ROBOT_HOST" ]]; then
    [[ -x "$ws_python" ]] || { echo "No rock-climbing-workshop Python at $ws_python (uv sync --group sim --group rl there, or set RCR_PYTHON)" >&2; exit 1; }
  fi
  if $SIM_CHECK; then
    echo "[INFO] Closed loop in the organisers' simulator: $SIM_SECONDS s of climbing"
    "$ws_python" "$ws/scripts/sim_check_climb_policy.py" "$out" --seconds "$SIM_SECONDS"
  fi

  if [[ -n "$ROBOT_HOST" ]]; then
    robot_cmd[2]="$out"
    echo "[INFO] On the robot: ${robot_cmd[*]}"
    exec "${robot_cmd[@]}"
  else
    echo "[INFO] Bundle ready: $out"
    echo "       on the robot: $0 --deploy ... --host <robot>"
  fi
}

if $PLAY; then
  cd "$REPO_ROOT"

  if [[ -n "$WANDB_RUN" ]]; then
    if [[ -n "$PLAY_CHECKPOINT" || -n "$PLAY_RUN" ]]; then
      echo "--wandb-run-path plays a W&B run: drop --run / --checkpoint" >&2; exit 1
    fi
    # https://wandb.ai/<entity>/<project>/runs/<id>[?...] -> <entity>/<project>/<id>
    if [[ "$WANDB_RUN" == http*://* ]]; then
      WANDB_RUN="${WANDB_RUN%%[?#]*}"; WANDB_RUN="${WANDB_RUN%/}"
      WANDB_RUN="$(echo "$WANDB_RUN" | sed -E 's#^https?://[^/]+/([^/]+)/([^/]+)/runs/([^/]+).*#\1/\2/\3#')"
    fi
    [[ "$WANDB_RUN" =~ ^[^/]+/[^/]+/[^/]+$ ]] || {
      echo "--wandb-run-path takes entity/project/run_id or the run's URL, got $WANDB_RUN" >&2; exit 1; }
  elif [[ -n "$WANDB_CHECKPOINT" ]]; then
    echo "--wandb-checkpoint-name needs --wandb-run-path" >&2; exit 1
  elif $REPLAY; then
    # No checkpoint: a run only gives the motion.
    RUN_DIR=""
    if [[ -n "$PLAY_CHECKPOINT" ]]; then
      RUN_DIR="$(dirname "$(realpath "$PLAY_CHECKPOINT")")"
    elif [[ -n "$PLAY_RUN" ]]; then
      [[ -d "$PLAY_RUN" ]] || { echo "Run directory not found: $PLAY_RUN" >&2; exit 1; }
      RUN_DIR="$(realpath "$PLAY_RUN")"
    fi
  elif [[ -n "$PLAY_CHECKPOINT" ]]; then
    [[ -f "$PLAY_CHECKPOINT" ]] || { echo "Checkpoint not found: $PLAY_CHECKPOINT" >&2; exit 1; }
    CHECKPOINT="$(realpath "$PLAY_CHECKPOINT")"
    RUN_DIR="$(dirname "$CHECKPOINT")"
  else
    if [[ -n "$PLAY_RUN" ]]; then
      [[ -d "$PLAY_RUN" ]] || { echo "Run directory not found: $PLAY_RUN" >&2; exit 1; }
      RUN_DIR="$(realpath "$PLAY_RUN")"
    else
      EXPERIMENT="$(experiment_name "$TASK")" || {
        echo "Unknown experiment of task $TASK: give the run (--run) or the checkpoint (--checkpoint)" >&2; exit 1; }
      LOG_ROOT="$REPO_ROOT/logs/rsl_rl/$EXPERIMENT"
      RUN_DIR=""
      # Run directories are named <date>_<time>[_<run name>]: the last one with a checkpoint.
      while IFS= read -r d; do
        [[ -n "$(latest_checkpoint "$d")" ]] && RUN_DIR="$d"
      done < <(find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
      [[ -n "$RUN_DIR" ]] || {
        echo "No run with a checkpoint in $LOG_ROOT (train $TASK first, or give --run / --checkpoint)" >&2; exit 1; }
    fi
    CHECKPOINT="$(latest_checkpoint "$RUN_DIR")"
    [[ -n "$CHECKPOINT" ]] || { echo "No model_<iteration>.pt in $RUN_DIR" >&2; exit 1; }
  fi

  if $REPLAY; then
    CHECKPOINT="the motion (zero residual actions)"
    CMD=(uv run play "$TASK" --agent zero --no-terminations True --num-envs "$NUM_ENVS")
    $PLAY_VIDEO && echo "[WARN] play records no video without a policy: --video ignored"
    PLAY_VIDEO=false
  elif [[ -n "$WANDB_RUN" ]]; then
    CHECKPOINT="W&B run $WANDB_RUN (${WANDB_CHECKPOINT:-latest checkpoint})"
    CMD=(uv run play "$TASK" --wandb-run-path "$WANDB_RUN" --num-envs "$NUM_ENVS")
    [[ -n "$WANDB_CHECKPOINT" ]] && CMD+=(--wandb-checkpoint-name "$WANDB_CHECKPOINT")
  else
    CMD=(uv run play "$TASK" --checkpoint-file "$CHECKPOINT" --num-envs "$NUM_ENVS")
  fi
  $PLAY_VIDEO && CMD+=(--video True)
  $PLAY_VISER && CMD+=(--viewer viser)

  if $MULTI_BOARD; then
    if [[ $# -ge 1 && "$1" != -* ]]; then
      echo "--multi-board takes its motions from the board file, not $1 (use --boards-file)" >&2
      exit 1
    fi
    if $REPLAY; then
      # play needs a motion file without a policy: the first board's (the env's own), each env replays its board's.
      FIRST_MOTION="$(python3 -c "
import json, sys
b = json.load(open(sys.argv[1]))['boards'][0]
print(b.get('motion', f\"multi_board/l{b['level']}s{b['seed']}.npz\"))
" "${MICROBAN_BOARDS_FILE:-$REPO_ROOT/src/mjlab_microban/tasks/climbing/boards/multi_board.json}")"
      CMD+=(--motion-file "$REPO_ROOT/src/mjlab_microban/motion/$FIRST_MOTION")
    fi
    echo "[INFO] Playing $CHECKPOINT ($TASK) on the boards of ${MICROBAN_BOARDS_FILE:-src/mjlab_microban/tasks/climbing/boards/multi_board.json}"
  else
    # The motion (and so the wall): the command line's, else the one the run was trained on.
    if [[ -n "$WANDB_RUN" ]]; then
      # The run's motion, cached next to its downloaded checkpoints (for --offline / no internet).
      MOTION_CACHE=""
      if EXP="$(experiment_name "$TASK")"; then
        MOTION_CACHE="$REPO_ROOT/logs/rsl_rl/$EXP/wandb_checkpoints/${WANDB_RUN##*/}/motion_file.txt"
      fi
      if $OFFLINE; then
        RUN_MOTION="$(cat "$MOTION_CACHE" 2>/dev/null || true)"
      else
      # rsl_rl logs the env cfg to the W&B run's config (env_cfg).
      RUN_MOTION="$(WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-20}" uv run python -c "
import sys, wandb
try:
    cfg = wandb.Api().run(sys.argv[1]).config['env_cfg']
    cfg = cfg.get('value', cfg)
    print(cfg['commands']['motion']['motion_file'] or '')
except Exception as e:
    print(f'[WARN] No motion in the config of W&B run {sys.argv[1]}: {e!r}', file=sys.stderr)
" "$WANDB_RUN" || true)"
        if [[ -n "$RUN_MOTION" && -n "$MOTION_CACHE" ]]; then
          mkdir -p "$(dirname "$MOTION_CACHE")" && echo "$RUN_MOTION" > "$MOTION_CACHE"
        elif [[ -z "$RUN_MOTION" && -f "$MOTION_CACHE" ]]; then
          RUN_MOTION="$(cat "$MOTION_CACHE")"
          echo "[INFO] W&B unreachable: the run's cached motion $RUN_MOTION"
        fi
      fi
    else
      RUN_MOTION="$(uv run python -c "
import sys, yaml
try:
    with open(sys.argv[1]) as f:
        print(yaml.safe_load(f)['commands']['motion']['motion_file'] or '')
except (OSError, KeyError, TypeError, yaml.YAMLError):
    pass
" "$RUN_DIR/params/env.yaml" 2>/dev/null || true)"
    fi
    RUN_MOTION_LOCAL="$(local_motion "$RUN_MOTION")"
    if [[ $# -ge 1 && "$1" != -* ]]; then
      MOTION_FILE="$(realpath "$1")"
      shift
      if [[ -n "$RUN_MOTION" && "${RUN_MOTION_LOCAL:-$(realpath -m "$RUN_MOTION")}" != "$MOTION_FILE" ]]; then
        echo "[WARN] The run was trained on $RUN_MOTION, playing on $MOTION_FILE"
      fi
    elif [[ -n "$RUN_MOTION_LOCAL" ]]; then
      MOTION_FILE="$RUN_MOTION_LOCAL"
      echo "[INFO] Motion of the run: $MOTION_FILE"
    else
      if [[ -n "$RUN_MOTION" ]]; then
        echo "[WARN] The run's motion $RUN_MOTION is not here: playing on the level 1 wall (l1s0.npz)"
      elif $REPLAY && [[ -z "$RUN_DIR" && -z "$WANDB_RUN" ]]; then
        echo "[INFO] No motion given: replaying the level 1 wall's l1s0.npz"
      else
        echo "[INFO] Unknown motion of the run: playing on the level 1 wall (l1s0.npz)"
      fi
      MOTION_FILE="$REPO_ROOT/src/mjlab_microban/motion/l1s0.npz"
    fi
    [[ -f "$MOTION_FILE" ]] || { echo "Motion file not found: $MOTION_FILE" >&2; exit 1; }
    # play's --motion-file is also where the env reads its wall from (tasks/climbing/wall.py).
    CMD+=(--motion-file "$MOTION_FILE")
    echo "[INFO] Playing $CHECKPOINT ($TASK) on $MOTION_FILE"
  fi

  if $DEPLOY; then
    deploy "$@"
    exit 0
  fi

  CMD+=("$@")
  if $DRY_RUN; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
  fi
  exec "${CMD[@]}"
fi

# ---------------------------------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------------------------------

export MUJOCO_GL=egl

# Fine-tuning: resume the run's checkpoint (downloaded from W&B), under a run name after it.
RESUME_ARGS=()
if $FINETUNE; then
  RESUME_ARGS=(--agent.resume True --wandb-run-path "$FINETUNE_FROM" --agent.run-name "finetune-${FINETUNE_FROM##*/}")
  [[ -n "$FINETUNE_CHECKPOINT" ]] && RESUME_ARGS+=(--wandb-checkpoint-name "$FINETUNE_CHECKPOINT")
  echo "[INFO] Fine-tuning W&B run $FINETUNE_FROM (${FINETUNE_CHECKPOINT:-latest checkpoint})"
fi

if $MULTI_BOARD; then
  if [[ $# -ge 1 && "$1" != -* ]]; then
    echo "--multi-board takes its motions from the board file, not $1 (use --boards-file)" >&2
    exit 1
  fi
  cd "$REPO_ROOT"
  CMD=(uv run train "$TASK" --env.scene.num-envs "$NUM_ENVS" --video "$VIDEO" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} "$@")
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
elif $FINETUNE; then
  MOTION_FILE="$REPO_ROOT/src/mjlab_microban/motion/l1s0_cycle_settle2.npz"
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
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
  "$@")

echo "[INFO] Training $TASK with $NUM_ENVS envs on $MOTION_FILE, wall $WALL"
if $DRY_RUN; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
exec "${CMD[@]}"
