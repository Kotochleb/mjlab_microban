# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_microban.tasks.tracking.microban_tracking_env_cfg import make_microban_tracking_env_cfg
from mjlab_microban.tasks.tracking.rl_cfg import MicrobanTrackingRlCfg

register_mjlab_task(
    task_id="Mjlab-Tracking-Microban",
    env_cfg=make_microban_tracking_env_cfg(),
    play_env_cfg=make_microban_tracking_env_cfg(play=True),
    rl_cfg=MicrobanTrackingRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-Microban-No-State-Estimation",
    env_cfg=make_microban_tracking_env_cfg(has_state_estimation=False),
    play_env_cfg=make_microban_tracking_env_cfg(has_state_estimation=False, play=True),
    rl_cfg=MicrobanTrackingRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)
