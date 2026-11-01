"""PPO without a learning signal from the episodes that fail right after their spawn.

An episode that ends by a termination (not a time-out) within ``early_failure_steps`` policy steps of its start
(mdp.EARLY_FAILURE_STEPS, the same "early failure" the samplers and curricula ignore, see mdp.early_failures) is
masked out of the update: its transitions get a zero advantage (no policy gradient) and their stored value as the
return (no new value target), and the advantages are normalised over the other transitions only. The env is reset
as usual on the failure, so it starts again from a newly sampled spawn.

Only the episode's transitions in the current rollout are masked: one that started near the end of the previous
rollout already had its first steps used in that update.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from mjlab_microban.tasks.climbing.mdp import EARLY_FAILURE_STEPS


class EarlyFailureMaskedPPO(PPO):
    def __init__(self, *args, early_failure_steps: int = EARLY_FAILURE_STEPS, **kwargs):
        super().__init__(*args, **kwargs)
        if self.normalize_advantage_per_mini_batch:
            raise ValueError("EarlyFailureMaskedPPO normalises the advantages itself: per-mini-batch normalisation off")
        self.early_failure_steps = int(early_failure_steps)
        st = self.storage
        self.episode_steps = torch.zeros(st.num_envs, dtype=torch.long, device=self.device)
        self.valid = torch.ones(st.num_transitions_per_env, st.num_envs, 1, dtype=torch.bool, device=self.device)
        self.masked_fraction = 0.0

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        step = self.storage.step  # the storage slot of this transition
        super().process_env_step(obs, rewards, dones, extras)
        if step == 0:
            self.valid.fill_(True)
        self.episode_steps += 1
        done = dones.to(self.device).view(-1).bool()
        time_out = extras.get("time_outs")
        failed = done if time_out is None else done & ~time_out.to(self.device).view(-1).bool()
        early = failed & (self.episode_steps <= self.early_failure_steps)
        if torch.any(early):
            # This episode's transitions in the rollout: slots step - episode_steps + 1 .. step.
            first = (step + 1 - self.episode_steps).clamp(min=0)
            slots = torch.arange(self.storage.num_transitions_per_env, device=self.device)[:, None]
            in_episode = (slots >= first[None]) & (slots <= step) & early[None]
            self.valid[..., 0] &= ~in_episode
        self.episode_steps[done] = 0

    def compute_returns(self, obs: TensorDict) -> None:
        # GAE as PPO computes it, without its normalisation over all transitions (done below, over the valid ones).
        self.normalize_advantage_per_mini_batch = True
        try:
            super().compute_returns(obs)
        finally:
            self.normalize_advantage_per_mini_batch = False
        st, valid = self.storage, self.valid
        self.masked_fraction = 1.0 - float(valid.float().mean())
        ref = st.advantages[valid] if int(valid.sum()) > 1 else st.advantages.flatten()
        st.advantages = (st.advantages - ref.mean()) / (ref.std() + 1e-8)
        st.advantages[~valid] = 0.0
        st.returns[~valid] = st.values[~valid]

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        loss_dict["early_failure_masked_frac"] = self.masked_fraction
        return loss_dict
