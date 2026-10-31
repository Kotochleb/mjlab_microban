# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""rsl_rl model with one Conv1d encoder per temporal observation group (e.g. a past history and a future window).

Like FutureConv1dModel (future_conv_model.py), but any number of the model's observation groups can be temporal
windows, each with its own Conv1d stack. A temporal group is either
  - 3D, [B, T, C] (an mjlab group with ``history_length=T`` and ``flatten_history_dim=False``: oldest frame first,
    terms concatenated per frame), or
  - 2D and frame-major, [B, T * C] (as the climbing task's future terms).
Every group is flattened (frame-major) and concatenated as [plain groups | temporal groups, in ``encoders`` order],
normalised together, then each temporal window is reshaped to [B, C, T] and encoded. Optionally one frame of a
window (``direct_frame``, e.g. -1: the newest frame of a history) also goes to the MLP head as is, so the policy
reads the current state without having to recover it from the conv features.

Exported models (ONNX / JIT) take a single flat input: the plain groups in their ``obs_groups`` order, then the
temporal groups in ``encoders`` order, each flattened frame-major (for a history: oldest frame first).

Config (``RslRlModelCfg.cnn_cfg``, with ``class_name`` set to this class)::

    {"encoders": {
        "actor": {"num_frames": 25, "channels": (64, 64, 32), "kernel_size": (5, 3, 3), "stride": (2, 2, 1),
                  "direct_frame": -1},
        "actor_future": {"num_frames": 50, "channels": (32, 32, 32), "kernel_size": (5, 5, 3), "stride": (2, 2, 1)},
    }}
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState

from mjlab_microban.tasks.climbing.future_conv_model import FutureConv1dEncoder, _per_layer


class TemporalConv1dModel(MLPModel):
    """MLP model whose temporal observation groups each go through a Conv1d encoder first."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, Any] | None = None,
    ) -> None:
        if cnn_cfg is None or not cnn_cfg.get("encoders"):
            raise ValueError("TemporalConv1dModel needs a cnn_cfg with a non-empty 'encoders' dict")
        cfg = dict(cnn_cfg)
        encoder_cfgs: dict[str, dict[str, Any]] = {k: dict(v) for k, v in cfg.pop("encoders").items()}
        if cfg:
            raise ValueError(f"unknown cnn_cfg keys for TemporalConv1dModel: {sorted(cfg)}")

        # Per temporal group: (flat dim, frames, channels, direct frame index or None).
        self.temporal_groups: list[str] = list(encoder_cfgs)
        self.temporal_specs: list[tuple[int, int, int, int | None]] = []
        encoders: dict[str, nn.Module] = {}
        direct_dim = 0
        for group, ecfg in encoder_cfgs.items():
            if group not in obs_groups[obs_set]:
                raise ValueError(f"temporal group {group!r} is not in the {obs_set!r} observation set")
            num_frames = int(ecfg.pop("num_frames"))
            channels = tuple(int(c) for c in ecfg.pop("channels", (32, 32, 32)))
            kernel_size = _per_layer(ecfg.pop("kernel_size", 5), len(channels), f"{group}.kernel_size")
            stride = _per_layer(ecfg.pop("stride", 1), len(channels), f"{group}.stride")
            direct_frame = ecfg.pop("direct_frame", None)
            if ecfg:
                raise ValueError(f"unknown cnn_cfg.encoders[{group!r}] keys: {sorted(ecfg)}")

            shape = tuple(obs[group].shape[1:])
            if len(shape) == 2:
                if shape[0] != num_frames:
                    raise ValueError(f"group {group!r} has {shape[0]} frames, cnn_cfg says {num_frames}")
                in_channels = shape[1]
            elif len(shape) == 1 and shape[0] % num_frames == 0:
                in_channels = shape[0] // num_frames
            else:
                raise ValueError(
                    f"group {group!r} has shape {tuple(obs[group].shape)}, expected [B, {num_frames}, C] "
                    f"or [B, {num_frames} * C]"
                )
            if direct_frame is not None:
                direct_frame = int(direct_frame)
                if not -num_frames <= direct_frame < num_frames:
                    raise ValueError(f"direct_frame {direct_frame} out of range for {num_frames} frames")
                direct_frame %= num_frames
                direct_dim += in_channels
            self.temporal_specs.append((num_frames * in_channels, num_frames, in_channels, direct_frame))
            encoders[group] = FutureConv1dEncoder(
                in_channels, num_frames, channels, kernel_size, stride, activation
            )
        self.temporal_dim = sum(s[0] for s in self.temporal_specs)
        self.latent_extra_dim = sum(e.output_dim for e in encoders.values()) + direct_dim

        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        self.temporal_encoders = nn.ModuleDict(encoders)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active = list(obs_groups[obs_set])
        groups = [g for g in active if g not in self.temporal_groups] + self.temporal_groups
        for g in groups:
            if g not in self.temporal_groups and obs[g].dim() != 2:
                raise ValueError(f"non-temporal group {g!r} must be 2D, got shape {tuple(obs[g].shape)}")
        obs_dim = sum(math.prod(obs[g].shape[1:]) for g in groups)
        return groups, obs_dim

    def _get_latent_dim(self) -> int:
        return self.obs_dim - self.temporal_dim + self.latent_extra_dim

    def _flat_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[g].flatten(start_dim=1) for g in self.obs_groups], dim=-1)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        return _encode(self.obs_normalizer(self._flat_obs(obs)), self.temporal_specs, self.temporal_encoders)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(self._flat_obs(obs))  # type: ignore

    def as_jit(self) -> nn.Module:
        return _ExportTemporalConv1dModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _ExportTemporalConv1dModel(self)


def _encode(
    x: torch.Tensor, specs: list[tuple[int, int, int, int | None]], encoders: nn.ModuleDict
) -> torch.Tensor:
    """x: normalised flat obs [B, D] = [plain | temporal windows (frame-major)]. Returns the MLP latent."""
    split = x.shape[1] - sum(s[0] for s in specs)
    parts = [x[:, :split]]
    direct = []
    start = split
    for (dim, frames, channels, direct_frame), encoder in zip(specs, encoders.values()):
        window = x[:, start : start + dim].reshape(x.shape[0], frames, channels)  # [B, T, C]
        start += dim
        parts.append(encoder(window.transpose(1, 2)))
        if direct_frame is not None:
            direct.append(window[:, direct_frame])
    return torch.cat(parts + direct, dim=-1)


class _ExportTemporalConv1dModel(nn.Module):
    """Deterministic, single flat input export (ONNX / JIT) of a TemporalConv1dModel."""

    is_recurrent: bool = False

    def __init__(self, model: TemporalConv1dModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.temporal_encoders = copy.deepcopy(model.temporal_encoders)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim
        self.temporal_specs = list(model.temporal_specs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = _encode(self.obs_normalizer(x), self.temporal_specs, self.temporal_encoders)
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        pass

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
