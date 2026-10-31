# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""rsl_rl model with a 1D temporal convolution encoder over a window of future reference frames.

The model reads its observation groups as the MLP model does (flat [B, D] groups, concatenated and normalised
together), except that one of them, ``future_group``, is a frame-major window of ``num_frames`` future frames
([B, num_frames * C], as the climbing task's future terms). That group is always put last, reshaped to
[B, C, num_frames] (channels = per-frame features, length = time) and encoded by a stack of Conv1d layers; the
flattened conv features are concatenated with the other (normalised) groups and fed to the MLP head.

Exported models (ONNX / JIT) take a single flat input: the model's other observation groups, in their
``obs_groups`` order, followed by the future group.

Config (``RslRlModelCfg.cnn_cfg``, with ``class_name`` set to this class)::

    {"future_group": "actor_future", "num_frames": 50,
     "channels": (32, 32, 32), "kernel_size": (5, 5, 3), "stride": (2, 2, 1)}
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation


def _per_layer(value: int | tuple[int, ...] | list[int], num_layers: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * num_layers
    value = tuple(int(v) for v in value)
    if len(value) != num_layers:
        raise ValueError(f"cnn_cfg.{name} has {len(value)} entries for {num_layers} conv layers")
    return value


class FutureConv1dEncoder(nn.Module):
    """Conv1d stack over [B, C, T], flattened (the time positions are kept: no global pooling)."""

    def __init__(
        self,
        in_channels: int,
        num_frames: int,
        channels: tuple[int, ...],
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        activation: str,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_channels
        for c, k, s in zip(channels, kernel_size, stride, strict=True):
            layers += [nn.Conv1d(last, c, kernel_size=k, stride=s), resolve_nn_activation(activation)]
            last = c
        layers.append(nn.Flatten(start_dim=1))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            out = self.net(torch.zeros(1, in_channels, num_frames))
        if out.shape[-1] == 0:
            raise ValueError(f"the conv stack reduces the {num_frames}-frame window to nothing")
        self.output_dim = int(out.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _split_future(x: torch.Tensor, future_dim: int, num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """x: normalised flat obs [B, D] ending with the frame-major future window [B, future_dim].
    Returns the other observations [B, D - future_dim] and the window as [B, C, T]."""
    split = x.shape[1] - future_dim
    future = x[:, split:].reshape(x.shape[0], num_frames, future_dim // num_frames).transpose(1, 2)
    return x[:, :split], future


class FutureConv1dModel(MLPModel):
    """MLP model whose future-window observation group goes through a Conv1d encoder first."""

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
        if cnn_cfg is None:
            raise ValueError("FutureConv1dModel needs a cnn_cfg (future_group, num_frames, conv layers)")
        cfg = dict(cnn_cfg)
        self.future_group: str = cfg.pop("future_group")
        self.num_frames = int(cfg.pop("num_frames"))
        channels = tuple(int(c) for c in cfg.pop("channels", (32, 32, 32)))
        kernel_size = _per_layer(cfg.pop("kernel_size", 5), len(channels), "kernel_size")
        stride = _per_layer(cfg.pop("stride", 1), len(channels), "stride")
        if cfg:
            raise ValueError(f"unknown cnn_cfg keys for FutureConv1dModel: {sorted(cfg)}")

        if self.future_group not in obs_groups[obs_set]:
            raise ValueError(f"future group {self.future_group!r} is not in the {obs_set!r} observation set")
        future_shape = obs[self.future_group].shape
        if len(future_shape) != 2 or future_shape[-1] % self.num_frames:
            raise ValueError(
                f"future group {self.future_group!r} has shape {tuple(future_shape)}, expected [B, {self.num_frames} * C]"
            )
        self.future_dim = int(future_shape[-1])
        encoder = FutureConv1dEncoder(
            self.future_dim // self.num_frames, self.num_frames, channels, kernel_size, stride, activation
        )
        self.encoder_output_dim = encoder.output_dim

        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        self.future_encoder = encoder

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        groups, obs_dim = super()._get_obs_dim(obs, obs_groups, obs_set)
        # The future window goes last, so the flat (normalised) input is [other groups | future window].
        groups = [g for g in groups if g != self.future_group] + [self.future_group]
        return groups, obs_dim

    def _get_latent_dim(self) -> int:
        return self.obs_dim - self.future_dim + self.encoder_output_dim

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        x = super().get_latent(obs, masks, hidden_state)  # concatenated and normalised
        rest, future = _split_future(x, self.future_dim, self.num_frames)
        return torch.cat([rest, self.future_encoder(future)], dim=-1)

    def as_jit(self) -> nn.Module:
        return _ExportFutureConv1dModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _ExportFutureConv1dModel(self)


class _ExportFutureConv1dModel(nn.Module):
    """Deterministic, single flat input export (ONNX / JIT) of a FutureConv1dModel."""

    is_recurrent: bool = False

    def __init__(self, model: FutureConv1dModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.future_encoder = copy.deepcopy(model.future_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim
        self.future_dim = model.future_dim
        self.num_frames = model.num_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        rest, future = _split_future(x, self.future_dim, self.num_frames)
        latent = torch.cat([rest, self.future_encoder(future)], dim=-1)
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
