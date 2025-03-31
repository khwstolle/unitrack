from __future__ import annotations

import typing as T

import torch
import typing_extensions as TX
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from .consts import KEY_ACTIVE, KEY_INDEX
from .debug import check_debug_enabled
from .stages import Stage

__all__ = ["MultiStageTracker", "SelectField"]


class SelectField(nn.Module):
    """
    Select a field from a TensorDict.
    """

    def __init__(self, *keys: str, **keys_mapping: str):
        super().__init__()

        self.in_keys = [k for k in keys] + [k for k in keys_mapping]
        self.out_keys = [k for k in keys] + [v for v in keys_mapping.values()]

    @TX.override
    def forward(
        self, inp: TensorDictBase, tensordict_out: TensorDictBase
    ) -> TensorDictBase:
        for key_in, key_out in zip(self.in_keys, self.out_keys, strict=False):
            tensordict_out = tensordict_out.set(key_out, inp[key_in], inplace=True)
        return tensordict_out


class MultiStageTracker(nn.Module):
    """
    Multi-stage tracker that applies a cascade of stages to a set of detections.
    """

    def __init__(self, fields: T.Sequence[TensorDictModule], stages: T.Sequence[Stage]):
        super().__init__()

        assert len(stages) > 0

        self.fields = nn.ModuleList(fields)
        self.stages = nn.ModuleList(stages)

    def forward(
        self,
        ctx: TensorDictBase,
        obs: TensorDictBase,
        inp: TensorDictBase,
        num: int,
    ) -> tuple[TensorDictBase, TensorDictBase]:
        """
        Perform tracking, returns a tuple of updated observations and the field-values of new tracklets.


        Parameters
        ----------
        ctx: TensorDictBase
            The current state's context
        obs: TensorDictBase
            The current state's observations (i.e. when ``.observe()`` is called on
            each field in memory)
        inp: TensorDictBase
            The state of the next frame, from which new detections are gathered at
            every field.
        num: int
            The amount of detections made. Fields must allocate within a TensorDict
            that enforces ``batch_size=[num]``.


        Returns
        -------
        Tuple[TensorDict, TensorDict]
            Updated observations and field-values of new tracklets.
        """

        if inp.device is None:
            raise ValueError("Missing `device` attribute on inputs")

        # Create a dict of new tracklet candidates by passing the input state to
        # each field
        new_index = torch.arange(num, device=inp.device, dtype=torch.int64)
        new = TensorDict(
            {KEY_INDEX: new_index},
            batch_size=new_index.shape,
            device=inp.device,
        )

        obs = obs.to(device=inp.device)

        for field in self.fields:
            field(inp, tensordict_out=new)

        assert obs.device is not None
        assert new.device is not None

        # Candidates for matching are all active observed tracklets
        active_mask = obs.get(KEY_ACTIVE)
        obs_candidates = obs._get_sub_tensordict(active_mask)
        for stage in self.stages:
            if obs_candidates.batch_size[0] == 0 or new.batch_size[0] == 0:
                break
            obs_candidates, new = stage(ctx, obs_candidates, new)

        obs_unmatched_mask = obs.get(KEY_INDEX) < 0

        if check_debug_enabled():
            print(
                f"TRACKER COMPLETE: remaining {obs_unmatched_mask.int().sum().item()}/{len(obs)} unmatched observations"
            )
        obs[KEY_ACTIVE] = torch.where(obs_unmatched_mask, False, True)
        # obs.set_at_(KEY_ACTIVE, False, obs_unmatched_mask)
        # obs.set_at_(KEY_ACTIVE, True, ~obs_unmatched_mask)

        return obs, new
