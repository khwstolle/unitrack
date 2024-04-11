r"""
Tests for ``unitrack.memory``.
"""
from __future__ import annotations

import typing as T

import pytest
import torch
from tensordict import TensorDict

import unitrack as ut

KEY_STATE_FLOAT: T.Final = "float_state"
KEY_STATE_BOOL: T.Final = "bool_state"
KEY_STATE_INT: T.Final = "int_state"

STATES_USER: T.Final = [KEY_STATE_FLOAT, KEY_STATE_BOOL, KEY_STATE_INT]
STATES_INTERNAL: T.Final = [
    ut.consts.KEY_FRAME,
    ut.consts.KEY_START,
    ut.consts.KEY_ID,
    ut.consts.KEY_ACTIVE,
]


@pytest.fixture
def memory() -> ut.TrackletMemory:
    mem = ut.TrackletMemory(
        states={
            KEY_STATE_FLOAT: ut.states.Value(torch.float32),
            KEY_STATE_BOOL: ut.states.Value(torch.bool),
            KEY_STATE_INT: ut.states.Value(torch.int),
        },
        auto_reset=False,
    )

    assert len(mem) == 0, "Memory should be empty at initialization."
    return mem


def test_memory_state(memory: ut.TrackletMemory) -> None:
    r"""
    Tests whether the ``state_dict`` of the memory is correctly initialized.
    """

    for state in STATES_USER + STATES_INTERNAL:
        assert state in memory.states, f"State {state} not found in memory states."


def test_memory_read(memory: ut.TrackletMemory) -> None:
    r"""
    Tests whether the memory resets when the current frame is larger than the stored
    frame.
    """

    assert memory.frame == -1.0, "Memory frame should be 0 at initialization."
    assert memory.write_count == 0, "memory.write_count should be 0 at initialization."

    unit_delta = torch.tensor(1.0 / memory.fps)

    # Test reading at frames past the current index 0
    ctx, obs = memory.read(0)

    assert ctx[ut.consts.KEY_FRAME].item() == 0
    assert torch.allclose(ctx[ut.consts.KEY_DELTA], unit_delta)

    ctx, obs = memory.read(1)

    assert ctx[ut.consts.KEY_FRAME].item() == 1
    assert torch.allclose(ctx[ut.consts.KEY_DELTA], 2 * unit_delta)
    ctx, obs = memory.read(5)

    assert ctx[ut.consts.KEY_FRAME].item() == 5
    assert torch.allclose(ctx[ut.consts.KEY_DELTA], 6 * unit_delta)

    with pytest.raises(IndexError, match="negative frame index"):
        memory.read(-1)

    assert memory.frame == -1.0
    assert memory.write_count == 0

    # Test increasing the frame index and reading at subsequent frames
    memory.frame.fill_(10)
    ctx, obs = memory.read(11)
    assert ctx[ut.consts.KEY_FRAME].item() == 11
    assert torch.allclose(ctx[ut.consts.KEY_DELTA], unit_delta)

    assert memory.write_count == 0
    memory.write_count.fill_(5)
    assert memory.write_count == 5
    assert len(memory) == 5

    # Test that reading at negative frame indices raises an error when `auto_reset` is `False`
    setattr(memory, "auto_reset", False)

    assert memory.auto_reset is False
    with pytest.raises(IndexError, match="less than or equal to saved frame"):
        memory.read(9)

    assert memory.frame == 10
    assert memory.write_count == 5

    # Test that reading at previous frames does not raise an error and resets the counter when `auto_reset` is `True`
    setattr(memory, "auto_reset", True)

    ctx, obs = memory.read(6)

    assert memory.frame == -1, "auto reset was triggered"
    assert len(memory) == 0

    assert ctx[ut.consts.KEY_FRAME].item() == 6
    assert torch.allclose(ctx[ut.consts.KEY_DELTA], 7 * unit_delta)


def test_memory_write(memory: ut.TrackletMemory) -> None:
    def _print_ctx_states(frame_num: int, ctx: TensorDict, states: TensorDict) -> None:
        ctx_pretty = {k: v.tolist() for k, v in ctx.to_dict().items()}
        states_pretty = {k: v.tolist() for k, v in states.to_dict().items()}

        print(
            f"-- Frame {frame_num} -- \n- Context = {ctx_pretty}\n- States   = {states_pretty}"
        )

    cur_tracklets = 0
    cur_frame = -1
    cur_writes = 0

    assert memory.frame == cur_frame
    assert memory.tracklet_count == cur_tracklets
    assert len(memory) == cur_writes

    # -- READ FRAME 0 --

    cur_frame = 0
    ctx, obs = memory.read(cur_frame)
    _print_ctx_states(cur_frame, ctx, obs)
    assert obs.batch_size[0] == cur_tracklets

    ctx[ut.consts.KEY_FRAME].fill_(cur_frame)
    new = TensorDict(
        {
            KEY_STATE_FLOAT: torch.tensor([4.0, 5.0, 6.0]),
            KEY_STATE_BOOL: torch.tensor([False, True, False]),
            KEY_STATE_INT: torch.tensor([4, 5, 6]),
            ut.consts.KEY_INDEX: torch.tensor([0, 2, 1], dtype=torch.long),
        },
        batch_size=[3],
    )
    cur_tracklets += 3
    new_clone = new.clone()
    ass = memory.write(ctx, obs, new)
    cur_writes += 1

    for k in list(new.keys(leaves_only=True, include_nested=True)):
        v1 = new[k]
        v2 = new_clone[k]
        assert torch.allclose(v1, v2), "Values should not be modified in-place."

    print(f"Assignment 0->1: {ass}")

    assert ass.tolist() == [1, 3, 2], "Indices [0, 2, 1] over IDs [1, 2, 3]"
    assert memory.frame == cur_frame
    assert memory.tracklet_count == cur_tracklets
    assert len(memory) == cur_writes

    # -- READ FRAME 1 --

    cur_frame = 1
    ctx, obs = memory.read(cur_frame)
    _print_ctx_states(cur_frame, ctx, obs)
    assert obs.batch_size[0] == cur_tracklets

    ctx[ut.consts.KEY_FRAME].fill_(cur_frame)
    new = TensorDict(
        {
            KEY_STATE_FLOAT: torch.tensor([1e-6, 1e-9]),
            KEY_STATE_BOOL: torch.tensor([False, True]),
            KEY_STATE_INT: torch.tensor([40, 50]),
            ut.consts.KEY_INDEX: torch.tensor([2, 1], dtype=torch.long),
        },
        batch_size=[2],
    )
    cur_tracklets += 2

    obs[ut.consts.KEY_INDEX][1] = 0  # Assign tracklet [1] to detection [0]
    obs[ut.consts.KEY_ACTIVE][2] = False  # Deactivate tracklet [2]

    ass = memory.write(ctx, obs, new)
    cur_writes += 1

    assert ass.tolist() == [2, 5, 4], "States[1].id = 2, Obs[1].id = 5"
    assert memory.frame == cur_frame
    assert memory.tracklet_count == cur_tracklets
    assert len(memory) == cur_writes

    print(f"Assignment 1->2: {ass}")

    # skip frame 3

    # -- READ FRAME 4 --

    ctx, obs = memory.read(4)
    _print_ctx_states(4, ctx, obs)
    assert obs.batch_size[0] == cur_tracklets
    assert obs[ut.consts.KEY_FRAME].tolist() == [
        0,  # not assigned in previous frame
        1,
        0,  # deactivated
        1,
        1,
    ], "[2] was deactivated and thus not updated"
    assert obs[ut.consts.KEY_ID].tolist() == [1, 2, 3, 4, 5]

    assert memory.frame == cur_frame, "Memory frame not affected by read operation."
    assert memory.tracklet_count == cur_tracklets
    assert len(memory) == cur_writes

    # skip frames 5-7

    cur_frame = 8
    cur_writes += 1
    ctx[ut.consts.KEY_FRAME].fill_(cur_frame)
    obs[ut.consts.KEY_INDEX] = torch.tensor([4, 2, -1, 3, 0], dtype=torch.long)

    new = TensorDict(
        {
            KEY_STATE_FLOAT: torch.tensor([1e16]),
            KEY_STATE_BOOL: torch.tensor([False]),
            KEY_STATE_INT: torch.tensor([1_000_000]),
            ut.consts.KEY_INDEX: torch.tensor([1], dtype=torch.long),
        },
        batch_size=[1],
    )
    cur_tracklets += 1
    ass = memory.write(ctx, obs, new)
    cur_writes += 1

    assert ass.tolist() == [5, 6, 2, 4, 1]
    assert memory.frame == cur_frame
    assert memory.tracklet_count == cur_tracklets
    assert len(memory) == 3

    print(f"Assignment 4->8: {ass}")
