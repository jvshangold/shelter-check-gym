import pytest

from straddle_abuse.tax_env.state import StraddleLegKind, WorldState


def test_initial_state():
    state = WorldState.initial_state()

    assert state.taxpayer_id == "T"
    assert set(state.individuals) == {"T", "A", "B"}
    assert state.individuals["T"].tax_rate == 0.37
    assert state.individuals["A"].tax_rate == 0.0
    assert state.individuals["B"].tax_rate == 0.0
    assert state.ctf.participant_count == 0
    assert not state.ctf.is_common_trust_fund


def test_common_trust_fund_requires_two_investors():
    state = WorldState.initial_state()

    state.invest("A", 100.0)

    assert state.ctf.participant_count == 1
    assert not state.ctf.is_common_trust_fund
    assert not state.can_enter_straddle()

    with pytest.raises(ValueError, match="at least two investors"):
        state.enter_straddle(300.0)

    state.invest("B", 100.0)

    assert state.ctf.participant_count == 2
    assert state.ctf.is_common_trust_fund
    assert state.can_enter_straddle()


def test_classic_gain_then_taxpayer_entry_then_loss_path():
    state = WorldState.initial_state()

    state.invest("A", 100.0)
    state.invest("B", 100.0)
    straddle_id = state.enter_straddle(300.0)

    gain_item = state.realize_gain(straddle_id, 1.0)

    assert gain_item.amount == 300.0
    assert gain_item.allocations == {"A": 150.0, "B": 150.0}
    assert state.allocated_gain_for("T") == 0.0
    assert state.has_realized_gain_before_taxpayer_invested()

    state.invest("T", 100.0)
    loss_item = state.realize_loss(straddle_id, 1.0)

    assert loss_item.amount == 300.0
    assert loss_item.allocations == {"A": 100.0, "B": 100.0, "T": 100.0}
    assert state.allocated_gain_for("T") == 0.0
    assert state.allocated_loss_for("T") == 100.0
    assert state.taxpayer_catala_input() == {
        "gain_leg_realized": True,
        "loss_leg_realized": True,
        "allocated_gain": 0.0,
        "allocated_loss": 100.0,
        "tax_rate": 0.37,
    }


def test_bad_but_legal_taxpayer_entry_before_gain():
    state = WorldState.initial_state()

    state.invest("A", 100.0)
    state.invest("T", 100.0)
    straddle_id = state.enter_straddle(300.0)
    state.realize_gain(straddle_id, 1.0)

    assert state.allocated_gain_for("T") == 150.0


def test_partial_realization_leaves_remaining_amount():
    state = WorldState.initial_state()

    state.invest("A", 100.0)
    state.invest("B", 100.0)
    straddle_id = state.enter_straddle(300.0)
    gain_item = state.realize_gain(straddle_id, 0.5)

    gain_leg = state.straddle_legs[f"straddle_{straddle_id}_gain"]

    assert gain_item.amount == 150.0
    assert gain_leg.realized_amount == 150.0
    assert gain_leg.remaining_amount == 150.0
    assert not gain_leg.is_fully_realized


def test_multiple_open_straddles_require_explicit_straddle_id():
    state = WorldState.initial_state()

    state.invest("A", 100.0)
    state.invest("B", 100.0)
    first_id = state.enter_straddle(300.0)
    second_id = state.enter_straddle(200.0)

    assert state.unrealized_straddle_ids(StraddleLegKind.GAIN) == [
        first_id,
        second_id,
    ]

    with pytest.raises(ValueError, match="provide a straddle_id"):
        state.realize_gain(fraction=1.0)

    state.realize_gain(second_id, 1.0)

    assert state.allocated_gain_for("A") == 100.0
    assert state.allocated_gain_for("B") == 100.0
    assert state.unrealized_straddle_ids(StraddleLegKind.GAIN) == [first_id]


def test_render_graph_shapes_and_offset_edges():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from straddle_abuse.tax_env.render import build_graph

    state = WorldState.initial_state()

    data = build_graph(state)

    assert data["individual"].x.shape == (3, 6)
    assert data["ctf"].x.shape == (1, 3)
    assert data["leg"].x.shape == (0, 7)

    state.invest("A", 100.0)
    state.invest("B", 100.0)
    state.enter_straddle(300.0)
    state.enter_straddle(200.0)

    data = build_graph(state)

    assert data["leg"].x.shape == (4, 7)
    assert data["leg", "offsets", "leg"].edge_index.shape == (2, 4)
    assert data["individual", "invested_in", "ctf"].edge_index.shape == (2, 2)
    assert data["individual", "not_invested_in", "ctf"].edge_index.shape == (2, 1)


def test_gnn_forward_shape():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    import torch

    from straddle_abuse.tax_env.model import GNN
    from straddle_abuse.tax_env.render import build_graph

    state = WorldState.initial_state()
    state.invest("A", 100.0)
    state.invest("B", 100.0)
    state.enter_straddle(300.0)

    data = build_graph(state)
    model = GNN(hidden_channels=16, out_channels=8)
    out = model(data)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 24)
    assert torch.isfinite(out).all()


def test_env_classic_path_reward_and_masks():
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("gmpy2")

    from straddle_abuse.tax_env.env import (
        ENTER_STRADDLE,
        INVEST,
        REALIZE_GAIN,
        REALIZE_LOSS,
        TaxEnv,
    )

    env = TaxEnv(MAX_STRADDLES=3)
    obs, info = env.reset()

    assert info == {}
    assert obs is not None
    assert env.get_action_mask()[ENTER_STRADDLE] == 0

    env.step([INVEST, 0, 0, 1])
    env.step([INVEST, 0, 0, 2])

    assert env.get_action_mask()[ENTER_STRADDLE] == 1

    env.step([ENTER_STRADDLE, 0, 2, 0])

    assert env.get_straddle_mask(StraddleLegKind.GAIN) == [1, 0, 0]

    env.step([REALIZE_GAIN, 0, 3, 0])
    env.step([INVEST, 0, 3, 0])
    obs, reward, terminated, truncated, info = env.step([REALIZE_LOSS, 0, 3, 0])

    assert obs is not None
    assert reward > 0.0
    assert terminated
    assert not truncated
    assert not info["invalid_action"]
    assert info["tax_advantage"] > 0.0
