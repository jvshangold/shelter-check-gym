import torch

from barnes_group.tax_env.env import (
    ISSUE_STOCK,
    MAKE_SUBCORPORATION,
    TRANSFER_CASH,
    TRANSFER_STOCK,
    TaxEnv,
)
from barnes_group.tax_env.model import GNN
from barnes_group.tax_env.render import build_graph
from barnes_group.tax_env.state import TaxResidence, WorldState


def initialize():
    return WorldState.initial_state()


def test_initial_state():
    state = initialize()

    assert state.taxpayer_id == "T"
    assert state.cfc_id == "FSub"
    assert state.corporations["T"].tax_residence == TaxResidence.US
    assert state.corporations["FSub"].tax_residence == TaxResidence.FOREIGN
    assert state.corporations["FSub"].parent_id == "T"
    assert state.is_subcorporation_of("FSub", "T")
    assert state.is_cfc("FSub")
    assert state.cash_amount("T") == 100.0
    assert state.cash_amount("FSub") == 100.0
    assert state.ownership_percent("T", "FSub") == 100.0


def test_add_subcorporation():
    state = initialize()

    state.add_subcorporation("T", "DS")

    assert "DS" in state.corporations
    assert state.corporations["DS"].tax_residence == TaxResidence.US
    assert state.corporations["DS"].parent_id == "T"
    assert state.is_subcorporation_of("DS", "T")
    assert [corp.id for corp in state.subsidiaries_of("T")] == ["FSub", "DS"]
    assert state.ownership_percent("T", "DS") == 100.0


def test_barnes_exchange_gives_zero_basis_domestic_stock():
    state = initialize()
    state.add_subcorporation("T", "DS")

    state.transfer_stock("fsub_own_stock", "DS", 100.0)
    state.transfer_cash("FSub", "DS", 100.0)
    stock_id = state.issue_stock("DS", "FSub", 80.0)
    env = TaxEnv()
    env.state = state
    env.recompute_tax()

    ds_stock = state.stock[stock_id]
    assert ds_stock.holder_id == "FSub"
    assert ds_stock.issuer_id == "DS"
    assert ds_stock.percent == 80.0
    assert ds_stock.basis == 0.0
    assert state.ledger.section_956_inclusion == 0.0


def test_adversarial_70_percent_exchange_preserves_cash_basis_for_956():
    state = initialize()
    state.add_subcorporation("T", "DS")

    state.transfer_stock("fsub_own_stock", "DS", 100.0)
    state.transfer_cash("FSub", "DS", 100.0)
    stock_id = state.issue_stock("DS", "FSub", 70.0)
    env = TaxEnv()
    env.state = state
    env.recompute_tax()

    ds_stock = state.stock[stock_id]
    assert ds_stock.basis == 100.0
    assert state.ledger.section_956_inclusion == 100.0
    assert state.ledger.total_inclusion == 100.0
    assert state.ledger.tax_due == 35.0
    assert state.ledger.tax_paid == 35.0


def test_stock_issuance_requires_contributed_property():
    state = initialize()
    state.add_subcorporation("T", "DS")

    try:
        state.issue_stock("DS", "FSub", 80.0)
    except ValueError as exc:
        assert "requires property contributed" in str(exc)
    else:
        raise AssertionError("stock issuance without contributed property should fail")


def test_direct_repatriation_is_baseline_not_success():
    env = TaxEnv()
    env.reset()

    obs, reward, terminated, truncated, info = env.step([
        TRANSFER_CASH,
        1,
        0,
        0,
        3,
    ])

    assert env.state.cash_amount("T") == 165.0
    assert env.state.ledger.direct_cash_inclusion == 100.0
    assert env.state.ledger.section_956_inclusion == 0.0
    assert env.state.ledger.total_inclusion == 100.0
    assert env.compute_tax_advantage() == 0.0
    assert not terminated
    assert not info["invalid_action"]


def test_direct_cash_and_956_basis_share_applicable_earnings_cap():
    state = initialize()
    state.add_subcorporation("T", "DS")

    state.record_direct_repatriation(75.0)
    state.add_stock("DS", "FSub", 100.0, 100.0, stock_id="fsub_owns_ds")
    env = TaxEnv()
    env.state = state
    env.recompute_tax()

    assert state.ledger.direct_cash_inclusion == 75.0
    assert state.ledger.section_956_inclusion == 25.0
    assert state.ledger.total_inclusion == 100.0
    assert state.ledger.tax_due == 35.0
    assert state.ledger.tax_paid == 35.0


def test_domestic_cash_only_moves_from_direct_child_to_parent():
    env = TaxEnv()
    env.reset()

    env.step([MAKE_SUBCORPORATION, 0, 0, 0, 0])

    obs, reward, terminated, truncated, info = env.step([
        TRANSFER_CASH,
        0,
        2,
        0,
        0,
    ])

    assert info["invalid_action"]
    assert not terminated
    assert env.state.cash_amount("T") == 100.0
    assert env.state.cash_amount("DS_1") == 0.0


def test_staged_cash_in_direct_domestic_child_shapes_reward():
    env = TaxEnv()
    env.reset()

    env.step([MAKE_SUBCORPORATION, 0, 0, 0, 0])
    obs, reward, terminated, truncated, info = env.step([
        TRANSFER_CASH,
        1,
        2,
        0,
        3,
    ])

    assert not info["invalid_action"]
    assert not terminated
    assert env.compute_tax_advantage() == -65.0
    assert info["tax_advantage"] == 0.0
    assert info["staged_cash"] == 100.0
    assert info["reward_potential"] == 10.0
    assert reward > 0.0


def test_foreign_contributed_cash_cannot_move_to_taxpayer_without_qualifying_stock():
    env = TaxEnv()
    env.reset()

    env.step([MAKE_SUBCORPORATION, 0, 0, 0, 0])
    env.step([TRANSFER_CASH, 1, 2, 0, 3])
    obs, reward, terminated, truncated, info = env.step([
        TRANSFER_CASH,
        2,
        0,
        0,
        3,
    ])

    assert info["invalid_action"]
    assert not terminated
    assert env.state.cash_amount("T") == 100.0
    assert env.state.cash_amount("DS_1") == 100.0


def test_env_can_execute_barnes_strategy():
    env = TaxEnv()
    env.reset()

    env.step([MAKE_SUBCORPORATION, 0, 0, 0, 0])
    env.step([TRANSFER_STOCK, 0, 2, 1, 4])
    env.step([TRANSFER_CASH, 1, 2, 0, 3])
    env.step([ISSUE_STOCK, 2, 1, 0, 3])
    obs, reward, terminated, truncated, info = env.step([
        TRANSFER_CASH,
        2,
        0,
        0,
        3,
    ])

    assert env.state.cash_amount("T") == 200.0
    assert env.compute_tax_advantage() == 35.0
    assert terminated
    assert not truncated
    assert not info["invalid_action"]
    assert info["section_956_inclusion"] == 0.0


def test_cash_routed_through_fsub_child_is_not_barnes_success():
    env = TaxEnv()
    env.reset()

    env.step([MAKE_SUBCORPORATION, 1, 0, 0, 0])  # make DS_1 as a child of FSub
    env.step([TRANSFER_CASH, 1, 2, 0, 3])  # FSub -> DS_1
    obs, reward, terminated, truncated, info = env.step([
        TRANSFER_CASH,
        2,
        0,
        0,
        3,
    ])

    assert info["invalid_action"]
    assert not terminated
    assert env.state.cash_amount("T") == 100.0
    assert info["tax_advantage"] == 0.0


def test_render_graph():
    state = initialize()
    state.add_subcorporation("T", "DS")

    data = build_graph(state)

    assert "corporation" in data.node_types
    assert "cash" in data.node_types
    assert "stock" in data.node_types
    assert data["corporation"].x.shape == (3, 5)
    assert data["cash"].x.shape == (2, 2)
    assert data["stock"].x.shape == (3, 4)
    assert ("corporation", "has_subsidiary", "corporation") in data.edge_types
    assert data["corporation", "has_subsidiary", "corporation"].edge_index.shape == (2, 2)


def test_gnn():
    state = initialize()
    state.add_subcorporation("T", "DS")
    data = build_graph(state)

    model = GNN(hidden_channels=16, out_channels=8)
    out = model(data)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 24)
    assert torch.isfinite(out).all()
