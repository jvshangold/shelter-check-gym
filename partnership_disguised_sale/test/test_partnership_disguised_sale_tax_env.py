import pytest

from partnership_disguised_sale.tax_env.env import (
    CONTRIBUTE_ASSET,
    CONTRIBUTE_CASH,
    DISTRIBUTE_CASH,
    TAKE_OUT_LOAN,
    TaxEnv,
    TaxModel,
)
from partnership_disguised_sale.tax_env.state import CashSource, OwnerType, WorldState


def test_initial_state():
    state = WorldState.initial_state()

    assert state.taxpayer_id == "T"
    assert set(state.individuals) == {"T", "Buyer"}
    assert set(state.partnerships["P"].partner_ids) == {"T", "Buyer"}
    assert state.capital_account("P", "T") == 0.0
    assert state.capital_account("P", "Buyer") == 0.0
    assert state.individual_cash("Buyer") == 100.0
    assert state.individual_cash("T") == 0.0

    asset = state.assets["appreciated_asset"]
    assert asset.owner_type == OwnerType.INDIVIDUAL
    assert asset.owner_id == "T"
    assert asset.basis == 0.0
    assert asset.fair_market_value == 100.0


def test_catala_target_is_loaded():
    assert TaxModel is not None


def test_canonical_tribune_path_state_and_tax_inputs():
    state = WorldState.initial_state()

    state.contribute_asset("appreciated_asset", "T", "P")
    state.contribute_cash("Buyer", "P", 10.0)
    loan_id = state.take_out_loan("P", guarantor_id="T", amount=90.0)
    state.distribute_cash("P", "T", 100.0)

    assert loan_id == "loan_1"
    assert state.individual_cash("T") == 100.0
    assert state.individual_cash("Buyer") == 90.0
    assert state.partnership_cash("P") == 0.0
    assert state.capital_account("P", "T") == 0.0
    assert state.capital_account("P", "Buyer") == 10.0
    assert state.has_completed_economic_sale_to_partnership("T")
    assert state.guaranteed_liability_share("T") == 90.0
    assert state.cash_distributed_to("T") == 100.0
    assert state.debt_financed_distribution_to("T") == 90.0
    assert state.taxpayer_tax_input() == {
        "asset_basis": 0.0,
        "asset_fair_market_value": 100.0,
        "cash_distributed_to_transferor": 100.0,
        "debt_financed_distribution": 90.0,
        "transferor_allocable_liability": 90.0,
        "tax_rate": 0.20,
    }


def test_asset_contribution_without_taxpayer_distribution_has_no_tax_advantage():
    env = TaxEnv()
    env.reset()

    env.step([CONTRIBUTE_ASSET, 0, 0, 0, 0])
    env.step([CONTRIBUTE_CASH, 1, 0, 0, 2])
    env.step([DISTRIBUTE_CASH, 1, 0, 0, 1])

    tax_input = env.state.taxpayer_tax_input()
    tax = env.compute_tax()

    assert tax_input["asset_fair_market_value"] == 0.0
    assert tax_input["cash_distributed_to_transferor"] == 0.0
    assert tax.baseline_gain == 0.0
    assert tax.recognized_gain == 0.0
    assert tax.tax_savings == 0.0
    assert env.compute_tax_advantage() == 0.0


def test_taxpayer_self_financed_cash_out_without_buyer_economics_has_no_reward_advantage():
    env = TaxEnv()
    env.reset()

    env.step([CONTRIBUTE_ASSET, 0, 0, 0, 0])
    env.step([TAKE_OUT_LOAN, 0, 0, 0, 3])
    _, _, terminated, _, info = env.step([DISTRIBUTE_CASH, 0, 0, 0, 3])

    tax = info["tax_computation"]
    assert tax.baseline_gain == 100.0
    assert tax.recognized_gain == 0.0
    assert tax.deferred_gain == 100.0
    assert info["tax_advantage"] == 0.0
    assert env.compute_tax_advantage() == 0.0
    assert not terminated
    assert not env.state.has_completed_economic_sale_to_partnership("T")


def test_env_reaches_success_threshold_with_four_actions():
    env = TaxEnv()
    env.reset()

    # Individuals: 0 = T, 1 = Buyer. Asset 0 = appreciated_asset.
    # Amount buckets: 0 = 10, 2 = 90, 3 = 100.
    env.step([CONTRIBUTE_ASSET, 0, 0, 0, 0])
    env.step([CONTRIBUTE_CASH, 1, 0, 0, 0])
    env.step([TAKE_OUT_LOAN, 0, 0, 0, 2])
    _, _, terminated, _, info = env.step([DISTRIBUTE_CASH, 0, 0, 0, 3])

    tax = info["tax_computation"]
    assert terminated
    assert info["tax_advantage"] == 18.0
    assert tax.baseline_gain == 100.0
    assert tax.recognized_gain == 10.0
    assert tax.deferred_gain == 90.0
    assert tax.tax_savings == 18.0


def test_wrong_guarantor_does_not_get_liability_exclusion():
    env = TaxEnv()
    env.reset()

    env.step([CONTRIBUTE_ASSET, 0, 0, 0, 0])
    env.step([CONTRIBUTE_CASH, 1, 0, 0, 0])
    env.step([TAKE_OUT_LOAN, 1, 0, 0, 2])
    _, _, terminated, _, info = env.step([DISTRIBUTE_CASH, 0, 0, 0, 3])

    tax = info["tax_computation"]
    assert not terminated
    assert tax.recognized_gain == 100.0
    assert tax.deferred_gain == 0.0


def test_render_graph_uses_forward_edges_only():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from partnership_disguised_sale.tax_env.render import build_graph

    state = WorldState.initial_state()
    state.contribute_asset("appreciated_asset", "T", "P")
    state.contribute_cash("Buyer", "P", 10.0)
    state.take_out_loan("P", guarantor_id="T", amount=90.0)

    data = build_graph(state)

    assert data["individual"].x.shape == (2, 5)
    assert data["partnership"].x.shape == (1, 3)
    assert data["asset"].x.shape == (1, 5)
    assert data["loan"].x.shape == (1, 2)
    assert data["individual", "partner_of", "partnership"].edge_index.shape == (2, 2)
    assert data["partnership", "owns_asset", "asset"].edge_index.shape == (2, 1)
    assert data["individual", "guarantees", "loan"].edge_index.shape == (2, 1)


def test_gnn_forward_shape():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from partnership_disguised_sale.tax_env.model import GNN
    from partnership_disguised_sale.tax_env.render import build_graph

    state = WorldState.initial_state()
    state.contribute_asset("appreciated_asset", "T", "P")
    state.contribute_cash("Buyer", "P", 10.0)
    state.take_out_loan("P", guarantor_id="T", amount=90.0)

    data = build_graph(state)
    model = GNN(hidden_channels=16, out_channels=8)
    out = model(data)

    assert out.shape == (1, 40)
