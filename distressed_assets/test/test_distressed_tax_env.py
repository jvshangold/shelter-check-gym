import pytest
import torch

from distressed_assets.tax_env.render import build_graph
from distressed_assets.tax_env.state import (
    WorldState,
    TaxResidence,
    OwnerType,
    AssetKind,
)
from distressed_assets.tax_env.model import GNN
from distressed_assets.tax_env.env import MOVE_ASSET, TaxEnv
from distressed_assets.tax_env.hard_env import HardTaxEnv
from distressed_assets.rl.train_agent import make_asset_mask, make_trust_mask


def initialize():
    return WorldState.initial_state()


def test_initial_state():
    state = initialize()

    assert state.taxpayer_id == "T"
    assert "T" in state.individuals
    assert "root_trust" in state.trusts
    assert state.root_trust_id == "root_trust"

    assert "t_cash" in state.assets
    cash = state.assets["t_cash"]
    assert cash.kind == AssetKind.CASH
    assert cash.owner_type == OwnerType.INDIVIDUAL
    assert cash.owner_id == "T"


def test_add_subtrust():
    state = initialize()

    state.add_trust("sub_trust_1", parent_trust_id="root_trust")

    assert "sub_trust_1" in state.trusts
    assert state.trusts["sub_trust_1"].parent_trust_id == "root_trust"
    assert state.trusts["sub_trust_1"].section_678_power_holder_id is None


def test_add_asset():
    state = initialize()

    state.add_individual("FP_0", TaxResidence.FOREIGN)

    state.add_asset(
        asset_id="distressed_asset_0",
        kind=AssetKind.PROPERTY,
        basis=100.0,
        fair_market_value=20.0,
        owner_type=OwnerType.INDIVIDUAL,
        owner_id="FP_0",
    )

    asset = state.assets["distressed_asset_0"]
    assert asset.kind == AssetKind.PROPERTY
    assert asset.basis == 100.0
    assert asset.fair_market_value == 20.0
    assert asset.owner_type == OwnerType.INDIVIDUAL
    assert asset.owner_id == "FP_0"


def test_transfer_asset():
    state = initialize()

    state.add_individual("FP_0", TaxResidence.FOREIGN)
    state.add_trust("sub_trust_1", parent_trust_id="root_trust")

    state.add_asset(
        asset_id="distressed_asset_0",
        kind=AssetKind.PROPERTY,
        basis=100.0,
        fair_market_value=20.0,
        owner_type=OwnerType.INDIVIDUAL,
        owner_id="FP_0",
    )

    state.transfer_asset(
        asset_id="distressed_asset_0",
        new_owner_type=OwnerType.TRUST,
        new_owner_id="sub_trust_1",
    )

    asset = state.assets["distressed_asset_0"]
    assert asset.owner_type == OwnerType.TRUST
    assert asset.owner_id == "sub_trust_1"


def test_render():
    state = initialize()

    state.add_individual("FP_0", TaxResidence.FOREIGN)
    state.add_trust("sub_trust_1", parent_trust_id="root_trust")

    state.add_asset(
        asset_id="distressed_asset_0",
        kind=AssetKind.PROPERTY,
        basis=100.0,
        fair_market_value=20.0,
        owner_type=OwnerType.TRUST,
        owner_id="sub_trust_1",
    )

    data = build_graph(state)

    assert "trust" in data.node_types
    assert "asset" in data.node_types
    assert "individual" in data.node_types

    assert data["trust"].x.shape == (2, 3)
    assert data["asset"].x.shape == (2, 5)
    assert data["individual"].x.shape == (2, 2)

    expected_edge_types = {
        ("trust", "has_subtrust", "trust"),
        ("trust", "is_subtrust_of", "trust"),
        ("individual", "vesting_power", "trust"),
        ("trust", "controlled_by", "individual"),
        ("trust", "owns", "asset"),
        ("asset", "owned_by_trust", "trust"),
        ("individual", "owns", "asset"),
        ("asset", "owned_by_individual", "individual"),
    }

    assert expected_edge_types.issubset(set(data.edge_types))

    assert data["trust", "has_subtrust", "trust"].edge_index.shape == (2, 1)
    assert data["trust", "is_subtrust_of", "trust"].edge_index.shape == (2, 1)
    assert data["trust", "owns", "asset"].edge_index.shape == (2, 1)
    assert data["asset", "owned_by_trust", "trust"].edge_index.shape == (2, 1)
    assert data["individual", "owns", "asset"].edge_index.shape == (2, 1)
    assert data["asset", "owned_by_individual", "individual"].edge_index.shape == (2, 1)


def test_GNN():
    state = initialize()

    state.add_individual("FP_0", TaxResidence.FOREIGN)
    state.add_trust("sub_trust_1", parent_trust_id="root_trust")

    state.add_asset(
        asset_id="distressed_asset_0",
        kind=AssetKind.PROPERTY,
        basis=100.0,
        fair_market_value=20.0,
        owner_type=OwnerType.TRUST,
        owner_id="sub_trust_1",
    )

    data = build_graph(state)

    model = GNN(hidden_channels=16, out_channels=8)
    out = model(data)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 24)
    assert torch.isfinite(out).all()


def test_env_reset():
    env = TaxEnv()

    obs, info = env.reset()

    assert info == {}
    assert env.steps == 0
    assert env.prev_tax_advantage == 0.0
    assert obs is not None


def test_hard_env_reset_has_distressed_asset_and_decoys():
    env = HardTaxEnv()
    obs, info = env.reset()

    assert info == {}
    assert obs is not None
    assert env.idx_to_individual == {
        0: "T",
        1: "FP_0",
        2: "FP_1",
        3: "FP_2",
    }
    assert env.idx_to_trust == {0: "root_trust"}
    assert env.idx_to_asset == {
        0: "t_cash",
        1: "distressed_property_0",
        2: "neutral_property_1",
        3: "gain_property_2",
    }
    assert env.state.assets["distressed_property_0"].basis == 200.0
    assert env.state.assets["distressed_property_0"].fair_market_value == 40.0
    assert env.state.assets["neutral_property_1"].basis == 100.0
    assert env.state.assets["neutral_property_1"].fair_market_value == 100.0
    assert env.state.assets["gain_property_2"].basis == 40.0
    assert env.state.assets["gain_property_2"].fair_market_value == 100.0


def test_hard_env_known_sequence_can_still_succeed():
    env = HardTaxEnv()
    env.reset()

    env.step([0, 0, 0, 0])  # make subtrust
    env.step([1, 1, 1, 0])  # move distressed property to subtrust
    env.step([3, 1, 0, 0])  # give vesting power to T
    _, reward, terminated, _, info = env.step([2, 0, 1, 0])  # sell distressed property

    assert info["raw_tax_savings"] > 0.0
    assert info["tax_advantage"] == info["raw_tax_savings"]
    assert reward > 0.0
    assert terminated


def test_hard_env_decoy_assets_do_not_succeed():
    for asset_idx in (2, 3):
        env = HardTaxEnv()
        env.reset()

        env.step([0, 0, 0, 0])  # make subtrust
        env.step([1, 1, asset_idx, 0])  # move decoy property to subtrust
        env.step([3, 1, 0, 0])  # give vesting power to T
        _, _, terminated, _, info = env.step([2, 0, asset_idx, 0])  # sell decoy

        assert info["raw_tax_savings"] <= 0.0
        assert info["tax_advantage"] <= 0.0
        assert not terminated


def test_env_make_subtrust():
    env = TaxEnv()
    env.reset()

    obs, reward, terminated, truncated, info = env.step([0, 0, 0, 0])

    assert len(env.state.trusts) == 2
    assert "sub_trust_1" in env.state.trusts
    assert not info["invalid_action"]


def test_env_move_asset_sell_and_vesting_power():
    env = TaxEnv()
    env.reset()

    env.state.add_individual("FP_0", TaxResidence.FOREIGN)
    env.state.add_asset(
        asset_id="distressed_asset_0",
        kind=AssetKind.PROPERTY,
        basis=100.0,
        fair_market_value=20.0,
        owner_type=OwnerType.INDIVIDUAL,
        owner_id="FP_0",
    )
    env._refresh_indices()

    env.step([0, 0, 0, 0])  # make subtrust
    env.step([1, 1, 1, 0])  # move asset index 1 to trust index 1
    env.step([3, 1, 0, 0])  # give vesting power to T
    env.step([2, 0, 1, 0])  # sell asset index 1

    asset = env.state.assets["distressed_asset_0"]
    trust = env.state.trusts["sub_trust_1"]

    assert asset.is_sold
    assert trust.section_678_power_holder_id == "T"
    assert env.compute_savings() > 0.0
    assert env.compute_tax_advantage() > 0.0


def test_nested_trust_tax_savings_are_rewarded_fiscally():
    env = TaxEnv()
    env.reset()

    env.state.add_individual("FP_0", TaxResidence.FOREIGN)
    env.state.add_asset(
        asset_id="distressed_asset_0",
        kind=AssetKind.PROPERTY,
        basis=100.0,
        fair_market_value=20.0,
        owner_type=OwnerType.INDIVIDUAL,
        owner_id="FP_0",
    )
    env._refresh_indices()

    env.step([0, 0, 0, 0])  # make direct subtrust
    env.step([0, 1, 0, 0])  # make nested subtrust
    env.step([1, 2, 1, 0])  # move asset into nested subtrust
    env.step([3, 2, 0, 0])  # give vesting power to T
    _, reward, terminated, _, info = env.step([2, 0, 1, 0])  # sell asset

    assert info["raw_tax_savings"] > 0.0
    assert info["tax_advantage"] == info["raw_tax_savings"]
    assert reward > 0.0
    assert terminated


def test_env_rejects_moving_cash_asset():
    env = TaxEnv()
    env.reset()

    obs, reward, terminated, truncated, info = env.step([0, 0, 0, 0])
    assert not info["invalid_action"]

    obs, reward, terminated, truncated, info = env.step([MOVE_ASSET, 1, 0, 0])

    assert info["invalid_action"]
    assert env.state.assets["t_cash"].owner_type == OwnerType.INDIVIDUAL
    assert env.state.assets["t_cash"].owner_id == "T"


def test_move_asset_masks_exclude_cash_and_same_destination():
    env = TaxEnv()
    env.reset()

    env.state.add_individual("FP_0", TaxResidence.FOREIGN)
    env.state.add_asset(
        asset_id="distressed_asset_0",
        kind=AssetKind.PROPERTY,
        basis=100.0,
        fair_market_value=20.0,
        owner_type=OwnerType.INDIVIDUAL,
        owner_id="FP_0",
    )
    env._refresh_indices()

    env.step([0, 0, 0, 0])  # make subtrust

    asset_mask = make_asset_mask(env, MOVE_ASSET, torch.device("cpu"))
    assert not asset_mask[0].item()  # t_cash
    assert asset_mask[1].item()  # distressed property

    env.step([MOVE_ASSET, 1, 1, 0])

    same_destination_mask = make_trust_mask(
        env,
        MOVE_ASSET,
        torch.device("cpu"),
        asset_idx=1,
    )
    assert same_destination_mask[0].item()  # root_trust remains a different destination
    assert not same_destination_mask[1].item()  # current trust is masked out
