import torch

from subsidiary_stock_comp.tax_env.env import (
    BUY_STOCK,
    COMPENSATE_EMPLOYEES,
    FORM_SUBCORP,
    LIQUIDATE_CORP,
    TaxEnv,
)
from subsidiary_stock_comp.tax_env.hard_env import HardTaxEnv
from subsidiary_stock_comp.tax_env.model import GNN
from subsidiary_stock_comp.tax_env.render import build_graph
from subsidiary_stock_comp.tax_env.state import TaxResidence, WorldState
from subsidiary_stock_comp.rl.model import PolicyValueNet
from subsidiary_stock_comp.rl.ppo import evaluate_action
from subsidiary_stock_comp.rl.train_agent import (
    action_dict_to_env_action,
    has_available_action,
    make_action_mask,
    make_amount_mask,
    make_corp_a_mask,
    make_corp_b_mask,
    make_stock_mask,
    sample_action,
)


def initialize():
    return WorldState.initial_state()


def test_initial_state():
    state = initialize()

    assert state.taxpayer_id == "P"
    assert state.subsidiary_id == "X"
    assert state.corporations["P"].tax_residence == TaxResidence.US
    assert state.corporations["X"].parent_id == "P"
    assert state.cash_amount("P") == 79.0
    assert state.cash_amount("X") == 21.0
    assert state.ownership_percent("P", "X") == 100.0


def test_form_subcorp_with_p_and_x_cash():
    state = initialize()

    subcorp_id = state.form_subcorp("P", "X", 79.0, 21.0)

    assert subcorp_id == "S_0"
    assert state.cash_amount("P") == 0.0
    assert state.cash_amount("X") == 0.0
    assert state.cash_amount("S_0") == 100.0
    assert state.ownership_percent("P", "S_0") == 79.0
    assert state.ownership_percent("X", "S_0") == 21.0


def test_stock_purchase_and_compensation_gives_p_deduction():
    state = initialize()
    state.form_subcorp("P", "X", 79.0, 21.0)
    stock_id = state.buy_stock("S_0", "P", 100.0)

    state.compensate_employees("S_0", "P", stock_id, 99.0)

    assert state.cash_amount("S_0") == 0.0
    assert state.stock_fmv("S_0", "P") == 1.0
    assert state.ledger.ordinary_deductions["P"] == 99.0
    assert state.ledger.total_tax_advantage == 99.0


def test_compensation_must_leave_stock_for_liquidation():
    state = initialize()
    state.form_subcorp("P", "X", 79.0, 21.0)
    stock_id = state.buy_stock("S_0", "P", 100.0)

    try:
        state.compensate_employees("S_0", "P", stock_id, 100.0)
    except ValueError as exc:
        assert "leave stock value" in str(exc)
    else:
        raise AssertionError("full stock compensation should fail")


def test_liquidation_after_compensation_creates_combined_99_capital_loss():
    state = initialize()
    state.form_subcorp("P", "X", 79.0, 21.0)
    stock_id = state.buy_stock("S_0", "P", 100.0)
    state.compensate_employees("S_0", "P", stock_id, 99.0)

    state.liquidate_corp("S_0")

    assert "S_0" not in state.corporations
    assert state.ledger.ordinary_deductions["P"] == 99.0
    assert state.ledger.capital_losses["P"] == 78.21
    assert state.ledger.capital_losses["X"] == 20.79
    assert state.ledger.total_capital_losses == 99.0
    assert state.ledger.total_tax_advantage == 198.0


def test_80_percent_corporate_shareholder_does_not_get_331_loss():
    state = initialize()
    state.add_cash("X", 79.0, 79.0)
    state.form_subcorp("P", "X", 20.0, 80.0)
    stock_id = state.buy_stock("S_0", "P", 100.0)
    state.compensate_employees("S_0", "P", stock_id, 99.0)

    state.liquidate_corp("S_0")

    assert "X" not in state.ledger.capital_losses
    assert state.ledger.capital_losses["P"] == 19.8
    assert state.ledger.total_capital_losses == 19.8


def test_env_can_execute_stock_compensation_strategy():
    env = TaxEnv()
    env.reset()

    env.step([FORM_SUBCORP, 0, 1, 0, 0])
    env.step([BUY_STOCK, 2, 0, 0, 0])
    p_stock_idx = next(
        idx
        for idx, stock_id in env.idx_to_stock.items()
        if env.state.stock[stock_id].issuer_id == "P"
        and env.state.stock[stock_id].holder_id == "S_0"
    )
    env.step([COMPENSATE_EMPLOYEES, 2, 0, p_stock_idx, 1])
    obs, reward, terminated, truncated, info = env.step([LIQUIDATE_CORP, 2, 0, 0, 0])

    assert not info["invalid_action"]
    assert terminated
    assert not truncated
    assert info["ordinary_deductions"] == 99.0
    assert info["capital_losses"] == 99.0
    assert info["tax_advantage"] == 198.0
    assert info["normalized_tax_advantage"] == 1.0
    assert reward > 1.0
    assert obs is not None


def test_hard_env_uses_base_start_without_action_masks():
    env = HardTaxEnv()
    env.reset()

    assert not env.use_action_masks
    assert env.state.taxpayer_id == "P"
    assert env.state.subsidiary_id == "X"
    assert set(env.state.corporations) == {"P", "X"}
    assert env.state.corporations["P"].tax_residence == TaxResidence.US
    assert env.state.corporations["X"].parent_id == "P"
    assert env.state.cash_amount("P") == 79.0
    assert env.state.cash_amount("X") == 21.0
    assert env.state.ownership_percent("P", "X") == 100.0


def test_hard_env_known_sequence_can_still_succeed():
    env = HardTaxEnv()
    env.reset()

    env.step([FORM_SUBCORP, 0, 1, 0, 0])
    env.step([BUY_STOCK, 2, 0, 0, 0])
    p_stock_idx = next(
        idx
        for idx, stock_id in env.idx_to_stock.items()
        if env.state.stock[stock_id].issuer_id == "P"
        and env.state.stock[stock_id].holder_id == "S_0"
    )
    obs, reward, terminated, truncated, info = env.step(
        [COMPENSATE_EMPLOYEES, 2, 0, p_stock_idx, 1]
    )

    assert not info["invalid_action"]
    assert not terminated
    assert not truncated
    assert info["ordinary_deductions"] == 99.0
    assert obs is not None

    obs, reward, terminated, truncated, info = env.step([LIQUIDATE_CORP, 2, 0, 0, 0])

    assert not info["invalid_action"]
    assert terminated
    assert not truncated
    assert info["ordinary_deductions"] == 99.0
    assert info["capital_losses"] == 99.0
    assert info["tax_advantage"] == 198.0
    assert info["normalized_tax_advantage"] == 1.0
    assert reward > 1.0
    assert obs is not None


def test_render_graph():
    state = initialize()
    state.form_subcorp("P", "X", 79.0, 21.0)
    state.buy_stock("S_0", "P", 100.0)

    data = build_graph(state)

    assert "corporation" in data.node_types
    assert "cash" in data.node_types
    assert "stock" in data.node_types
    assert "stock_market" in data.node_types
    assert data["corporation"].x.shape == (3, 6)
    assert data["cash"].x.shape == (0, 2)
    assert data["stock"].x.shape == (4, 5)
    assert set(data.edge_types) == {
        ("corporation", "has_cash", "cash"),
        ("stock", "held_by", "corporation"),
        ("corporation", "issues_stock", "stock"),
        ("corporation", "has_subcorp", "corporation"),
        ("corporation", "stock_listed_on", "stock_market"),
    }


def test_gnn():
    state = initialize()
    state.form_subcorp("P", "X", 79.0, 21.0)
    state.buy_stock("S_0", "P", 100.0)
    data = build_graph(state)

    model = GNN(hidden_channels=16, out_channels=8)
    out = model(data)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 32)
    assert torch.isfinite(out).all()


def test_rl_masks_follow_stock_compensation_strategy():
    env = TaxEnv()
    env.reset()
    device = torch.device("cpu")

    action_mask = make_action_mask(env, device)
    assert action_mask.tolist() == [True, False, False, False]

    corp_a_mask = make_corp_a_mask(env, FORM_SUBCORP, device)
    assert corp_a_mask[:2].tolist() == [True, False]

    corp_b_mask = make_corp_b_mask(env, FORM_SUBCORP, 0, 0, device)
    assert corp_b_mask[:2].tolist() == [False, True]

    amount_mask = make_amount_mask(env, FORM_SUBCORP, 0, 1, 0, device)
    assert amount_mask.tolist()[:3] == [True, False, False]

    env.step([FORM_SUBCORP, 0, 1, 0, 0])
    action_mask = make_action_mask(env, device)
    assert action_mask.tolist() == [False, True, False, True]
    amount_mask = make_amount_mask(env, BUY_STOCK, 2, 0, 0, device)
    assert amount_mask.tolist()[:3] == [True, False, False]

    env.step([BUY_STOCK, 2, 0, 0, 0])

    action_mask = make_action_mask(env, device)
    assert action_mask[COMPENSATE_EMPLOYEES]
    stock_mask = make_stock_mask(env, COMPENSATE_EMPLOYEES, 2, 0, device)
    assert stock_mask.any()
    amount_mask = make_amount_mask(env, COMPENSATE_EMPLOYEES, 2, 0, 3, device)
    assert amount_mask.tolist()[:2] == [False, True]


def test_rl_detects_no_available_actions_after_partial_dead_end():
    env = TaxEnv()
    env.reset()
    device = torch.device("cpu")

    env.state.cash.clear()
    env.state.stock.clear()
    env._refresh_indices()

    assert not has_available_action(env, device)


def test_rl_sample_and_evaluate_action_smoke():
    env = TaxEnv()
    obs, _ = env.reset()
    device = torch.device("cpu")
    obs = obs.to(device)

    model = PolicyValueNet(
        embed_dim=16,
        num_action_types=4,
        max_corporations=env.max_corporations,
        max_stocks=env.max_stocks,
        num_amounts=max(len(env.cash_amounts), len(env.formation_splits)),
    )

    action, log_prob, entropy, value, masks, used_random = sample_action(
        model=model,
        env=env,
        obs=obs,
        device=device,
        random_action_prob=1.0,
    )
    env_action = action_dict_to_env_action(action)
    next_obs, reward, terminated, truncated, info = env.step(env_action)
    new_log_prob, new_entropy, new_value = evaluate_action(model, obs, action, masks)

    assert not info["invalid_action"]
    assert used_random
    assert torch.isfinite(log_prob)
    assert torch.isfinite(entropy)
    assert torch.isfinite(value).all()
    assert torch.isfinite(new_log_prob)
    assert torch.isfinite(new_entropy)
    assert torch.isfinite(new_value).all()
    assert next_obs is not None
