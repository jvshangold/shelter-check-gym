import copy
import sys
from dataclasses import dataclass
from typing import Dict

import gymnasium as gym
from gymnasium import Space, spaces

from .render import build_graph
from .state import WorldState

sys.path.append("subsidiary_stock_comp/formalizations/_target/subsidiary_stock_comp_tax_rules")
sys.path.append("subsidiary_stock_comp/formalizations/_build/libcatala/python")

from python import SubsidiaryStockCompTaxModel as TaxModel
from python import catala_runtime


FORM_SUBCORP = 0
BUY_STOCK = 1
COMPENSATE_EMPLOYEES = 2
LIQUIDATE_CORP = 3


@dataclass
class TaxComputation:
    ordinary_deduction: float
    capital_loss: float
    total_tax_advantage: float


class TaxEnv(gym.Env):
    def __init__(
        self,
        MAX_CORPORATIONS=5,
        MAX_STOCKS=8,
        MAX_STEPS=10,
        CASH_AMOUNTS=None,
        FORMATION_SPLITS=None,
        EXTRA_CORPORATION_PENALTY=0.01,
        SUCCESS_ADVANTAGE=198.0,
        SUCCESS_BONUS=1.0,
        PRINT_INVALID_ACTIONS=False,
    ):
        super().__init__()

        self.observation_space: Space = spaces.Dict({})
        self.cash_amounts = CASH_AMOUNTS or [100.0, 99.0, 79.0, 21.0, 1.0]
        self.formation_splits = FORMATION_SPLITS or [
            (79.0, 21.0),
            (80.0, 20.0),
            (50.0, 50.0),
        ]
        self.action_space: Space = spaces.MultiDiscrete([
            4,
            MAX_CORPORATIONS,
            MAX_CORPORATIONS,
            MAX_STOCKS,
            max(len(self.cash_amounts), len(self.formation_splits)),
        ])

        self.max_corporations = MAX_CORPORATIONS
        self.max_stocks = MAX_STOCKS
        self.max_steps = MAX_STEPS
        self.extra_corporation_penalty = EXTRA_CORPORATION_PENALTY
        self.success_advantage = SUCCESS_ADVANTAGE
        self.success_bonus = SUCCESS_BONUS
        self.print_invalid_actions = PRINT_INVALID_ACTIONS
        self.use_action_masks = True
        self.action_reward_penalty = 0.0

        self.state = WorldState.initial_state()
        self.idx_to_corporation: Dict[int, str] = {}
        self.idx_to_stock: Dict[int, str] = {}
        self.prev_normalized_tax_advantage = 0.0
        self.steps = 0
        self._refresh_indices()

    def step(self, action):
        action_type, corp_a_idx, corp_b_idx, stock_idx, amount_idx = action

        terminated = False
        truncated = False
        invalid_action = False
        previous_state = copy.deepcopy(self.state)
        previous_normalized_tax_advantage = self.prev_normalized_tax_advantage

        try:
            if action_type == FORM_SUBCORP:
                self._form_subcorp(corp_a_idx, corp_b_idx, amount_idx)
            elif action_type == BUY_STOCK:
                self._buy_stock(corp_a_idx, corp_b_idx, amount_idx)
            elif action_type == COMPENSATE_EMPLOYEES:
                self._compensate_employees(corp_a_idx, corp_b_idx, stock_idx, amount_idx)
            elif action_type == LIQUIDATE_CORP:
                self._liquidate_corp(corp_a_idx)
            else:
                raise ValueError("Unknown action type")

        except Exception as e:
            if self.print_invalid_actions:
                print("Action:", action)
                print("Exception:", e)
            self.state = previous_state
            self.prev_normalized_tax_advantage = previous_normalized_tax_advantage
            invalid_action = True

        tax_computation = self.compute_tax()
        current_advantage = tax_computation.total_tax_advantage
        current_normalized_advantage = self.normalize_tax_advantage(current_advantage)
        reward = (
            current_normalized_advantage
            - self.prev_normalized_tax_advantage
            - self.compute_complexity_penalty()
        )
        reward -= self.action_reward_penalty
        self.prev_normalized_tax_advantage = current_normalized_advantage

        if invalid_action:
            reward -= 1.0

        if current_advantage >= self.success_advantage and not invalid_action:
            terminated = True
            reward += self.success_bonus

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        self._refresh_indices()
        info = {
            "invalid_action": invalid_action,
            "tax_advantage": current_advantage,
            "raw_tax_advantage": current_advantage,
            "normalized_tax_advantage": current_normalized_advantage,
            "ordinary_deductions": tax_computation.ordinary_deduction,
            "capital_losses": tax_computation.capital_loss,
            "tax_computation": tax_computation,
        }
        return self.get_observation(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = WorldState.initial_state()
        self.prev_normalized_tax_advantage = 0.0
        self.steps = 0
        self._refresh_indices()
        return self.get_observation(), {}

    def get_observation(self):
        return build_graph(self.state)

    def compute_tax_advantage(self) -> float:
        return self.compute_tax().total_tax_advantage

    def compute_tax(self) -> TaxComputation:
        ordinary_deduction = self._compute_catala_tax(
            stock_compensation_fair_market_value=(
                self.state.ledger.total_ordinary_deductions
            ),
            shareholder_stock_basis=0.0,
            liquidation_distribution_fair_market_value=0.0,
            shareholder_ownership_percent=100.0,
        ).ordinary_deduction

        capital_loss = 0.0
        for event in self.state.ledger.liquidation_events:
            capital_loss += self._compute_catala_tax(
                stock_compensation_fair_market_value=0.0,
                shareholder_stock_basis=event.stock_basis,
                liquidation_distribution_fair_market_value=event.distribution_fmv,
                shareholder_ownership_percent=event.ownership_percent,
            ).capital_loss

        return TaxComputation(
            ordinary_deduction=ordinary_deduction,
            capital_loss=capital_loss,
            total_tax_advantage=ordinary_deduction + capital_loss,
        )

    def normalize_tax_advantage(self, tax_advantage: float) -> float:
        if self.success_advantage <= 0:
            return tax_advantage
        return tax_advantage / self.success_advantage

    def compute_complexity_penalty(self) -> float:
        extra_corps = max(0, len(self.state.corporations) - 3)
        return self.extra_corporation_penalty * extra_corps

    def _form_subcorp(self, contributor_a_idx: int, contributor_b_idx: int, split_idx: int) -> None:
        if len(self.state.corporations) >= self.max_corporations:
            raise ValueError("Maximum number of corporations reached")
        contributor_a = self._get_corporation_id(contributor_a_idx)
        contributor_b = self._get_corporation_id(contributor_b_idx)
        if split_idx >= len(self.formation_splits):
            raise ValueError(f"Unknown formation split index: {split_idx}")
        contribution_a, contribution_b = self.formation_splits[split_idx]
        self.state.form_subcorp(
            contributor_a,
            contributor_b,
            contribution_a,
            contribution_b,
        )
        self._refresh_indices()

    def _buy_stock(self, buyer_idx: int, issuer_idx: int, amount_idx: int) -> None:
        if len(self.state.stock) >= self.max_stocks:
            raise ValueError("Maximum number of stock lots reached")
        buyer_id = self._get_corporation_id(buyer_idx)
        issuer_id = self._get_corporation_id(issuer_idx)
        amount = self._get_cash_amount(amount_idx)
        self.state.buy_stock(buyer_id, issuer_id, amount)
        self._refresh_indices()

    def _compensate_employees(
        self,
        holder_idx: int,
        service_recipient_idx: int,
        stock_idx: int,
        amount_idx: int,
    ) -> None:
        holder_id = self._get_corporation_id(holder_idx)
        service_recipient_id = self._get_corporation_id(service_recipient_idx)
        stock_id = self._get_stock_id(stock_idx)
        amount = self._get_cash_amount(amount_idx)
        self.state.compensate_employees(
            holder_id,
            service_recipient_id,
            stock_id,
            amount,
        )

    def _liquidate_corp(self, corporation_idx: int) -> None:
        corporation_id = self._get_corporation_id(corporation_idx)
        self.state.liquidate_corp(corporation_id)
        self._refresh_indices()

    def _get_corporation_id(self, idx: int) -> str:
        if idx not in self.idx_to_corporation:
            raise ValueError(f"Unknown corporation index: {idx}")
        return self.idx_to_corporation[idx]

    def _get_stock_id(self, idx: int) -> str:
        if idx not in self.idx_to_stock:
            raise ValueError(f"Unknown stock index: {idx}")
        return self.idx_to_stock[idx]

    def _get_cash_amount(self, idx: int) -> float:
        if idx >= len(self.cash_amounts):
            raise ValueError(f"Unknown cash amount index: {idx}")
        return self.cash_amounts[idx]

    def _compute_catala_tax(
        self,
        stock_compensation_fair_market_value: float,
        shareholder_stock_basis: float,
        liquidation_distribution_fair_market_value: float,
        shareholder_ownership_percent: float,
    ) -> TaxComputation:
        transaction = TaxModel.StockCompLiquidationInput(
            stock_compensation_fair_market_value=self._money(
                stock_compensation_fair_market_value,
            ),
            shareholder_stock_basis=self._money(shareholder_stock_basis),
            liquidation_distribution_fair_market_value=self._money(
                liquidation_distribution_fair_market_value,
            ),
            shareholder_ownership_percent=self._decimal_percent(
                shareholder_ownership_percent,
            ),
        )

        result = TaxModel.subsidiary_stock_comp_computation(
            TaxModel.SubsidiaryStockCompComputationIn(
                transaction_in=transaction,
            )
        )

        return TaxComputation(
            ordinary_deduction=self._money_to_float(result.ordinary_deduction),
            capital_loss=self._money_to_float(result.capital_loss),
            total_tax_advantage=self._money_to_float(result.total_tax_advantage),
        )

    def _money(self, amount: float):
        return catala_runtime.Money(catala_runtime.Integer(round(amount * 100)))

    def _money_to_float(self, money) -> float:
        return float(money.value.value) / 100.0

    def _decimal_percent(self, percent: float):
        basis_points = round(percent * 100)
        return catala_runtime.Decimal(f"{basis_points}/10000")

    def _refresh_indices(self) -> None:
        self.idx_to_corporation = {
            i: corporation_id
            for i, corporation_id in enumerate(self.state.corporations.keys())
        }
        self.idx_to_stock = {
            i: stock_id
            for i, stock_id in enumerate(self.state.stock.keys())
        }
