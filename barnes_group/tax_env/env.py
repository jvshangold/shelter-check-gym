import copy
import sys
from typing import Dict

import gymnasium as gym
from graphviz import Digraph
from gymnasium import Space, spaces

from .render import build_graph
from .state import TaxResidence, WorldState

sys.path.append("barnes_group/formalizations/_target/barnes_group_tax_rules")
sys.path.append("barnes_group/formalizations/_build/libcatala/python")

from python import BarnesGroupTaxModel as TaxModel
from python import catala_runtime


MAKE_SUBCORPORATION = 0
TRANSFER_CASH = 1
TRANSFER_STOCK = 2
ISSUE_STOCK = 3


class TaxEnv(gym.Env):
    def __init__(
        self,
        MAX_CORPORATIONS=5,
        MAX_STOCKS=8,
        MAX_STEPS=10,
        CASH_AMOUNTS=None,
        STOCK_PERCENTS=None,
        EXTRA_CORPORATION_PENALTY=0.01,
        STEP_PENALTY=0.001,
        CASH_STAGING_REWARD_WEIGHT=0.1,
        SUCCESS_ADVANTAGE=35.0,
        TAX_RATE=0.35,
        INITIAL_T_CASH=100.0,
        INITIAL_FSUB_CASH=100.0,
        PRINT_INVALID_ACTIONS=False,
    ):
        super().__init__()

        self.observation_space: Space = spaces.Dict({})
        self.cash_amounts = CASH_AMOUNTS or [25.0, 50.0, 75.0, 100.0]
        self.stock_percents = STOCK_PERCENTS or [25.0, 50.0, 70.0, 80.0, 100.0]

        self.action_space: Space = spaces.MultiDiscrete([
            4,
            MAX_CORPORATIONS,
            MAX_CORPORATIONS,
            MAX_STOCKS,
            max(len(self.cash_amounts), len(self.stock_percents)),
        ])

        self.max_corporations = MAX_CORPORATIONS
        self.max_stocks = MAX_STOCKS
        self.max_steps = MAX_STEPS
        self.extra_corporation_penalty = EXTRA_CORPORATION_PENALTY
        self.step_penalty = STEP_PENALTY
        self.cash_staging_reward_weight = CASH_STAGING_REWARD_WEIGHT
        self.success_advantage = SUCCESS_ADVANTAGE
        self.tax_rate = TAX_RATE
        self.initial_t_cash = INITIAL_T_CASH
        self.initial_fsub_cash = INITIAL_FSUB_CASH
        self.print_invalid_actions = PRINT_INVALID_ACTIONS

        self.state = WorldState.initial_state(
            t_cash=self.initial_t_cash,
            fsub_cash=self.initial_fsub_cash,
            applicable_earnings=self.initial_fsub_cash,
            tax_rate=self.tax_rate,
        )
        self.idx_to_corporation: Dict[int, str] = {}
        self.idx_to_stock: Dict[int, str] = {}
        self.prev_reward_potential = 0.0
        self.steps = 0
        self._refresh_indices()

    def step(self, action):
        action_type, corp_a_idx, corp_b_idx, stock_idx, amount_idx = action

        terminated = False
        truncated = False
        invalid_action = False
        previous_state = copy.deepcopy(self.state)
        previous_reward_potential = self.prev_reward_potential

        try:
            if action_type == MAKE_SUBCORPORATION:
                self._make_subcorporation(parent_idx=corp_a_idx)
            elif action_type == TRANSFER_CASH:
                self._transfer_cash(
                    from_idx=corp_a_idx,
                    to_idx=corp_b_idx,
                    amount_idx=amount_idx,
                )
            elif action_type == TRANSFER_STOCK:
                self._transfer_stock(
                    stock_idx=stock_idx,
                    to_idx=corp_b_idx,
                    percent_idx=amount_idx,
                )
            elif action_type == ISSUE_STOCK:
                self._issue_stock(
                    issuer_idx=corp_a_idx,
                    recipient_idx=corp_b_idx,
                    percent_idx=amount_idx,
                )
            else:
                raise ValueError("Unknown action type")

            self.recompute_tax()

        except Exception as e:
            if self.print_invalid_actions:
                print("Action:", action)
                print("Exception:", e)
            self.state = previous_state
            self.prev_reward_potential = previous_reward_potential
            self._refresh_indices()
            invalid_action = True

        current_advantage = max(0.0, self.compute_tax_advantage())
        current_reward_potential = self.compute_reward_potential(current_advantage)
        reward = (
            current_reward_potential
            - self.prev_reward_potential
            - self.compute_complexity_penalty()
            - self.step_penalty
        )
        self.prev_reward_potential = current_reward_potential

        if invalid_action:
            reward -= 1.0

        if current_advantage >= self.success_advantage and not invalid_action:
            terminated = True

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        self._refresh_indices()
        info = {
            "invalid_action": invalid_action,
            "tax_advantage": current_advantage,
            "reward_potential": current_reward_potential,
            "staged_cash": self.compute_staged_cash(),
            "direct_cash_inclusion": self.state.ledger.direct_cash_inclusion,
            "section_956_inclusion": self.state.ledger.section_956_inclusion,
            "total_inclusion": self.state.ledger.total_inclusion,
            "tax_due": self.state.ledger.tax_due,
            "tax_paid": self.state.ledger.tax_paid,
        }
        return self.get_observation(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = WorldState.initial_state(
            t_cash=self.initial_t_cash,
            fsub_cash=self.initial_fsub_cash,
            applicable_earnings=self.initial_fsub_cash,
            tax_rate=self.tax_rate,
        )
        self.prev_reward_potential = 0.0
        self.steps = 0
        self._refresh_indices()
        return self.get_observation(), {}

    def get_observation(self):
        return build_graph(self.state)

    def compute_tax_advantage(self) -> float:
        t_gain = self.state.cash_amount("T") - self.initial_t_cash
        return t_gain - self.state.direct_transfer_baseline_net()

    def compute_reward_potential(self, current_advantage: float | None = None) -> float:
        if current_advantage is None:
            current_advantage = max(0.0, self.compute_tax_advantage())

        return (
            current_advantage
            + self.cash_staging_reward_weight * self.compute_staged_cash()
        )

    def compute_staged_cash(self) -> float:
        total = 0.0
        for corporation in self.state.corporations.values():
            if corporation.parent_id != self.state.taxpayer_id:
                continue
            if not corporation.is_domestic:
                continue
            total += self.state.cash_amount(corporation.id)
        return total

    def recompute_tax(self) -> None:
        result = TaxModel.barnes_group_tax_computation(
            TaxModel.BarnesGroupTaxComputationIn(
                direct_repatriated_cash_in=self._money(
                    self.state.ledger.direct_repatriated_cash,
                ),
                domestic_stock_basis_in=self._money(
                    self.state.section_956_us_property_basis(),
                ),
                applicable_earnings_in=self._money(self.state.applicable_earnings),
                tax_rate_in=self._decimal_rate(self.state.tax_rate),
            )
        )

        self.state.ledger.direct_cash_inclusion = self._money_to_float(
            result.direct_cash_inclusion,
        )
        self.state.ledger.section_956_inclusion = self._money_to_float(
            result.section_956_inclusion,
        )
        self.state.ledger.total_inclusion = self._money_to_float(
            result.total_inclusion,
        )
        self.state.ledger.tax_due = self._money_to_float(result.tax_due)
        self.state.pay_incremental_tax(self.state.ledger.tax_due)

    def compute_complexity_penalty(self) -> float:
        extra_corps = max(0, len(self.state.corporations) - 3)
        return self.extra_corporation_penalty * extra_corps

    def render(self):
        return self.render_world()

    def render_world(self, filename="barnes_group_world"):
        dot = Digraph()

        for corp_id, corporation in self.state.corporations.items():
            label = (
                f"{corp_id}\\n"
                f"{corporation.tax_residence.value}\\n"
                f"cash: {self.state.cash_amount(corp_id):.0f}"
            )
            if corp_id == self.state.taxpayer_id:
                label += "\\ntaxpayer"
            if self.state.is_cfc(corp_id):
                label += "\\nCFC"

            fillcolor = "lightblue" if corporation.tax_residence == TaxResidence.US else "lightyellow"
            dot.node(corp_id, label, shape="box", style="filled", fillcolor=fillcolor)

            if corporation.parent_id is not None:
                dot.edge(corporation.parent_id, corp_id, label="sub corp", color="blue")

        for stock_id, stock in self.state.stock.items():
            label = (
                f"{stock_id}\\n"
                f"{stock.percent:.0f}% {stock.issuer_id}\\n"
                f"basis: {stock.basis:.0f}"
            )
            dot.node(stock_id, label, shape="note")
            dot.edge(stock.holder_id, stock_id, label="holds")
            dot.edge(stock.issuer_id, stock_id, label="issues", style="dashed")
            if stock.contributed_by_id is not None:
                dot.edge(stock.contributed_by_id, stock_id, label="contributed", style="dotted")

        for cash_id, cash in self.state.cash.items():
            label = f"{cash_id}\\namount: {cash.amount:.0f}\\nbasis: {cash.basis:.0f}"
            dot.node(cash_id, label, shape="oval")
            dot.edge(cash.owner_id, cash_id, label="owns")
            if cash.contributed_by_id is not None:
                dot.edge(cash.contributed_by_id, cash_id, label="contributed", style="dotted")

        with open(f"{filename}.png", "wb") as image_file:
            image_file.write(dot.pipe(format="png"))

    def _make_subcorporation(self, parent_idx: int) -> None:
        if len(self.state.corporations) >= self.max_corporations:
            raise ValueError("Maximum number of corporations reached")
        parent_id = self._get_corporation_id(parent_idx)
        sub_id = f"DS_{len(self.state.corporations) - 1}"
        self.state.add_subcorporation(parent_id, sub_id, TaxResidence.US)
        self._refresh_indices()

    def _transfer_cash(self, from_idx: int, to_idx: int, amount_idx: int) -> None:
        from_id = self._get_corporation_id(from_idx)
        to_id = self._get_corporation_id(to_idx)
        amount = self._get_cash_amount(amount_idx)
        from_foreign = self.state.corporations[from_id].is_foreign
        to_domestic = self.state.corporations[to_id].is_domestic
        from_domestic = self.state.corporations[from_id].is_domestic

        if from_domestic and to_domestic:
            if not self.state.is_subcorporation_of(from_id, to_id):
                raise ValueError("Domestic cash transfer must move from direct child to parent")
            if (
                to_id == self.state.taxpayer_id
                and self._transfer_uses_foreign_contributed_cash(from_id, amount)
                and not self.state.has_qualifying_zero_basis_cfc_stock(from_id)
            ):
                raise ValueError("Foreign-contributed cash requires qualifying zero-basis CFC stock")

        self.state.transfer_cash(from_id, to_id, amount)

        if from_foreign and to_domestic and to_id == self.state.taxpayer_id:
            self.state.record_direct_repatriation(amount)

    def _transfer_stock(self, stock_idx: int, to_idx: int, percent_idx: int) -> None:
        stock_id = self._get_stock_id(stock_idx)
        to_id = self._get_corporation_id(to_idx)
        source = self.state.stock[stock_id]
        percent = min(source.percent, self._get_stock_percent(percent_idx))
        self.state.transfer_stock(stock_id, to_id, percent)

    def _issue_stock(self, issuer_idx: int, recipient_idx: int, percent_idx: int) -> None:
        if len(self.state.stock) >= self.max_stocks:
            raise ValueError("Maximum number of stock lots reached")
        issuer_id = self._get_corporation_id(issuer_idx)
        recipient_id = self._get_corporation_id(recipient_idx)
        percent = self._get_stock_percent(percent_idx)
        self.state.issue_stock(issuer_id, recipient_id, percent)

    def _transfer_uses_foreign_contributed_cash(self, from_id: str, amount: float) -> bool:
        cash = self.state._find_cash_lot(from_id, amount)
        contributor_id = cash.contributed_by_id
        if contributor_id is None:
            return False
        return self.state.corporations[contributor_id].is_foreign

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

    def _get_stock_percent(self, idx: int) -> float:
        if idx >= len(self.stock_percents):
            raise ValueError(f"Unknown stock percent index: {idx}")
        return self.stock_percents[idx]

    def _money(self, amount: float):
        return catala_runtime.Money(catala_runtime.Integer(int(amount)))

    def _decimal_rate(self, amount: float):
        return catala_runtime.Decimal(str(amount))

    def _money_to_float(self, money) -> float:
        return float(money.value.value)

    def _refresh_indices(self) -> None:
        self.idx_to_corporation = {
            i: corporation_id
            for i, corporation_id in enumerate(self.state.corporations.keys())
        }
        self.idx_to_stock = {
            i: stock_id
            for i, stock_id in enumerate(self.state.stock.keys())
        }
