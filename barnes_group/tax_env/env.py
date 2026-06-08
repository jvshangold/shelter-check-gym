from typing import Dict

import gymnasium as gym
from graphviz import Digraph
from gymnasium import Space, spaces

from .render import build_graph
from .state import TaxResidence, WorldState


MAKE_SUBCORPORATION = 0
TRANSFER_CASH = 1
TRANSFER_STOCK = 2
ISSUE_STOCK = 3
DISTRIBUTE_CASH = 4


class TaxEnv(gym.Env):
    def __init__(
        self,
        MAX_CORPORATIONS=5,
        MAX_STOCKS=8,
        MAX_STEPS=10,
        CASH_AMOUNTS=None,
        STOCK_PERCENTS=None,
        EXTRA_CORPORATION_PENALTY=0.01,
        SUCCESS_ADVANTAGE=1.0,
        TAX_RATE=0.35,
        INITIAL_T_CASH=100.0,
        INITIAL_FSUB_CASH=100.0,
    ):
        super().__init__()

        self.observation_space: Space = spaces.Dict({})
        self.cash_amounts = CASH_AMOUNTS or [25.0, 50.0, 75.0, 100.0]
        self.stock_percents = STOCK_PERCENTS or [25.0, 50.0, 70.0, 80.0, 100.0]

        self.action_space: Space = spaces.MultiDiscrete([
            5,
            MAX_CORPORATIONS,
            MAX_CORPORATIONS,
            MAX_STOCKS,
            max(len(self.cash_amounts), len(self.stock_percents)),
        ])

        self.max_corporations = MAX_CORPORATIONS
        self.max_stocks = MAX_STOCKS
        self.max_steps = MAX_STEPS
        self.extra_corporation_penalty = EXTRA_CORPORATION_PENALTY
        self.success_advantage = SUCCESS_ADVANTAGE
        self.tax_rate = TAX_RATE
        self.initial_t_cash = INITIAL_T_CASH
        self.initial_fsub_cash = INITIAL_FSUB_CASH

        self.state = WorldState.initial_state(
            t_cash=self.initial_t_cash,
            fsub_cash=self.initial_fsub_cash,
            applicable_earnings=self.initial_fsub_cash,
            tax_rate=self.tax_rate,
        )
        self.idx_to_corporation: Dict[int, str] = {}
        self.idx_to_stock: Dict[int, str] = {}
        self.prev_advantage = 0.0
        self.steps = 0
        self._refresh_indices()

    def step(self, action):
        action_type, corp_a_idx, corp_b_idx, stock_idx, amount_idx = action

        terminated = False
        truncated = False
        invalid_action = False

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
            elif action_type == DISTRIBUTE_CASH:
                self._distribute_cash(
                    from_idx=corp_a_idx,
                    to_idx=corp_b_idx,
                    amount_idx=amount_idx,
                )
            else:
                raise ValueError("Unknown action type")

        except Exception as e:
            print("Action:", action)
            print("Exception:", e)
            invalid_action = True

        self.state.recompute_section_956()
        current_advantage = max(0.0, self.compute_tax_advantage())
        reward = (
            current_advantage
            - self.prev_advantage
            - self.compute_complexity_penalty()
        )
        self.prev_advantage = current_advantage

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
        self.prev_advantage = 0.0
        self.steps = 0
        self._refresh_indices()
        return self.get_observation(), {}

    def get_observation(self):
        return build_graph(self.state)

    def compute_tax_advantage(self) -> float:
        t_gain = self.state.cash_amount("T") - self.initial_t_cash
        return t_gain - self.state.direct_transfer_baseline_net()

    def has_desired_loophole_structure(self) -> bool:
        return (
            self.compute_tax_advantage() > 0.0
            and self.state.ledger.section_956_inclusion == 0.0
            and self.state.cash_amount("T") > self.initial_t_cash
            and any(
                corporation.parent_id == self.state.taxpayer_id
                and corporation.is_domestic
                for corporation in self.state.corporations.values()
            )
        )

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
        self.state.create_subcorporation(parent_id, sub_id, TaxResidence.US)
        self._refresh_indices()

    def _transfer_cash(self, from_idx: int, to_idx: int, amount_idx: int) -> None:
        from_id = self._get_corporation_id(from_idx)
        to_id = self._get_corporation_id(to_idx)
        amount = self._get_cash_amount(amount_idx)
        from_foreign = self.state.corporations[from_id].is_foreign
        to_domestic = self.state.corporations[to_id].is_domestic

        self.state.transfer_cash(from_id, to_id, amount)

        if from_foreign and to_domestic and to_id == self.state.taxpayer_id:
            self.state.apply_direct_repatriation_tax(amount)

    def _distribute_cash(self, from_idx: int, to_idx: int, amount_idx: int) -> None:
        from_id = self._get_corporation_id(from_idx)
        to_id = self._get_corporation_id(to_idx)
        amount = self._get_cash_amount(amount_idx)
        self.state.distribute_cash(from_id, to_id, amount)

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

    def _refresh_indices(self) -> None:
        self.idx_to_corporation = {
            i: corporation_id
            for i, corporation_id in enumerate(self.state.corporations.keys())
        }
        self.idx_to_stock = {
            i: stock_id
            for i, stock_id in enumerate(self.state.stock.keys())
        }
