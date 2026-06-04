import sys
from typing import Dict

import gymnasium as gym
from gymnasium import Space, spaces
from graphviz import Digraph

from .render import build_graph
from .state import StraddleLegKind, WorldState

sys.path.append("straddle_abuse/formalizations/_target/straddle_abuse_tax_rules")
sys.path.append("straddle_abuse/formalizations/_build/libcatala/python")

from python import StraddleAbuseTaxModel as TaxModel
from python import catala_runtime


ENTER_STRADDLE = 0
REALIZE_GAIN = 1
REALIZE_LOSS = 2
INVEST = 3


class TaxEnv(gym.Env):
    def __init__(
        self,
        MAX_STRADDLES=5,
        MAX_STEPS=12,
        STRADDLE_AMOUNTS=None,
        FRACTIONS=None,
        EXTRA_STRADDLE_PENALTY=0.01,
        SUCCESS_TAX_ADVANTAGE=30.0,
        INITIAL_FACILITATOR_INVESTMENT=300.0,
        TAXPAYER_ORDINARY_INCOME=100.0,
    ):
        super().__init__()

        self.observation_space: Space = spaces.Dict({})

        self.straddle_amounts = STRADDLE_AMOUNTS or [300.0, 500.0, 700.0, 900.0]
        self.fractions = FRACTIONS or [0.25, 0.5, 0.75, 1.0]
        self.individual_ids = ["T", "A", "B"]

        self.action_space: Space = spaces.MultiDiscrete([
            4,
            MAX_STRADDLES,
            len(self.fractions),
            len(self.individual_ids),
        ])

        self.max_straddles = MAX_STRADDLES
        self.max_steps = MAX_STEPS
        self.extra_straddle_penalty = EXTRA_STRADDLE_PENALTY
        self.success_tax_advantage = SUCCESS_TAX_ADVANTAGE
        self.initial_facilitator_investment = INITIAL_FACILITATOR_INVESTMENT
        self.taxpayer_ordinary_income = TAXPAYER_ORDINARY_INCOME

        self.state = WorldState.initial_state(
            facilitator_investment=self.initial_facilitator_investment,
        )
        self.idx_to_straddle: Dict[int, int] = {}
        self.idx_to_individual: Dict[int, str] = {
            i: individual_id for i, individual_id in enumerate(self.individual_ids)
        }

        self.prev_tax_advantage = 0.0
        self.steps = 0

        self._refresh_indices()

    def step(self, action):
        action_type, straddle_idx, fraction_idx, individual_idx = action

        terminated = False
        truncated = False
        invalid_action = False

        try:
            if action_type == ENTER_STRADDLE:
                self._enter_straddle(amount_idx=fraction_idx)

            elif action_type == REALIZE_GAIN:
                self._realize_gain(straddle_idx=straddle_idx, fraction_idx=fraction_idx)

            elif action_type == REALIZE_LOSS:
                self._realize_loss(straddle_idx=straddle_idx, fraction_idx=fraction_idx)

            elif action_type == INVEST:
                self._invest(individual_idx=individual_idx, fraction_idx=fraction_idx)

            else:
                raise ValueError("Unknown action type")

        except Exception as e:
            print("Action:", action)
            print("Exception:", e)
            invalid_action = True

        current_tax_reduction = self.compute_tax_reduction()
        current_tax_advantage = max(0.0, current_tax_reduction)

        reward = (
            current_tax_advantage
            - self.prev_tax_advantage
            - self.compute_complexity_penalty()
        )

        self.prev_tax_advantage = current_tax_advantage

        if invalid_action:
            reward -= 1.0

        if current_tax_advantage >= self.success_tax_advantage and not invalid_action:
            terminated = True

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        self._refresh_indices()

        obs = self.get_observation()
        info = {
            "invalid_action": invalid_action,
            "tax_reduction": current_tax_reduction,
            "tax_advantage": current_tax_advantage,
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = WorldState.initial_state(
            facilitator_investment=self.initial_facilitator_investment,
        )
        self.steps = 0
        self.prev_tax_advantage = 0.0
        self._refresh_indices()

        return self.get_observation(), {}

    def get_observation(self):
        return build_graph(state=self.state)

    def render(self):
        return self.render_world()

    def compute_tax_reduction(self) -> float:
        catala_input = self.state.taxpayer_catala_input()

        ctf_tax_input = TaxModel.CommonTrustFundTaxInput(
            straddle_history=TaxModel.RecognizedStraddleHistory(
                gain_leg_realized=catala_input["gain_leg_realized"],
                loss_leg_realized=catala_input["loss_leg_realized"],
            ),
            taxpayer_allocation=TaxModel.TaxpayerCtfAllocation(
                ordinary_income=self._money(self.taxpayer_ordinary_income),
                allocated_gain=self._money(catala_input["allocated_gain"]),
                allocated_loss=self._money(catala_input["allocated_loss"]),
                unrecognized_gain=self._money(catala_input["unrecognized_gain"]),
                tax_rate=self._decimal_rate(catala_input["tax_rate"]),
            ),
        )

        result = TaxModel.taxpayer_straddle_tax_computation(
            TaxModel.TaxpayerStraddleTaxComputationIn(
                ctf_tax_input_in=ctf_tax_input,
            )
        )

        return self._money_to_float(result.tax_reduction)

    def compute_complexity_penalty(self) -> float:
        extra_straddles = max(0, self._straddle_count() - 1)
        return self.extra_straddle_penalty * extra_straddles

    def get_action_mask(self) -> list[int]:
        mask = [1, 1, 1, 1]

        if not self.state.can_enter_straddle() or self._straddle_count() >= self.max_straddles:
            mask[ENTER_STRADDLE] = 0

        if not self._can_realize_any(StraddleLegKind.GAIN):
            mask[REALIZE_GAIN] = 0

        if not self._can_realize_any(StraddleLegKind.LOSS):
            mask[REALIZE_LOSS] = 0

        if not any(individual.cash > 0.0 for individual in self.state.individuals.values()):
            mask[INVEST] = 0

        return mask

    def get_straddle_mask(self, kind: StraddleLegKind | None = None) -> list[int]:
        valid_ids = set(self.state.unrealized_straddle_ids(kind))
        return [
            1 if self.idx_to_straddle.get(idx) in valid_ids else 0
            for idx in range(self.max_straddles)
        ]

    def get_individual_mask(self) -> list[int]:
        return [
            1 if self.state.individuals[individual_id].cash > 0.0 else 0
            for _, individual_id in sorted(self.idx_to_individual.items())
        ]

    def get_fraction_mask(self) -> list[int]:
        return [1 for _ in self.fractions]

    def _enter_straddle(self, amount_idx: int) -> None:
        if self._straddle_count() >= self.max_straddles:
            raise ValueError("Maximum number of straddles reached")

        amount = self._get_fraction_or_amount(amount_idx, self.straddle_amounts)
        self.state.enter_straddle(amount)
        self._refresh_indices()

    def _realize_gain(self, straddle_idx: int, fraction_idx: int) -> None:
        straddle_id = self._get_straddle_id(straddle_idx)
        fraction = self._get_fraction_or_amount(fraction_idx, self.fractions)
        self.state.realize_gain(straddle_id=straddle_id, fraction=fraction)

    def _realize_loss(self, straddle_idx: int, fraction_idx: int) -> None:
        straddle_id = self._get_straddle_id(straddle_idx)
        fraction = self._get_fraction_or_amount(fraction_idx, self.fractions)
        self.state.realize_loss(straddle_id=straddle_id, fraction=fraction)

    def _invest(self, individual_idx: int, fraction_idx: int) -> None:
        individual_id = self._get_individual_id(individual_idx)
        fraction = self._get_fraction_or_amount(fraction_idx, self.fractions)
        individual = self.state.individuals[individual_id]
        amount = individual.cash * fraction
        self.state.invest(individual_id, amount)

    def _can_realize_any(self, kind: StraddleLegKind) -> bool:
        return any(
            self.state.can_realize_gain(straddle_id)
            if kind == StraddleLegKind.GAIN
            else self.state.can_realize_loss(straddle_id)
            for straddle_id in self.state.unrealized_straddle_ids(kind)
        )

    def _get_straddle_id(self, idx: int) -> int:
        if idx not in self.idx_to_straddle:
            raise ValueError(f"Invalid straddle index: {idx}")
        return self.idx_to_straddle[idx]

    def _get_individual_id(self, idx: int) -> str:
        if idx not in self.idx_to_individual:
            raise ValueError(f"Invalid individual index: {idx}")
        return self.idx_to_individual[idx]

    def _get_fraction_or_amount(self, idx: int, values: list[float]) -> float:
        if idx < 0 or idx >= len(values):
            raise ValueError(f"Invalid bucket index: {idx}")
        return values[idx]

    def _straddle_count(self) -> int:
        return len({leg.straddle_id for leg in self.state.straddle_legs.values()})

    def _refresh_indices(self) -> None:
        straddle_ids = sorted({
            leg.straddle_id for leg in self.state.straddle_legs.values()
        })
        self.idx_to_straddle = {
            i: straddle_id for i, straddle_id in enumerate(straddle_ids)
        }

    def _money(self, amount: float):
        return catala_runtime.Money(catala_runtime.Integer(round(amount * 100)))

    def _money_to_float(self, money) -> float:
        return float(money.value.value) / 100.0

    def _decimal_rate(self, rate: float):
        basis_points = round(rate * 10_000)
        return catala_runtime.Decimal(f"{basis_points}/10000")

    def render_world(self, filename="straddle_world"):
        dot = Digraph()

        ctf_label = (
            f"CTF\\nparticipants: {self.state.ctf.participant_count}\\n"
            f"contributions: {self.state.ctf.total_contributions:.2f}"
        )
        dot.node("ctf", ctf_label, shape="box", style="filled", fillcolor="lightblue")

        for individual_id, individual in self.state.individuals.items():
            contribution = self.state.ctf.contribution_of(individual_id)
            label = (
                f"{individual_id}\\ncash: {individual.cash:.2f}\\n"
                f"ctf: {contribution:.2f}\\n"
                f"gain: {self.state.allocated_gain_for(individual_id):.2f}\\n"
                f"loss: {self.state.allocated_loss_for(individual_id):.2f}"
            )
            dot.node(individual_id, label, shape="ellipse")
            if contribution > 0.0:
                dot.edge(individual_id, "ctf", label="invested")
            else:
                dot.edge(individual_id, "ctf", label="not invested", style="dashed")

        for leg_id, leg in self.state.straddle_legs.items():
            label = (
                f"{leg_id}\\n{leg.kind.value}\\n"
                f"built-in: {leg.built_in_amount:.2f}\\n"
                f"realized: {leg.realized_amount:.2f}\\n"
                f"remaining: {leg.remaining_amount:.2f}"
            )
            fillcolor = "lightgreen" if leg.kind == StraddleLegKind.GAIN else "lightpink"
            dot.node(leg_id, label, shape="note", style="filled", fillcolor=fillcolor)
            dot.edge("ctf", leg_id, label="owns")

        for leg in self.state.straddle_legs.values():
            if leg.kind != StraddleLegKind.GAIN:
                continue
            loss_leg_id = f"straddle_{leg.straddle_id}_loss"
            if loss_leg_id in self.state.straddle_legs:
                dot.edge(leg.id, loss_leg_id, label="offsets", style="dotted")

        with open(f"{filename}.png", "wb") as image_file:
            image_file.write(dot.pipe(format="png"))
