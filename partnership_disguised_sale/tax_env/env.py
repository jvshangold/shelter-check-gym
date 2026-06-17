import sys
from dataclasses import dataclass
from typing import Dict

import gymnasium as gym
from gymnasium import Space, spaces
from graphviz import Digraph

from .render import build_graph
from .state import CashSource, OwnerType, WorldState

sys.path.append(
    "partnership_disguised_sale/formalizations/_target/"
    "partnership_disguised_sale_tax_rules"
)
sys.path.append("partnership_disguised_sale/formalizations/_build/libcatala/python")

try:
    from python import PartnershipDisguisedSaleTaxModel as TaxModel
    from python import catala_runtime
except ImportError:
    TaxModel = None
    catala_runtime = None


TAKE_OUT_LOAN = 0
CONTRIBUTE_ASSET = 1
DISTRIBUTE_CASH = 2
CONTRIBUTE_CASH = 3


@dataclass
class TaxComputation:
    recognized_gain: float
    baseline_gain: float
    deferred_gain: float
    tax_savings: float


class TaxEnv(gym.Env):
    def __init__(
        self,
        MAX_INDIVIDUALS=4,
        MAX_ASSETS=4,
        MAX_LOANS=4,
        MAX_STEPS=8,
        AMOUNT_BUCKETS=None,
        EXTRA_LOAN_PENALTY=0.01,
        EXTRA_CASH_LOT_PENALTY=0.005,
    ):
        super().__init__()

        self.observation_space: Space = spaces.Dict({})
        self.amount_buckets = AMOUNT_BUCKETS or [10.0, 50.0, 90.0, 100.0]
        self.action_space: Space = spaces.MultiDiscrete([
            4,
            MAX_INDIVIDUALS,
            MAX_ASSETS,
            MAX_LOANS,
            len(self.amount_buckets),
        ])

        self.max_individuals = MAX_INDIVIDUALS
        self.max_assets = MAX_ASSETS
        self.max_loans = MAX_LOANS
        self.max_steps = MAX_STEPS
        self.extra_loan_penalty = EXTRA_LOAN_PENALTY
        self.extra_cash_lot_penalty = EXTRA_CASH_LOT_PENALTY

        self.state = WorldState.initial_state()
        self.idx_to_individual: Dict[int, str] = {}
        self.idx_to_asset: Dict[int, str] = {}
        self.idx_to_loan: Dict[int, str] = {}

        self.prev_tax_advantage = 0.0
        self.steps = 0
        self._refresh_indices()

    def step(self, action):
        action_type, individual_idx, asset_idx, loan_idx, amount_idx = action

        terminated = False
        truncated = False
        invalid_action = False

        try:
            if action_type == TAKE_OUT_LOAN:
                self._take_out_loan(
                    guarantor_idx=individual_idx,
                    amount_idx=amount_idx,
                )
            elif action_type == CONTRIBUTE_ASSET:
                self._contribute_asset(
                    partner_idx=individual_idx,
                    asset_idx=asset_idx,
                )
            elif action_type == DISTRIBUTE_CASH:
                self._distribute_cash(
                    recipient_idx=individual_idx,
                    amount_idx=amount_idx,
                )
            elif action_type == CONTRIBUTE_CASH:
                self._contribute_cash(
                    partner_idx=individual_idx,
                    amount_idx=amount_idx,
                )
            else:
                raise ValueError("Unknown action type")
        except Exception as e:
            print("Action:", action)
            print("Exception:", e)
            invalid_action = True

        current_tax_advantage = max(0.0, self.compute_tax_advantage())
        reward = (
            current_tax_advantage
            - self.prev_tax_advantage
            - self.compute_complexity_penalty()
        )
        self.prev_tax_advantage = current_tax_advantage

        if invalid_action:
            reward -= 1.0

        if self.has_desired_loophole_structure() and not invalid_action:
            terminated = True

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        self._refresh_indices()
        obs = self.get_observation()
        info = {
            "invalid_action": invalid_action,
            "tax_advantage": current_tax_advantage,
            "tax_computation": self.compute_tax(),
        }
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = WorldState.initial_state()
        self.steps = 0
        self.prev_tax_advantage = 0.0
        self._refresh_indices()
        return self.get_observation(), {}

    def get_observation(self):
        return build_graph(self.state)

    def render(self):
        return self.render_world()

    def compute_tax_advantage(self) -> float:
        return self.compute_tax().tax_savings

    def compute_tax(self) -> TaxComputation:
        tax_input = self.state.taxpayer_tax_input()

        if TaxModel is not None:
            return self._compute_catala_tax(tax_input)

        return self._compute_python_tax(tax_input)

    def compute_complexity_penalty(self) -> float:
        extra_loans = max(0, len(self.state.loans) - 1)
        extra_cash_lots = max(0, len(self.state.cash_lots) - 2)
        return (
            self.extra_loan_penalty * extra_loans
            + self.extra_cash_lot_penalty * extra_cash_lots
        )

    def has_desired_loophole_structure(self) -> bool:
        tax = self.compute_tax()
        return (
            self.state.cash_distributed_to(self.state.taxpayer_id) >= 100.0
            and self.state.contributed_asset_by(self.state.taxpayer_id) is not None
            and tax.recognized_gain <= 10.0
            and tax.deferred_gain >= 90.0
        )

    def render_world(self, filename="partnership_disguised_sale_world"):
        dot = Digraph()

        for individual_id, individual in self.state.individuals.items():
            label = (
                f"{individual_id}\\nindividual\\n"
                f"cash: {self.state.individual_cash(individual_id)}\\n"
                f"rate: {individual.tax_rate}"
            )
            fillcolor = "lightblue" if individual_id == self.state.taxpayer_id else "white"
            dot.node(
                individual_id,
                label,
                shape="ellipse",
                style="filled",
                fillcolor=fillcolor,
            )

        for partnership_id, partnership in self.state.partnerships.items():
            label = (
                f"{partnership_id}\\npartnership\\n"
                f"cash: {self.state.partnership_cash(partnership_id)}"
            )
            dot.node(partnership_id, label, shape="box", style="filled", fillcolor="gray95")
            for partner_id in partnership.partner_ids:
                dot.edge(partner_id, partnership_id, label="partner")

        for asset_id, asset in self.state.assets.items():
            label = (
                f"{asset_id}\\nasset\\n"
                f"basis: {asset.basis}\\n"
                f"FMV: {asset.fair_market_value}"
            )
            if asset.contributed_by_id is not None:
                label += f"\\ncontrib: {asset.contributed_by_id}"
            dot.node(asset_id, label, shape="note", style="filled", fillcolor="lightyellow")
            dot.edge(asset.owner_id, asset_id, label="owns")

        for cash_id, cash in self.state.cash_lots.items():
            label = f"{cash_id}\\ncash\\n{cash.amount}\\n{cash.source.value}"
            dot.node(cash_id, label, shape="note")
            dot.edge(cash.owner_id, cash_id, label="holds")

        for loan_id, loan in self.state.loans.items():
            label = f"{loan_id}\\nloan\\n{loan.principal}"
            dot.node(loan_id, label, shape="diamond")
            dot.edge(loan.borrower_partnership_id, loan_id, label="borrower")
            dot.edge(loan.guarantor_id, loan_id, label="guarantees")

        with open(f"{filename}.png", "wb") as image_file:
            image_file.write(dot.pipe(format="png"))

    def _take_out_loan(self, guarantor_idx: int, amount_idx: int) -> None:
        if len(self.state.loans) >= self.max_loans:
            raise ValueError("Maximum number of loans reached")
        guarantor_id = self._get_individual_id(guarantor_idx)
        amount = self._get_amount(amount_idx)
        self.state.take_out_loan(
            partnership_id=self.state.partnership_id,
            guarantor_id=guarantor_id,
            amount=amount,
        )

    def _contribute_asset(self, partner_idx: int, asset_idx: int) -> None:
        partner_id = self._get_individual_id(partner_idx)
        asset_id = self._get_asset_id(asset_idx)
        self.state.contribute_asset(
            asset_id=asset_id,
            partner_id=partner_id,
            partnership_id=self.state.partnership_id,
        )

    def _distribute_cash(self, recipient_idx: int, amount_idx: int) -> None:
        recipient_id = self._get_individual_id(recipient_idx)
        amount = self._get_amount(amount_idx)
        self.state.distribute_cash(
            partnership_id=self.state.partnership_id,
            recipient_id=recipient_id,
            amount=amount,
        )

    def _contribute_cash(self, partner_idx: int, amount_idx: int) -> None:
        partner_id = self._get_individual_id(partner_idx)
        amount = self._get_amount(amount_idx)
        self.state.contribute_cash(
            partner_id=partner_id,
            partnership_id=self.state.partnership_id,
            amount=amount,
        )

    def _compute_catala_tax(self, tax_input: dict[str, float]) -> TaxComputation:
        transaction = TaxModel.PartnershipDisguisedSaleInput(
            asset_basis=self._money(tax_input["asset_basis"]),
            asset_fair_market_value=self._money(
                tax_input["asset_fair_market_value"]
            ),
            cash_distributed_to_transferor=self._money(
                tax_input["cash_distributed_to_transferor"]
            ),
            debt_financed_distribution=self._money(
                tax_input["debt_financed_distribution"]
            ),
            transferor_allocable_liability=self._money(
                tax_input["transferor_allocable_liability"]
            ),
            tax_rate=self._decimal_rate(tax_input["tax_rate"]),
        )

        result = TaxModel.partnership_disguised_sale_computation(
            TaxModel.PartnershipDisguisedSaleComputationIn(
                transaction_in=transaction,
            )
        )

        return TaxComputation(
            recognized_gain=self._money_to_float(result.recognized_gain_output),
            baseline_gain=self._money_to_float(result.baseline_gain_output),
            deferred_gain=self._money_to_float(result.deferred_gain_output),
            tax_savings=self._money_to_float(result.tax_savings),
        )

    def _compute_python_tax(self, tax_input: dict[str, float]) -> TaxComputation:
        baseline_gain = max(
            0.0,
            tax_input["asset_fair_market_value"] - tax_input["asset_basis"],
        )
        liability_excluded_distribution = min(
            tax_input["debt_financed_distribution"],
            tax_input["transferor_allocable_liability"],
        )
        amount_taken_into_account = max(
            0.0,
            tax_input["cash_distributed_to_transferor"]
            - liability_excluded_distribution,
        )
        recognized_gain = max(
            0.0,
            amount_taken_into_account - tax_input["asset_basis"],
        )
        deferred_gain = max(0.0, baseline_gain - recognized_gain)
        tax_savings = deferred_gain * tax_input["tax_rate"]
        return TaxComputation(
            recognized_gain=recognized_gain,
            baseline_gain=baseline_gain,
            deferred_gain=deferred_gain,
            tax_savings=tax_savings,
        )

    def _money(self, amount: float):
        return catala_runtime.Money(catala_runtime.Integer(int(amount)))

    def _money_to_float(self, money) -> float:
        return float(money.value.value)

    def _decimal_rate(self, rate: float):
        return catala_runtime.Decimal(str(rate))

    def _get_individual_id(self, idx: int) -> str:
        if idx not in self.idx_to_individual:
            raise ValueError(f"Invalid individual index: {idx}")
        return self.idx_to_individual[idx]

    def _get_asset_id(self, idx: int) -> str:
        if idx not in self.idx_to_asset:
            raise ValueError(f"Invalid asset index: {idx}")
        return self.idx_to_asset[idx]

    def _get_amount(self, idx: int) -> float:
        if idx < 0 or idx >= len(self.amount_buckets):
            raise ValueError(f"Invalid amount bucket: {idx}")
        return self.amount_buckets[idx]

    def _refresh_indices(self) -> None:
        self.idx_to_individual = {
            i: individual_id
            for i, individual_id in enumerate(self.state.individuals.keys())
        }
        self.idx_to_asset = {
            i: asset_id
            for i, asset_id in enumerate(self.state.assets.keys())
        }
        self.idx_to_loan = {
            i: loan_id
            for i, loan_id in enumerate(self.state.loans.keys())
        }
