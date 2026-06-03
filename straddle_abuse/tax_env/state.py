from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StraddleLegKind(Enum):
    GAIN = "Gain"
    LOSS = "Loss"


@dataclass
class Individual:
    id: str
    cash: float
    tax_rate: float


@dataclass
class CommonTrustFund:
    id: str = "ctf"
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def participant_count(self) -> int:
        return sum(1 for amount in self.contributions.values() if amount > 0.0)

    @property
    def total_contributions(self) -> float:
        return sum(self.contributions.values())

    @property
    def is_common_trust_fund(self) -> bool:
        return self.participant_count >= 2

    def contribution_of(self, individual_id: str) -> float:
        return self.contributions.get(individual_id, 0.0)


@dataclass
class StraddleLeg:
    id: str
    straddle_id: int
    kind: StraddleLegKind
    built_in_amount: float
    realized_amount: float = 0.0

    @property
    def remaining_amount(self) -> float:
        return max(0.0, self.built_in_amount - self.realized_amount)

    @property
    def is_fully_realized(self) -> bool:
        return self.remaining_amount <= 0.0


@dataclass
class RealizedItem:
    straddle_id: int
    leg_id: str
    kind: StraddleLegKind
    amount: float
    allocations: dict[str, float]


@dataclass
class WorldState:
    individuals: dict[str, Individual] = field(default_factory=dict)
    ctf: CommonTrustFund = field(default_factory=CommonTrustFund)
    straddle_legs: dict[str, StraddleLeg] = field(default_factory=dict)
    realized_items: list[RealizedItem] = field(default_factory=list)

    taxpayer_id: str = "T"
    _next_straddle_id: int = 1

    def add_individual(
        self,
        individual_id: str,
        cash: float,
        tax_rate: float,
    ) -> None:
        if individual_id in self.individuals:
            raise ValueError(f"Individual already exists: {individual_id}")
        if cash < 0.0:
            raise ValueError("Cash cannot be negative.")
        if tax_rate < 0.0:
            raise ValueError("Tax rate cannot be negative.")

        self.individuals[individual_id] = Individual(
            id=individual_id,
            cash=cash,
            tax_rate=tax_rate,
        )

    def invest(self, individual_id: str, amount: float) -> None:
        if individual_id not in self.individuals:
            raise ValueError(f"Unknown individual: {individual_id}")
        if amount <= 0.0:
            raise ValueError("Investment amount must be positive.")

        individual = self.individuals[individual_id]
        if amount > individual.cash:
            raise ValueError(f"{individual_id} does not have enough cash.")

        individual.cash -= amount
        self.ctf.contributions[individual_id] = (
            self.ctf.contribution_of(individual_id) + amount
        )

    def enter_straddle(self, built_in_amount: float = 300.0) -> int:
        if not self.ctf.is_common_trust_fund:
            raise ValueError("The CTF must have at least two investors.")
        if built_in_amount <= 0.0:
            raise ValueError("Straddle amount must be positive.")

        straddle_id = self._next_straddle_id
        self._next_straddle_id += 1

        gain_leg_id = f"straddle_{straddle_id}_gain"
        loss_leg_id = f"straddle_{straddle_id}_loss"

        self.straddle_legs[gain_leg_id] = StraddleLeg(
            id=gain_leg_id,
            straddle_id=straddle_id,
            kind=StraddleLegKind.GAIN,
            built_in_amount=built_in_amount,
        )
        self.straddle_legs[loss_leg_id] = StraddleLeg(
            id=loss_leg_id,
            straddle_id=straddle_id,
            kind=StraddleLegKind.LOSS,
            built_in_amount=built_in_amount,
        )

        return straddle_id

    @property
    def has_unrealized_straddle_leg(self) -> bool:
        return any(not leg.is_fully_realized for leg in self.straddle_legs.values())

    def can_enter_straddle(self) -> bool:
        return self.ctf.is_common_trust_fund

    def can_realize_gain(self, straddle_id: int | None = None) -> bool:
        return (
            self.ctf.is_common_trust_fund
            and self._unrealized_leg_id(StraddleLegKind.GAIN, straddle_id) is not None
        )

    def can_realize_loss(self, straddle_id: int | None = None) -> bool:
        return (
            self.ctf.is_common_trust_fund
            and self._unrealized_leg_id(StraddleLegKind.LOSS, straddle_id) is not None
        )

    def has_unrealized_gain_leg(self) -> bool:
        return self._unrealized_leg_id(StraddleLegKind.GAIN) is not None

    def has_realized_gain_before_taxpayer_invested(self) -> bool:
        return any(
            item.kind == StraddleLegKind.GAIN
            and item.allocations.get(self.taxpayer_id, 0.0) == 0.0
            for item in self.realized_items
        )

    def realize_gain(
        self,
        straddle_id: int | None = None,
        fraction: float = 1.0,
    ) -> RealizedItem:
        return self._realize_leg(StraddleLegKind.GAIN, straddle_id, fraction)

    def realize_loss(
        self,
        straddle_id: int | None = None,
        fraction: float = 1.0,
    ) -> RealizedItem:
        return self._realize_leg(StraddleLegKind.LOSS, straddle_id, fraction)

    def _realize_leg(
        self,
        kind: StraddleLegKind,
        straddle_id: int | None,
        fraction: float,
    ) -> RealizedItem:
        if not self.ctf.is_common_trust_fund:
            raise ValueError("The CTF must have at least two investors.")
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Realization fraction must be in (0, 1].")

        leg_id = self._unrealized_leg_id(kind, straddle_id)
        if leg_id is None:
            candidate_ids = self.unrealized_straddle_ids(kind)
            if straddle_id is None and len(candidate_ids) > 1:
                raise ValueError(
                    f"Multiple unrealized {kind.value.lower()} legs are available; "
                    "provide a straddle_id."
                )
            raise ValueError(f"No unrealized {kind.value.lower()} leg is available.")

        leg = self.straddle_legs[leg_id]
        realized_amount = leg.remaining_amount * fraction
        allocations = self._allocate_to_current_participants(realized_amount)
        leg.realized_amount += realized_amount

        realized_item = RealizedItem(
            straddle_id=leg.straddle_id,
            leg_id=leg.id,
            kind=leg.kind,
            amount=realized_amount,
            allocations=allocations,
        )
        self.realized_items.append(realized_item)
        return realized_item

    def unrealized_straddle_ids(self, kind: StraddleLegKind | None = None) -> list[int]:
        straddle_ids = {
            leg.straddle_id
            for leg in self.straddle_legs.values()
            if not leg.is_fully_realized and (kind is None or leg.kind == kind)
        }
        return sorted(straddle_ids)

    def _unrealized_leg_id(
        self,
        kind: StraddleLegKind,
        straddle_id: int | None = None,
    ) -> str | None:
        candidate_ids = self.unrealized_straddle_ids(kind)
        if straddle_id is None:
            if len(candidate_ids) != 1:
                return None
            straddle_id = candidate_ids[0]

        if straddle_id not in candidate_ids:
            return None

        for leg_id, leg in self.straddle_legs.items():
            if (
                leg.straddle_id == straddle_id
                and leg.kind == kind
                and not leg.is_fully_realized
            ):
                return leg_id
        return None

    def _allocate_to_current_participants(self, amount: float) -> dict[str, float]:
        total = self.ctf.total_contributions
        if total <= 0.0:
            raise ValueError("Cannot allocate without CTF contributions.")

        return {
            individual_id: amount * contribution / total
            for individual_id, contribution in self.ctf.contributions.items()
            if contribution > 0.0
        }

    def allocated_gain_for(self, individual_id: str) -> float:
        return self._allocated_amount_for(individual_id, StraddleLegKind.GAIN)

    def allocated_loss_for(self, individual_id: str) -> float:
        return self._allocated_amount_for(individual_id, StraddleLegKind.LOSS)

    def _allocated_amount_for(
        self,
        individual_id: str,
        kind: StraddleLegKind,
    ) -> float:
        if individual_id not in self.individuals:
            raise ValueError(f"Unknown individual: {individual_id}")

        return sum(
            item.allocations.get(individual_id, 0.0)
            for item in self.realized_items
            if item.kind == kind
        )

    def taxpayer_catala_input(self) -> dict[str, object]:
        taxpayer = self.individuals[self.taxpayer_id]

        return {
            "gain_leg_realized": self._has_realized(StraddleLegKind.GAIN),
            "loss_leg_realized": self._has_realized(StraddleLegKind.LOSS),
            "allocated_gain": self.allocated_gain_for(self.taxpayer_id),
            "allocated_loss": self.allocated_loss_for(self.taxpayer_id),
            "tax_rate": taxpayer.tax_rate,
        }

    def _has_realized(self, kind: StraddleLegKind) -> bool:
        return any(item.kind == kind for item in self.realized_items)

    @classmethod
    def initial_state(cls) -> WorldState:
        state = cls()

        state.add_individual(
            individual_id="T",
            cash=100.0,
            tax_rate=0.37,
        )
        state.add_individual(
            individual_id="A",
            cash=500.0,
            tax_rate=0.0,
        )
        state.add_individual(
            individual_id="B",
            cash=500.0,
            tax_rate=0.0,
        )

        return state
