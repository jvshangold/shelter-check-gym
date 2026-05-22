from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaxResidence(Enum):
    US = "US"
    FOREIGN = "Foreign"


class AssetKind(Enum):
    CASH = "Cash"
    PROPERTY = "Property"

class OwnerType(Enum):
    TRUST = "Trust"
    INDIVIDUAL = "Individual"

@dataclass
class Individual:
    id: str
    tax_residence: TaxResidence
    income: float | None = None


@dataclass
class Asset:
    id: str
    kind: AssetKind
    basis: float
    fair_market_value: float
    owner_type: OwnerType
    owner_id: str
    sale_price: float | None = None

    @property
    def is_sold(self) -> bool:
        return self.sale_price is not None
    
    @property
    def realized_loss(self) -> float:
        if self.sale_price is None:
            return 0.0
        return max(0.0, self.basis - self.sale_price)
@dataclass
class Trust:
    id: str
    parent_trust_id: str | None = None
    beneficiary_id: str | None = None
    section_678_power_holder_id: str | None = None


@dataclass
class WorldState:
    individuals: dict[str, Individual] = field(default_factory=dict)
    trusts: dict[str, Trust] = field(default_factory=dict)
    assets: dict[str, Asset] = field(default_factory=dict)

    root_trust_id: str | None = None
    taxpayer_id: str | None = None

    def add_individual(self, individual_id: str, tax_residence: TaxResidence) -> None:
        self.individuals[individual_id] = Individual(
            id=individual_id,
            tax_residence=tax_residence,
        )

    def add_trust(
        self,
        trust_id: str,
        parent_trust_id: str | None = None,
        beneficiary_id: str | None = None,
        section_678_power_holder_id: str | None = None,
    ) -> None:
        if parent_trust_id is not None and parent_trust_id not in self.trusts:
            raise ValueError(f"Unknown parent trust: {parent_trust_id}")

        if beneficiary_id is not None and beneficiary_id not in self.individuals:
            raise ValueError(f"Unknown beneficiary: {beneficiary_id}")

        if (
            section_678_power_holder_id is not None
            and section_678_power_holder_id not in self.individuals
        ):
            raise ValueError(
                f"Unknown §678 power holder: {section_678_power_holder_id}"
            )

        self.trusts[trust_id] = Trust(
            id=trust_id,
            parent_trust_id=parent_trust_id,
            beneficiary_id=beneficiary_id,
            section_678_power_holder_id=section_678_power_holder_id,
        )

        if self.root_trust_id is None:
            self.root_trust_id = trust_id

    def add_asset(
        self,
        asset_id: str,
        kind: AssetKind,
        basis: float,
        fair_market_value: float,
        owner_trust_id: str,
    ) -> None:
        if owner_trust_id not in self.trusts:
            raise ValueError(f"Unknown owner trust: {owner_trust_id}")

        self.assets[asset_id] = Asset(
            id=asset_id,
            kind=kind,
            basis=basis,
            fair_market_value=fair_market_value,
            owner_trust_id=owner_trust_id,
        )

    def transfer_asset(self, asset_id: str, new_owner_trust_id: str) -> None:
        if asset_id not in self.assets:
            raise ValueError(f"Unknown asset: {asset_id}")

        if new_owner_trust_id not in self.trusts:
            raise ValueError(f"Unknown trust: {new_owner_trust_id}")

        self.assets[asset_id].owner_trust_id = new_owner_trust_id

    def set_section_678_power_holder(
        self,
        trust_id: str,
        individual_id: str | None,
    ) -> None:
        if trust_id not in self.trusts:
            raise ValueError(f"Unknown trust: {trust_id}")

        if individual_id is not None and individual_id not in self.individuals:
            raise ValueError(f"Unknown individual: {individual_id}")

        self.trusts[trust_id].section_678_power_holder_id = individual_id

    def assets_owned_by_trust(self, trust_id: str) -> list[Asset]:
        if trust_id not in self.trusts:
            raise ValueError(f"Unknown trust: {trust_id}")

        return [
            asset
            for asset in self.assets.values()
            if asset.owner_trust_id == trust_id
        ]

    def subtrusts_of(self, trust_id: str) -> list[Trust]:
        if trust_id not in self.trusts:
            raise ValueError(f"Unknown trust: {trust_id}")

        return [
            trust
            for trust in self.trusts.values()
            if trust.parent_trust_id == trust_id
        ]

    @classmethod
    def initial_state(cls) -> WorldState:
        state = cls()

        state.add_individual("T", TaxResidence.US)
        state.taxpayer_id = "T"

        state.add_trust("root_trust")

        return state
    