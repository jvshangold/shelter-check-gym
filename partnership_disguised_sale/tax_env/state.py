from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OwnerType(Enum):
    INDIVIDUAL = "Individual"
    PARTNERSHIP = "Partnership"


class CashSource(Enum):
    INITIAL = "Initial"
    CONTRIBUTION = "Contribution"
    LOAN_PROCEEDS = "Loan Proceeds"
    DISTRIBUTION = "Distribution"


@dataclass
class Individual:
    id: str
    tax_rate: float


@dataclass
class Partnership:
    id: str
    partner_ids: set[str] = field(default_factory=set)
    capital_accounts: dict[str, float] = field(default_factory=dict)


@dataclass
class Asset:
    id: str
    basis: float
    fair_market_value: float
    owner_type: OwnerType
    owner_id: str
    contributed_by_id: str | None = None
    contributed_to_partnership_id: str | None = None

    @property
    def built_in_gain(self) -> float:
        return max(0.0, self.fair_market_value - self.basis)

    @property
    def is_contributed_to_partnership(self) -> bool:
        return self.contributed_to_partnership_id is not None


@dataclass
class CashLot:
    id: str
    amount: float
    owner_type: OwnerType
    owner_id: str
    source: CashSource
    contributed_by_id: str | None = None
    distributed_to_id: str | None = None
    loan_id: str | None = None


@dataclass
class Loan:
    id: str
    principal: float
    borrower_partnership_id: str
    guarantor_id: str


@dataclass
class WorldState:
    individuals: dict[str, Individual] = field(default_factory=dict)
    partnerships: dict[str, Partnership] = field(default_factory=dict)
    assets: dict[str, Asset] = field(default_factory=dict)
    cash_lots: dict[str, CashLot] = field(default_factory=dict)
    loans: dict[str, Loan] = field(default_factory=dict)

    taxpayer_id: str = "T"
    partnership_id: str = "P"
    _next_cash_id: int = 1
    _next_loan_id: int = 1

    def add_individual(self, individual_id: str, tax_rate: float) -> None:
        if individual_id in self.individuals:
            raise ValueError(f"Individual already exists: {individual_id}")
        if tax_rate < 0.0:
            raise ValueError("Tax rate cannot be negative")
        self.individuals[individual_id] = Individual(individual_id, tax_rate)

    def add_partnership(self, partnership_id: str, partner_ids: set[str]) -> None:
        if partnership_id in self.partnerships:
            raise ValueError(f"Partnership already exists: {partnership_id}")
        unknown = partner_ids - set(self.individuals)
        if unknown:
            raise ValueError(f"Unknown partners: {sorted(unknown)}")
        self.partnerships[partnership_id] = Partnership(
            id=partnership_id,
            partner_ids=set(partner_ids),
            capital_accounts={
                partner_id: 0.0
                for partner_id in partner_ids
            },
        )

    def add_asset(
        self,
        asset_id: str,
        basis: float,
        fair_market_value: float,
        owner_type: OwnerType,
        owner_id: str,
    ) -> None:
        self._validate_owner(owner_type, owner_id)
        self.assets[asset_id] = Asset(
            id=asset_id,
            basis=basis,
            fair_market_value=fair_market_value,
            owner_type=owner_type,
            owner_id=owner_id,
        )

    def add_cash_lot(
        self,
        amount: float,
        owner_type: OwnerType,
        owner_id: str,
        source: CashSource,
        contributed_by_id: str | None = None,
        distributed_to_id: str | None = None,
        loan_id: str | None = None,
    ) -> str:
        if amount <= 0.0:
            raise ValueError("Cash amount must be positive")
        self._validate_owner(owner_type, owner_id)
        cash_id = f"cash_{self._next_cash_id}"
        self._next_cash_id += 1
        self.cash_lots[cash_id] = CashLot(
            id=cash_id,
            amount=amount,
            owner_type=owner_type,
            owner_id=owner_id,
            source=source,
            contributed_by_id=contributed_by_id,
            distributed_to_id=distributed_to_id,
            loan_id=loan_id,
        )
        return cash_id

    def contribute_asset(
        self,
        asset_id: str,
        partner_id: str,
        partnership_id: str,
    ) -> None:
        if partner_id not in self.individuals:
            raise ValueError(f"Unknown partner: {partner_id}")
        partnership = self._get_partnership(partnership_id)
        if partner_id not in partnership.partner_ids:
            raise ValueError(f"{partner_id} is not a partner in {partnership_id}")
        if asset_id not in self.assets:
            raise ValueError(f"Unknown asset: {asset_id}")

        asset = self.assets[asset_id]
        if asset.owner_type != OwnerType.INDIVIDUAL or asset.owner_id != partner_id:
            raise ValueError("Partner must own asset before contribution")
        if asset.is_contributed_to_partnership:
            raise ValueError("Asset already contributed")

        asset.owner_type = OwnerType.PARTNERSHIP
        asset.owner_id = partnership_id
        asset.contributed_by_id = partner_id
        asset.contributed_to_partnership_id = partnership_id
        partnership.capital_accounts[partner_id] += asset.fair_market_value

    def contribute_cash(
        self,
        partner_id: str,
        partnership_id: str,
        amount: float,
    ) -> None:
        if partner_id not in self.individuals:
            raise ValueError(f"Unknown partner: {partner_id}")
        partnership = self._get_partnership(partnership_id)
        if partner_id not in partnership.partner_ids:
            raise ValueError(f"{partner_id} is not a partner in {partnership_id}")
        self._spend_individual_cash(partner_id, amount)
        partnership.capital_accounts[partner_id] += amount
        self.add_cash_lot(
            amount=amount,
            owner_type=OwnerType.PARTNERSHIP,
            owner_id=partnership_id,
            source=CashSource.CONTRIBUTION,
            contributed_by_id=partner_id,
        )

    def take_out_loan(
        self,
        partnership_id: str,
        guarantor_id: str,
        amount: float,
    ) -> str:
        self._get_partnership(partnership_id)
        if guarantor_id not in self.individuals:
            raise ValueError(f"Unknown guarantor: {guarantor_id}")
        if guarantor_id not in self.partnerships[partnership_id].partner_ids:
            raise ValueError("Guarantor must be a partner")
        if amount <= 0.0:
            raise ValueError("Loan amount must be positive")

        loan_id = f"loan_{self._next_loan_id}"
        self._next_loan_id += 1
        self.loans[loan_id] = Loan(
            id=loan_id,
            principal=amount,
            borrower_partnership_id=partnership_id,
            guarantor_id=guarantor_id,
        )
        self.add_cash_lot(
            amount=amount,
            owner_type=OwnerType.PARTNERSHIP,
            owner_id=partnership_id,
            source=CashSource.LOAN_PROCEEDS,
            loan_id=loan_id,
        )
        return loan_id

    def distribute_cash(
        self,
        partnership_id: str,
        recipient_id: str,
        amount: float,
    ) -> None:
        partnership = self._get_partnership(partnership_id)
        if recipient_id not in partnership.partner_ids:
            raise ValueError("Recipient must be a partner")
        if amount <= 0.0:
            raise ValueError("Distribution amount must be positive")
        if self.partnership_cash(partnership_id) < amount:
            raise ValueError("Partnership does not have enough cash")

        partnership.capital_accounts[recipient_id] -= amount
        remaining = amount
        for cash in list(self.cash_lots.values()):
            if remaining <= 0.0:
                break
            if cash.owner_type != OwnerType.PARTNERSHIP or cash.owner_id != partnership_id:
                continue

            used = min(cash.amount, remaining)
            remaining -= used
            cash.amount -= used

            self.add_cash_lot(
                amount=used,
                owner_type=OwnerType.INDIVIDUAL,
                owner_id=recipient_id,
                source=CashSource.DISTRIBUTION,
                distributed_to_id=recipient_id,
                loan_id=cash.loan_id,
            )

        self._drop_empty_cash_lots()

    def individual_cash(self, individual_id: str) -> float:
        return sum(
            cash.amount
            for cash in self.cash_lots.values()
            if cash.owner_type == OwnerType.INDIVIDUAL and cash.owner_id == individual_id
        )

    def partnership_cash(self, partnership_id: str) -> float:
        return sum(
            cash.amount
            for cash in self.cash_lots.values()
            if cash.owner_type == OwnerType.PARTNERSHIP and cash.owner_id == partnership_id
        )

    def cash_distributed_to(self, individual_id: str) -> float:
        return sum(
            cash.amount
            for cash in self.cash_lots.values()
            if cash.source == CashSource.DISTRIBUTION
            and cash.owner_type == OwnerType.INDIVIDUAL
            and cash.owner_id == individual_id
        )

    def debt_financed_distribution_to(self, individual_id: str) -> float:
        return sum(
            cash.amount
            for cash in self.cash_lots.values()
            if cash.source == CashSource.DISTRIBUTION
            and cash.owner_type == OwnerType.INDIVIDUAL
            and cash.owner_id == individual_id
            and cash.loan_id is not None
        )

    def guaranteed_liability_share(self, individual_id: str) -> float:
        return sum(
            loan.principal
            for loan in self.loans.values()
            if loan.guarantor_id == individual_id
        )

    def capital_account(self, partnership_id: str, partner_id: str) -> float:
        partnership = self._get_partnership(partnership_id)
        if partner_id not in partnership.partner_ids:
            raise ValueError(f"{partner_id} is not a partner in {partnership_id}")
        return partnership.capital_accounts.get(partner_id, 0.0)

    def non_transferor_positive_capital(
        self,
        partnership_id: str,
        transferor_id: str,
    ) -> float:
        partnership = self._get_partnership(partnership_id)
        return sum(
            max(0.0, capital)
            for partner_id, capital in partnership.capital_accounts.items()
            if partner_id != transferor_id
        )

    def has_non_transferor_economic_interest(
        self,
        partnership_id: str,
        transferor_id: str,
    ) -> bool:
        return self.non_transferor_positive_capital(
            partnership_id=partnership_id,
            transferor_id=transferor_id,
        ) > 0.0

    def has_completed_economic_sale_to_partnership(self, transferor_id: str) -> bool:
        asset = self.contributed_asset_by(transferor_id)
        if asset is None:
            return False

        return (
            self.cash_distributed_to(transferor_id) >= asset.fair_market_value
            and self.has_non_transferor_economic_interest(
                partnership_id=asset.contributed_to_partnership_id,
                transferor_id=transferor_id,
            )
        )

    def contributed_asset_by(self, individual_id: str) -> Asset | None:
        for asset in self.assets.values():
            if (
                asset.contributed_by_id == individual_id
                and asset.contributed_to_partnership_id is not None
            ):
                return asset
        return None

    def taxpayer_tax_input(self) -> dict[str, float]:
        asset = self.contributed_asset_by(self.taxpayer_id)
        cash_distributed = self.cash_distributed_to(self.taxpayer_id)

        if asset is None or cash_distributed <= 0.0:
            asset_basis = 0.0
            asset_fmv = 0.0
        else:
            asset_basis = asset.basis
            asset_fmv = asset.fair_market_value

        return {
            "asset_basis": asset_basis,
            "asset_fair_market_value": asset_fmv,
            "cash_distributed_to_transferor": cash_distributed,
            "debt_financed_distribution": self.debt_financed_distribution_to(
                self.taxpayer_id
            ),
            "transferor_allocable_liability": self.guaranteed_liability_share(
                self.taxpayer_id
            ),
            "tax_rate": self.individuals[self.taxpayer_id].tax_rate,
        }

    def _spend_individual_cash(self, individual_id: str, amount: float) -> None:
        if amount <= 0.0:
            raise ValueError("Cash amount must be positive")
        if self.individual_cash(individual_id) < amount:
            raise ValueError("Individual does not have enough cash")

        remaining = amount
        for cash in list(self.cash_lots.values()):
            if remaining <= 0.0:
                break
            if cash.owner_type != OwnerType.INDIVIDUAL or cash.owner_id != individual_id:
                continue

            used = min(cash.amount, remaining)
            remaining -= used
            cash.amount -= used

        self._drop_empty_cash_lots()

    def _drop_empty_cash_lots(self) -> None:
        empty_ids = [
            cash_id
            for cash_id, cash in self.cash_lots.items()
            if cash.amount <= 0.0
        ]
        for cash_id in empty_ids:
            del self.cash_lots[cash_id]

    def _get_partnership(self, partnership_id: str) -> Partnership:
        if partnership_id not in self.partnerships:
            raise ValueError(f"Unknown partnership: {partnership_id}")
        return self.partnerships[partnership_id]

    def _validate_owner(self, owner_type: OwnerType, owner_id: str) -> None:
        if owner_type == OwnerType.INDIVIDUAL and owner_id not in self.individuals:
            raise ValueError(f"Unknown individual owner: {owner_id}")
        if owner_type == OwnerType.PARTNERSHIP and owner_id not in self.partnerships:
            raise ValueError(f"Unknown partnership owner: {owner_id}")

    @classmethod
    def initial_state(cls) -> WorldState:
        state = cls()
        state.add_individual("T", tax_rate=0.20)
        state.add_individual("Buyer", tax_rate=0.0)
        state.add_partnership("P", {"T", "Buyer"})
        state.add_asset(
            asset_id="appreciated_asset",
            basis=0.0,
            fair_market_value=100.0,
            owner_type=OwnerType.INDIVIDUAL,
            owner_id="T",
        )
        state.add_cash_lot(
            amount=100.0,
            owner_type=OwnerType.INDIVIDUAL,
            owner_id="Buyer",
            source=CashSource.INITIAL,
        )
        return state
