from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaxResidence(Enum):
    US = "US"
    FOREIGN = "Foreign"


class PropertyKind(Enum):
    CASH = "Cash"
    STOCK = "Stock"


@dataclass
class Corporation:
    id: str
    tax_residence: TaxResidence
    parent_id: str | None = None

    @property
    def is_domestic(self) -> bool:
        return self.tax_residence == TaxResidence.US

    @property
    def is_foreign(self) -> bool:
        return self.tax_residence == TaxResidence.FOREIGN


@dataclass
class Cash:
    id: str
    owner_id: str
    amount: float
    basis: float
    contributed_by_id: str | None = None


@dataclass
class Stock:
    id: str
    issuer_id: str
    holder_id: str
    percent: float
    basis: float
    contributed_by_id: str | None = None

    @property
    def is_own_stock(self) -> bool:
        return self.issuer_id == self.holder_id


@dataclass
class TaxLedger:
    direct_repatriated_cash: float = 0.0
    direct_cash_inclusion: float = 0.0
    section_956_inclusion: float = 0.0
    total_inclusion: float = 0.0
    tax_due: float = 0.0
    tax_paid: float = 0.0


@dataclass
class WorldState:
    corporations: dict[str, Corporation] = field(default_factory=dict)
    cash: dict[str, Cash] = field(default_factory=dict)
    stock: dict[str, Stock] = field(default_factory=dict)
    ledger: TaxLedger = field(default_factory=TaxLedger)

    taxpayer_id: str = "T"
    cfc_id: str = "FSub"
    applicable_earnings: float = 100.0
    tax_rate: float = 0.35
    next_cash_id: int = 0
    next_stock_id: int = 0

    def add_corporation(
        self,
        corporation_id: str,
        tax_residence: TaxResidence,
    ) -> None:
        if corporation_id in self.corporations:
            raise ValueError(f"Corporation already exists: {corporation_id}")

        self.corporations[corporation_id] = Corporation(
            id=corporation_id,
            tax_residence=tax_residence,
        )

    def add_cash(
        self,
        owner_id: str,
        amount: float,
        basis: float | None = None,
        cash_id: str | None = None,
        contributed_by_id: str | None = None,
    ) -> str:
        self._require_corporation(owner_id)
        if amount < 0:
            raise ValueError("Cash amount cannot be negative")

        if basis is None:
            basis = amount

        if cash_id is None:
            cash_id = f"cash_{self.next_cash_id}"
            self.next_cash_id += 1

        if cash_id in self.cash:
            raise ValueError(f"Cash already exists: {cash_id}")

        self.cash[cash_id] = Cash(
            id=cash_id,
            owner_id=owner_id,
            amount=amount,
            basis=basis,
            contributed_by_id=contributed_by_id,
        )
        return cash_id

    def add_stock(
        self,
        issuer_id: str,
        holder_id: str,
        percent: float,
        basis: float,
        stock_id: str | None = None,
        contributed_by_id: str | None = None,
    ) -> str:
        self._require_corporation(issuer_id)
        self._require_corporation(holder_id)
        if percent <= 0:
            raise ValueError("Stock percent must be positive")

        if stock_id is None:
            stock_id = f"stock_{self.next_stock_id}"
            self.next_stock_id += 1

        if stock_id in self.stock:
            raise ValueError(f"Stock already exists: {stock_id}")

        self.stock[stock_id] = Stock(
            id=stock_id,
            issuer_id=issuer_id,
            holder_id=holder_id,
            percent=percent,
            basis=basis,
            contributed_by_id=contributed_by_id,
        )
        return stock_id

    def add_subcorporation(
        self,
        parent_id: str,
        sub_id: str,
        tax_residence: TaxResidence = TaxResidence.US,
    ) -> None:
        self._require_corporation(parent_id)
        if sub_id in self.corporations:
            raise ValueError(f"Corporation already exists: {sub_id}")

        self.corporations[sub_id] = Corporation(
            id=sub_id,
            tax_residence=tax_residence,
            parent_id=parent_id,
        )
        self.add_stock(
            issuer_id=sub_id,
            holder_id=parent_id,
            percent=100.0,
            basis=0.0,
            stock_id=f"{sub_id}_stock_parent",
        )

    def transfer_cash(self, from_id: str, to_id: str, amount: float) -> None:
        self._require_corporation(from_id)
        self._require_corporation(to_id)
        if amount <= 0:
            raise ValueError("Transfer amount must be positive")

        source = self._find_cash_lot(from_id, amount)
        source.amount -= amount
        source.basis -= amount

        self.add_cash(
            owner_id=to_id,
            amount=amount,
            basis=amount,
            contributed_by_id=from_id,
        )

        if source.amount == 0:
            del self.cash[source.id]

    def transfer_stock(self, stock_id: str, to_id: str, percent: float) -> str:
        if stock_id not in self.stock:
            raise ValueError(f"Unknown stock: {stock_id}")
        self._require_corporation(to_id)
        if percent <= 0:
            raise ValueError("Transfer percent must be positive")

        source = self.stock[stock_id]
        if percent > source.percent:
            raise ValueError("Cannot transfer more stock than the holder owns")

        basis_fraction = percent / source.percent
        transferred_basis = source.basis * basis_fraction
        source.percent -= percent
        source.basis -= transferred_basis

        new_stock_id = self.add_stock(
            issuer_id=source.issuer_id,
            holder_id=to_id,
            percent=percent,
            basis=transferred_basis,
            contributed_by_id=source.holder_id,
        )

        if source.percent == 0:
            del self.stock[stock_id]

        return new_stock_id

    def issue_stock(self, issuer_id: str, recipient_id: str, percent: float) -> str:
        self._require_corporation(issuer_id)
        self._require_corporation(recipient_id)
        if percent <= 0 or percent > 100:
            raise ValueError("Issued stock percent must be between 0 and 100")
        if not self.has_contributed_property(issuer_id, recipient_id):
            raise ValueError("Stock issuance requires property contributed by recipient")

        existing_total = self.total_issued_percent(issuer_id)
        if existing_total > 0:
            dilution = (100.0 - percent) / existing_total
            for stock in self.stock.values():
                if stock.issuer_id == issuer_id:
                    stock.percent *= dilution

        basis = self._basis_for_stock_issued_in_exchange(
            issuer_id=issuer_id,
            recipient_id=recipient_id,
            recipient_percent=percent,
        )

        stock_id = self.add_stock(
            issuer_id=issuer_id,
            holder_id=recipient_id,
            percent=percent,
            basis=basis,
        )
        return stock_id

    def record_direct_repatriation(self, amount: float) -> None:
        self.ledger.direct_repatriated_cash += amount

    def pay_incremental_tax(self, tax_due: float) -> None:
        incremental_tax = tax_due - self.ledger.tax_paid
        if incremental_tax > 0:
            self._pay_tax(incremental_tax)

    def section_956_us_property_basis(self) -> float:
        cfc = self.cfc_id
        if not self.is_cfc(cfc):
            return 0.0

        total = 0.0
        for stock in self.stock.values():
            if stock.holder_id != cfc:
                continue
            issuer = self.corporations[stock.issuer_id]
            if issuer.is_domestic:
                total += max(0.0, stock.basis)
        return total

    def is_cfc(self, corporation_id: str) -> bool:
        self._require_corporation(corporation_id)
        corporation = self.corporations[corporation_id]
        if not corporation.is_foreign:
            return False
        return self.ownership_percent(self.taxpayer_id, corporation_id) > 50.0

    def ownership_percent(self, holder_id: str, issuer_id: str) -> float:
        return sum(
            stock.percent
            for stock in self.stock.values()
            if stock.holder_id == holder_id and stock.issuer_id == issuer_id
        )

    def total_issued_percent(self, issuer_id: str) -> float:
        return sum(
            stock.percent
            for stock in self.stock.values()
            if stock.issuer_id == issuer_id
        )

    def cash_amount(self, owner_id: str) -> float:
        return sum(cash.amount for cash in self.cash.values() if cash.owner_id == owner_id)

    def subsidiaries_of(self, parent_id: str) -> list[Corporation]:
        self._require_corporation(parent_id)
        return [
            corporation
            for corporation in self.corporations.values()
            if corporation.parent_id == parent_id
        ]

    def is_subcorporation_of(self, sub_id: str, parent_id: str) -> bool:
        self._require_corporation(sub_id)
        self._require_corporation(parent_id)
        return self.corporations[sub_id].parent_id == parent_id

    def has_barnes_exchange(self, issuer_id: str, recipient_id: str, percent: float) -> bool:
        if percent < 80.0:
            return False

        contributed_cash = any(
            cash.owner_id == issuer_id
            and cash.contributed_by_id == recipient_id
            for cash in self.cash.values()
        )
        contributed_own_stock = any(
            stock.holder_id == issuer_id
            and stock.issuer_id == recipient_id
            and stock.contributed_by_id == recipient_id
            for stock in self.stock.values()
        )

        return contributed_cash and contributed_own_stock

    def has_contributed_property(self, issuer_id: str, contributor_id: str) -> bool:
        return any(
            cash.owner_id == issuer_id
            and cash.contributed_by_id == contributor_id
            for cash in self.cash.values()
        ) or any(
            stock.holder_id == issuer_id
            and stock.contributed_by_id == contributor_id
            for stock in self.stock.values()
        )

    def has_qualifying_zero_basis_cfc_stock(self, issuer_id: str) -> bool:
        return any(
            stock.holder_id == self.cfc_id
            and stock.issuer_id == issuer_id
            and stock.percent >= 80.0
            and stock.basis == 0.0
            for stock in self.stock.values()
        )

    def direct_transfer_baseline_net(self) -> float:
        return self.applicable_earnings * (1.0 - self.tax_rate)

    @classmethod
    def initial_state(
        cls,
        t_cash: float = 100.0,
        fsub_cash: float = 100.0,
        applicable_earnings: float = 100.0,
        tax_rate: float = 0.35,
    ) -> WorldState:
        state = cls(applicable_earnings=applicable_earnings, tax_rate=tax_rate)
        state.add_corporation("T", TaxResidence.US)
        state.add_subcorporation("T", "FSub", TaxResidence.FOREIGN)
        state.add_cash("T", t_cash, t_cash, cash_id="t_cash")
        state.add_cash("FSub", fsub_cash, fsub_cash, cash_id="fsub_cash")
        state.add_stock("FSub", "FSub", 100.0, 0.0, stock_id="fsub_own_stock")
        return state

    def _basis_for_stock_issued_in_exchange(
        self,
        issuer_id: str,
        recipient_id: str,
        recipient_percent: float,
    ) -> float:
        if self.has_barnes_exchange(issuer_id, recipient_id, recipient_percent):
            return 0.0

        property_basis = 0.0
        for cash in self.cash.values():
            if cash.owner_id == issuer_id and cash.contributed_by_id == recipient_id:
                property_basis += cash.basis
        for stock in self.stock.values():
            if stock.holder_id == issuer_id and stock.contributed_by_id == recipient_id:
                property_basis += stock.basis
        return property_basis

    def _find_cash_lot(self, owner_id: str, amount: float) -> Cash:
        for cash in self.cash.values():
            if cash.owner_id == owner_id and cash.amount >= amount:
                return cash
        raise ValueError(f"{owner_id} does not have enough cash")

    def _pay_tax(self, amount: float) -> None:
        cash = self._find_cash_lot(self.taxpayer_id, amount)
        cash.amount -= amount
        cash.basis -= amount
        self.ledger.tax_paid += amount
        if cash.amount == 0:
            del self.cash[cash.id]

    def _require_corporation(self, corporation_id: str) -> None:
        if corporation_id not in self.corporations:
            raise ValueError(f"Unknown corporation: {corporation_id}")
