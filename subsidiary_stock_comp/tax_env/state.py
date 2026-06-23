from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaxResidence(Enum):
    US = "US"
    FOREIGN = "Foreign"


@dataclass
class Corporation:
    id: str
    tax_residence: TaxResidence
    parent_id: str | None = None


@dataclass
class Cash:
    id: str
    owner_id: str
    amount: float
    basis: float


@dataclass
class Stock:
    id: str
    issuer_id: str
    holder_id: str
    fmv: float
    basis: float
    percent: float


@dataclass
class TaxLedger:
    ordinary_deductions: dict[str, float] = field(default_factory=dict)
    capital_losses: dict[str, float] = field(default_factory=dict)
    capital_gains: dict[str, float] = field(default_factory=dict)

    def add_ordinary_deduction(self, corporation_id: str, amount: float) -> None:
        self.ordinary_deductions[corporation_id] = (
            self.ordinary_deductions.get(corporation_id, 0.0) + amount
        )

    def add_capital_result(self, corporation_id: str, amount: float) -> None:
        if amount < 0:
            self.capital_losses[corporation_id] = (
                self.capital_losses.get(corporation_id, 0.0) + abs(amount)
            )
        elif amount > 0:
            self.capital_gains[corporation_id] = (
                self.capital_gains.get(corporation_id, 0.0) + amount
            )

    @property
    def total_ordinary_deductions(self) -> float:
        return sum(self.ordinary_deductions.values())

    @property
    def total_capital_losses(self) -> float:
        return sum(self.capital_losses.values())

    @property
    def total_capital_gains(self) -> float:
        return sum(self.capital_gains.values())

    @property
    def total_tax_advantage(self) -> float:
        return self.total_ordinary_deductions + self.total_capital_losses


@dataclass
class WorldState:
    corporations: dict[str, Corporation] = field(default_factory=dict)
    cash: dict[str, Cash] = field(default_factory=dict)
    stock: dict[str, Stock] = field(default_factory=dict)
    ledger: TaxLedger = field(default_factory=TaxLedger)

    taxpayer_id: str = "P"
    subsidiary_id: str = "X"
    market_id: str = "stock_market"
    next_cash_id: int = 0
    next_stock_id: int = 0
    next_subcorp_id: int = 0
    public_market_value: float = 10_000.0

    def add_corporation(
        self,
        corporation_id: str,
        tax_residence: TaxResidence = TaxResidence.US,
        parent_id: str | None = None,
    ) -> None:
        if corporation_id in self.corporations:
            raise ValueError(f"Corporation already exists: {corporation_id}")
        if parent_id is not None:
            self._require_corporation(parent_id)

        self.corporations[corporation_id] = Corporation(
            id=corporation_id,
            tax_residence=tax_residence,
            parent_id=parent_id,
        )

    def add_cash(
        self,
        owner_id: str,
        amount: float,
        basis: float | None = None,
        cash_id: str | None = None,
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
        )
        return cash_id

    def add_stock(
        self,
        issuer_id: str,
        holder_id: str,
        fmv: float,
        basis: float,
        percent: float,
        stock_id: str | None = None,
    ) -> str:
        self._require_corporation(issuer_id)
        self._require_corporation(holder_id)
        if fmv < 0:
            raise ValueError("Stock FMV cannot be negative")
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
            fmv=fmv,
            basis=basis,
            percent=percent,
        )
        return stock_id

    def form_subcorp(
        self,
        contributor_a_id: str,
        contributor_b_id: str,
        contribution_a: float,
        contribution_b: float,
    ) -> str:
        self._require_corporation(contributor_a_id)
        self._require_corporation(contributor_b_id)
        if contributor_a_id == contributor_b_id:
            raise ValueError("Formation requires two distinct contributors")
        if contribution_a <= 0 or contribution_b <= 0:
            raise ValueError("Formation contributions must be positive")

        total = contribution_a + contribution_b
        subcorp_id = f"S_{self.next_subcorp_id}"
        self.next_subcorp_id += 1
        self.add_corporation(subcorp_id, TaxResidence.US)

        self.transfer_cash(contributor_a_id, subcorp_id, contribution_a)
        self.transfer_cash(contributor_b_id, subcorp_id, contribution_b)

        self.add_stock(
            issuer_id=subcorp_id,
            holder_id=contributor_a_id,
            fmv=contribution_a,
            basis=contribution_a,
            percent=contribution_a / total * 100.0,
        )
        self.add_stock(
            issuer_id=subcorp_id,
            holder_id=contributor_b_id,
            fmv=contribution_b,
            basis=contribution_b,
            percent=contribution_b / total * 100.0,
        )
        return subcorp_id

    def transfer_cash(self, from_id: str, to_id: str, amount: float) -> str:
        self._require_corporation(from_id)
        self._require_corporation(to_id)
        if amount <= 0:
            raise ValueError("Transfer amount must be positive")

        remaining = amount
        transferred_basis = 0.0
        for cash_id, cash in list(self.cash.items()):
            if cash.owner_id != from_id:
                continue
            used = min(cash.amount, remaining)
            basis_fraction = used / cash.amount
            used_basis = cash.basis * basis_fraction
            cash.amount -= used
            cash.basis -= used_basis
            remaining -= used
            transferred_basis += used_basis
            if cash.amount == 0:
                del self.cash[cash_id]
            if remaining == 0:
                return self.add_cash(to_id, amount, transferred_basis)

        raise ValueError(f"{from_id} does not have enough cash")

    def buy_stock(self, buyer_id: str, issuer_id: str, amount: float) -> str:
        self._require_corporation(buyer_id)
        self._require_corporation(issuer_id)
        if amount <= 0:
            raise ValueError("Stock purchase amount must be positive")

        self._spend_cash(buyer_id, amount)
        percent = amount / self.public_market_value * 100.0
        return self.add_stock(
            issuer_id=issuer_id,
            holder_id=buyer_id,
            fmv=amount,
            basis=amount,
            percent=percent,
        )

    def compensate_employees(
        self,
        holder_id: str,
        service_recipient_id: str,
        stock_id: str,
        amount: float,
    ) -> None:
        self._require_corporation(holder_id)
        self._require_corporation(service_recipient_id)
        if amount <= 0:
            raise ValueError("Compensation amount must be positive")
        if stock_id not in self.stock:
            raise ValueError(f"Unknown stock: {stock_id}")

        stock = self.stock[stock_id]
        if stock.holder_id != holder_id:
            raise ValueError("Holder must own compensated stock")
        if stock.issuer_id != service_recipient_id:
            raise ValueError("Stock must be stock of the service recipient")
        if amount > stock.fmv:
            raise ValueError("Cannot compensate more stock value than the holder owns")

        self._remove_stock_value(stock_id, amount)
        self.ledger.add_ordinary_deduction(service_recipient_id, amount)

    def liquidate_corp(self, corporation_id: str) -> None:
        self._require_corporation(corporation_id)
        if corporation_id in {self.taxpayer_id, self.subsidiary_id}:
            raise ValueError("Initial corporations are not liquidation targets")

        shareholder_stock = [
            stock
            for stock in self.stock.values()
            if stock.issuer_id == corporation_id and stock.holder_id != corporation_id
        ]
        if not shareholder_stock:
            raise ValueError("Liquidation requires outstanding shareholder stock")

        total_percent = sum(stock.percent for stock in shareholder_stock)
        liquidating_assets = self._asset_value(corporation_id)

        for stock in list(shareholder_stock):
            share = stock.percent / total_percent
            amount_received = liquidating_assets * share
            if self._section_331_applies(stock.holder_id, corporation_id):
                self.ledger.add_capital_result(
                    stock.holder_id,
                    amount_received - stock.basis,
                )

        self._distribute_cash_on_liquidation(corporation_id, shareholder_stock, total_percent)
        self._distribute_stock_on_liquidation(corporation_id, shareholder_stock, total_percent)

        for stock_id, stock in list(self.stock.items()):
            if stock.issuer_id == corporation_id or stock.holder_id == corporation_id:
                del self.stock[stock_id]
        for cash_id, cash in list(self.cash.items()):
            if cash.owner_id == corporation_id:
                del self.cash[cash_id]
        del self.corporations[corporation_id]

    def cash_amount(self, owner_id: str) -> float:
        return sum(cash.amount for cash in self.cash.values() if cash.owner_id == owner_id)

    def ownership_percent(self, holder_id: str, issuer_id: str) -> float:
        return sum(
            stock.percent
            for stock in self.stock.values()
            if stock.holder_id == holder_id and stock.issuer_id == issuer_id
        )

    def stock_fmv(self, holder_id: str, issuer_id: str) -> float:
        return sum(
            stock.fmv
            for stock in self.stock.values()
            if stock.holder_id == holder_id and stock.issuer_id == issuer_id
        )

    @classmethod
    def initial_state(cls) -> WorldState:
        state = cls()
        state.add_corporation("P", TaxResidence.US)
        state.add_corporation("X", TaxResidence.US, parent_id="P")
        state.add_cash("P", 79.0, 79.0, cash_id="p_cash")
        state.add_cash("X", 21.0, 21.0, cash_id="x_cash")
        state.add_stock(
            issuer_id="X",
            holder_id="P",
            fmv=21.0,
            basis=0.0,
            percent=100.0,
            stock_id="p_owns_x",
        )
        return state

    def _asset_value(self, corporation_id: str) -> float:
        return self.cash_amount(corporation_id) + sum(
            stock.fmv for stock in self.stock.values() if stock.holder_id == corporation_id
        )

    def _section_331_applies(self, shareholder_id: str, corporation_id: str) -> bool:
        return self.ownership_percent(shareholder_id, corporation_id) < 80.0

    def _distribute_cash_on_liquidation(
        self,
        corporation_id: str,
        shareholder_stock: list[Stock],
        total_percent: float,
    ) -> None:
        cash_lots = [cash for cash in self.cash.values() if cash.owner_id == corporation_id]
        for cash in cash_lots:
            for stock in shareholder_stock:
                share = stock.percent / total_percent
                self.add_cash(
                    owner_id=stock.holder_id,
                    amount=cash.amount * share,
                    basis=cash.basis * share,
                )

    def _distribute_stock_on_liquidation(
        self,
        corporation_id: str,
        shareholder_stock: list[Stock],
        total_percent: float,
    ) -> None:
        stock_lots = [stock for stock in self.stock.values() if stock.holder_id == corporation_id]
        for asset_stock in stock_lots:
            for shareholder in shareholder_stock:
                share = shareholder.percent / total_percent
                self.add_stock(
                    issuer_id=asset_stock.issuer_id,
                    holder_id=shareholder.holder_id,
                    fmv=asset_stock.fmv * share,
                    basis=asset_stock.basis * share,
                    percent=asset_stock.percent * share,
                )

    def _remove_stock_value(self, stock_id: str, amount: float) -> None:
        stock = self.stock[stock_id]
        fraction = amount / stock.fmv
        stock.fmv -= amount
        stock.basis -= stock.basis * fraction
        stock.percent -= stock.percent * fraction
        if stock.fmv == 0:
            del self.stock[stock_id]

    def _spend_cash(self, owner_id: str, amount: float) -> None:
        remaining = amount
        for cash_id, cash in list(self.cash.items()):
            if cash.owner_id != owner_id:
                continue
            used = min(cash.amount, remaining)
            basis_fraction = used / cash.amount
            cash.amount -= used
            cash.basis -= cash.basis * basis_fraction
            remaining -= used
            if cash.amount == 0:
                del self.cash[cash_id]
            if remaining == 0:
                return
        raise ValueError(f"{owner_id} does not have enough cash")

    def _find_cash_lot(self, owner_id: str, amount: float) -> Cash:
        for cash in self.cash.values():
            if cash.owner_id == owner_id and cash.amount >= amount:
                return cash
        raise ValueError(f"{owner_id} does not have enough cash")

    def _require_corporation(self, corporation_id: str) -> None:
        if corporation_id not in self.corporations:
            raise ValueError(f"Unknown corporation: {corporation_id}")
