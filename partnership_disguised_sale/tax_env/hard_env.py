from partnership_disguised_sale.tax_env.env import TaxEnv
from partnership_disguised_sale.tax_env.state import CashSource, OwnerType, WorldState


class HardTaxEnv(TaxEnv):
    """Partnership disguised-sale environment with decoy contributed assets."""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options={})
        self.state = self._hard_initial_state()
        self.steps = 0
        self.prev_tax_advantage = 0.0
        self._refresh_indices()
        return self.get_observation(), {}

    @staticmethod
    def _hard_initial_state() -> WorldState:
        state = WorldState()
        state.add_individual("T", tax_rate=0.20)
        state.add_individual("Buyer", tax_rate=0.0)
        state.add_partnership("P", {"T", "Buyer"})

        asset_specs = [
            ("appreciated_asset", 0.0, 100.0, "T"),
            ("neutral_asset", 100.0, 100.0, "T"),
            ("loss_asset", 120.0, 80.0, "T"),
            ("buyer_asset", 0.0, 100.0, "Buyer"),
        ]
        for asset_id, basis, fair_market_value, owner_id in asset_specs:
            state.add_asset(
                asset_id=asset_id,
                basis=basis,
                fair_market_value=fair_market_value,
                owner_type=OwnerType.INDIVIDUAL,
                owner_id=owner_id,
            )

        state.add_cash_lot(
            amount=100.0,
            owner_type=OwnerType.INDIVIDUAL,
            owner_id="Buyer",
            source=CashSource.INITIAL,
        )
        return state
