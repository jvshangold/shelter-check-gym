from distressed_assets.tax_env.env import TaxEnv
from distressed_assets.tax_env.state import AssetKind, OwnerType, TaxResidence, WorldState


class HardTaxEnv(TaxEnv):
    """Distressed-assets environment with decoy assets and no random bootstrap."""

    def __init__(self, **kwargs):
        kwargs["RANDOM_FOREIGN_PARTY_PROB"] = 0.0
        kwargs["MAX_RANDOM_FOREIGN_PARTIES"] = 0
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options={})
        self.state = self._hard_initial_state()
        self.steps = 0
        self.prev_tax_advantage = 0.0
        self._refresh_indices()
        return self.get_observation(), {}

    @staticmethod
    def _hard_initial_state() -> WorldState:
        state = WorldState.initial_state()

        asset_specs = [
            ("FP_0", "distressed_property_0", 200.0, 40.0),
            ("FP_1", "neutral_property_1", 100.0, 100.0),
            ("FP_2", "gain_property_2", 40.0, 100.0),
        ]

        for foreign_party_id, asset_id, basis, fair_market_value in asset_specs:
            state.add_individual(foreign_party_id, TaxResidence.FOREIGN)
            state.add_asset(
                asset_id=asset_id,
                kind=AssetKind.PROPERTY,
                basis=basis,
                fair_market_value=fair_market_value,
                owner_type=OwnerType.INDIVIDUAL,
                owner_id=foreign_party_id,
            )

        return state
