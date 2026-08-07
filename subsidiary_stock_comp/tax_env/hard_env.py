from subsidiary_stock_comp.tax_env.env import TaxEnv
from subsidiary_stock_comp.tax_env.state import TaxResidence, WorldState


HARD_FORMATION_SPLITS = [
    (79.0, 21.0),
    (80.0, 20.0),
    (70.0, 30.0),
    (60.0, 40.0),
    (50.0, 50.0),
    (40.0, 60.0),
    (30.0, 70.0),
    (21.0, 79.0),
    (20.0, 80.0),
    (1.0, 99.0),
]


class HardTaxEnv(TaxEnv):
    """Subsidiary-stock-comp environment without the 79/21 cash hint."""

    def __init__(self, **kwargs):
        kwargs.setdefault("FORMATION_SPLITS", HARD_FORMATION_SPLITS)
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options={})
        self.state = self._hard_initial_state()
        self.prev_normalized_tax_advantage = 0.0
        self.steps = 0
        self._refresh_indices()
        return self.get_observation(), {}

    @staticmethod
    def _hard_initial_state() -> WorldState:
        state = WorldState()
        state.add_corporation("P", TaxResidence.US)
        state.add_corporation("X", TaxResidence.US, parent_id="P")
        state.add_cash("P", 100.0, 100.0, cash_id="p_cash")
        state.add_cash("X", 100.0, 100.0, cash_id="x_cash")
        state.add_stock(
            issuer_id="X",
            holder_id="P",
            fmv=100.0,
            basis=0.0,
            percent=100.0,
            stock_id="p_owns_x",
        )
        return state
