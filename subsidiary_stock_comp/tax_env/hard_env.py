from subsidiary_stock_comp.tax_env.env import TaxEnv


class HardTaxEnv(TaxEnv):
    """Subsidiary-stock-comp environment that learns legality without masks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_action_masks = False
