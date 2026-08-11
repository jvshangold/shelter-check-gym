from dub_irish_dutch.tax_env.env import TaxEnv


class HardTaxEnv(TaxEnv):
    """Dub Irish Dutch environment that learns legality without action masks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_action_masks = False
