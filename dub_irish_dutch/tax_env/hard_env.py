from dub_irish_dutch.tax_env.env import TaxEnv


class HardTaxEnv(TaxEnv):
    """Dub Irish Dutch environment without the seeded Bermuda holding company."""

    def __init__(self, **kwargs):
        kwargs.setdefault("START_WITH_BERMUDA_HOLDING", False)
        super().__init__(**kwargs)
