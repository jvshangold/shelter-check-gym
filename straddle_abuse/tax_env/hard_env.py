from straddle_abuse.tax_env.env import TaxEnv


class HardTaxEnv(TaxEnv):
    """Straddle-abuse environment without pre-invested facilitators."""

    def __init__(self, **kwargs):
        kwargs["INITIAL_FACILITATOR_INVESTMENT"] = 0.0
        super().__init__(**kwargs)
