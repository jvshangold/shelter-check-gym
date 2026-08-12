from straddle_abuse.tax_env.env import TaxEnv


class HardTaxEnv(TaxEnv):
    """Straddle-abuse environment that learns legality without action masks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_action_masks = False
        self.action_reward_penalty = 0.1 * self.success_tax_advantage
