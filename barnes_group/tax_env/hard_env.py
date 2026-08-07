from barnes_group.tax_env.env import TaxEnv


class HardTaxEnv(TaxEnv):
    """Barnes Group environment at the minimal feasible unscaffolded start.

    The current action space cannot create the foreign CFC, so hard mode keeps
    T and FSub but does not pre-create the domestic subsidiary used in the
    Barnes transaction.
    """

    pass
