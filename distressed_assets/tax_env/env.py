import gymnasium as gym
from gymnasium import Space, spaces

from typing import Dict, List

from .state import WorldState

# action list [
# MAKE_SUBTRUST,
# MOVE_ASSET,
# SELL_ASSET,
# MOVE_CASH_INTO_TRUST,
# GIVE_VESTING_POWER
# ]

# random event: GENERATE_FOREIGN_PARTY

class TaxEnv(gym.env):
    def __init__(self):
        super().__init__()