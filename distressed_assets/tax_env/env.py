import sys
from typing import Dict

import gymnasium as gym
from gymnasium import Space, spaces

from .state import WorldState, OwnerType, AssetKind, TaxResidence
from .render import build_graph

sys.path.append("distressed_assets/formalizations/_target/distressed_tax_rules")
sys.path.append("distressed_assets/formalizations/_build/libcatala/python")

from python import DistressedAssetTaxModel as TaxModel
from python import catala_runtime


MAKE_SUBTRUST = 0
MOVE_ASSET = 1
SELL_ASSET = 2
GIVE_VESTING_POWER = 3


class TaxEnv(gym.Env):
    def __init__(self, MAX_TRUSTS=5, MAX_ASSETS=5, MAX_INDIVIDUALS=4, MAX_STEPS=10):
        super().__init__()

        self.observation_space: Space = spaces.Dict({})

        self.action_space: Space = spaces.MultiDiscrete([
            4,
            MAX_TRUSTS,
            MAX_ASSETS,
            MAX_INDIVIDUALS,
        ])

        self.max_trusts = MAX_TRUSTS
        self.max_assets = MAX_ASSETS
        self.max_individuals = MAX_INDIVIDUALS
        self.max_steps = MAX_STEPS

        self.state = WorldState.initial_state()

        self.idx_to_trust: Dict[int, str] = {}
        self.idx_to_asset: Dict[int, str] = {}
        self.idx_to_individual: Dict[int, str] = {}

        self.prev_savings = 0.0
        self.steps = 0

        self._refresh_indices()

    def step(self, action):
        action_type, arg_1, arg_2, arg_3 = action

        terminated = False
        truncated = False
        invalid_action = False

        try:
            if action_type == MAKE_SUBTRUST:
                self._make_subtrust(parent_trust_idx=arg_1)

            elif action_type == MOVE_ASSET:
                self._move_asset(asset_idx=arg_2, dst_trust_idx=arg_1)

            elif action_type == SELL_ASSET:
                self._sell_asset(asset_idx=arg_2)

            elif action_type == GIVE_VESTING_POWER:
                self._give_vesting_power(trust_idx=arg_1, individual_idx=arg_3)

            else:
                raise ValueError("Unknown action type")

        except Exception as e:
            print("Action:", action)
            print("Exception:", e)
            invalid_action = True

        current_savings = self.compute_savings()
        reward = current_savings - self.prev_savings
        self.prev_savings = current_savings

        if invalid_action:
            reward -= 1.0

        self.steps += 1

        if self.steps >= self.max_steps:
            truncated = True

        obs = self.get_observation()
        info = {"invalid_action": invalid_action}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = WorldState.initial_state()
        self.steps = 0
        self.prev_savings = 0.0

        self._refresh_indices()

        obs = self.get_observation()
        info = {}

        return obs, info

    def compute_reward(self):
        current_savings = self.compute_savings()
        reward = current_savings - self.prev_savings
        self.prev_savings = current_savings
        return reward

    def compute_savings(self) -> float:
        total = 0.0

        for asset in self.state.assets.values():
            if asset.kind != AssetKind.PROPERTY:
                continue

            if not asset.is_sold:
                continue

            if asset.owner_type != OwnerType.TRUST:
                continue

            trust = self.state.trusts[asset.owner_id]

            if trust.section_678_power_holder_id is None:
                continue

            total += self._compute_catala_savings(
                asset=asset,
                power_holder_id=trust.section_678_power_holder_id,
            )

        return total

    def render(self):
        return self.render_world()

    def get_observation(self):
        return build_graph(state=self.state)

    def render_world(self, filename="world"):
        return {
            "individuals": self.state.individuals,
            "trusts": self.state.trusts,
            "assets": self.state.assets,
        }

    def _make_subtrust(self, parent_trust_idx: int) -> None:
        if len(self.state.trusts) >= self.max_trusts:
            raise ValueError("Max number of trusts reached")

        parent_id = self._get_trust_id(parent_trust_idx)
        trust_id = f"sub_trust_{len(self.state.trusts)}"

        self.state.add_trust(
            trust_id=trust_id,
            parent_trust_id=parent_id,
            section_678_power_holder_id=None,
        )

        self._refresh_indices()

    def _move_asset(self, asset_idx: int, dst_trust_idx: int) -> None:
        asset_id = self._get_asset_id(asset_idx)
        dst_trust_id = self._get_trust_id(dst_trust_idx)

        asset = self.state.assets[asset_id]

        if asset.is_sold:
            raise ValueError("Cannot move an asset after sale")

        self.state.transfer_asset(
            asset_id=asset_id,
            new_owner_type=OwnerType.TRUST,
            new_owner_id=dst_trust_id,
        )

        self._refresh_indices()

    def _sell_asset(self, asset_idx: int) -> None:
        asset_id = self._get_asset_id(asset_idx)
        asset = self.state.assets[asset_id]

        if asset.kind != AssetKind.PROPERTY:
            raise ValueError("Can only sell property assets")

        if asset.is_sold:
            raise ValueError("Asset already sold")

        if asset.owner_type != OwnerType.TRUST:
            raise ValueError("Asset must be inside a trust before sale")

        asset.sale_price = asset.fair_market_value

    def _give_vesting_power(self, trust_idx: int, individual_idx: int) -> None:
        trust_id = self._get_trust_id(trust_idx)
        individual_id = self._get_individual_id(individual_idx)

        trust = self.state.trusts[trust_id]

        if trust.section_678_power_holder_id is not None:
            raise ValueError("Trust already has §678 power holder")

        if trust.parent_trust_id is None:
            raise ValueError("Cannot give §678 power over root trust")

        asset = self._find_property_asset_in_trust(trust_id)

        if asset is None:
            raise ValueError("No property asset in trust")

        if not self._individual_has_sufficient_cash(individual_id, asset.fair_market_value):
            raise ValueError("Individual does not have enough cash to compensate FP")

        foreign_party_id = self._find_foreign_individual_id()

        if foreign_party_id is None:
            raise ValueError("No foreign party found")

        self._pay_cash(
            from_individual_id=individual_id,
            to_individual_id=foreign_party_id,
            amount=asset.fair_market_value,
        )

        self.state.set_section_678_power_holder(
            trust_id=trust_id,
            individual_id=individual_id,
        )

        self._refresh_indices()

    def _pay_cash(
        self,
        from_individual_id: str,
        to_individual_id: str,
        amount: float,
    ) -> None:
        cash = self._find_cash_owned_by_individual(from_individual_id)

        if cash is None:
            raise ValueError("Payer has no cash")

        if cash.fair_market_value < amount:
            raise ValueError("Payer does not have enough cash")

        cash.fair_market_value -= amount
        cash.basis -= amount

        cash_id = f"cash_to_{to_individual_id}_{len(self.state.assets)}"

        self.state.add_asset(
            asset_id=cash_id,
            kind=AssetKind.CASH,
            basis=amount,
            fair_market_value=amount,
            owner_type=OwnerType.INDIVIDUAL,
            owner_id=to_individual_id,
        )

        self._refresh_indices()

    def _compute_catala_savings(self, asset, power_holder_id: str) -> float:
        power_holder = self.state.individuals[power_holder_id]

        catala_asset = TaxModel.Asset(
            basis=self._money(asset.basis),
            fair_market_value=self._money(asset.fair_market_value),
            sale_price=self._money(asset.sale_price),
        )

        arrangement = TaxModel.TrustArrangement(
            transferor=self._catala_individual(TaxResidence.FOREIGN),
            taxpayer=self._catala_individual(power_holder.tax_residence),
            asset=catala_asset,
            taxpayer_has_section_678_power=True,
            taxpayer_bore_economic_loss=False,
        )

        result = TaxModel.distressed_asset_trust_computation(
            TaxModel.DistressedAssetTrustComputationIn(
                arrangement_in=arrangement,
                tax_rate_in=catala_runtime.Decimal("0.20"),
            )
        )

        return self._money_to_float(result.would_be_tax_savings)

    def _catala_individual(self, residence: TaxResidence):
        if residence == TaxResidence.US:
            tax_residence = TaxModel.TaxResidence(
                TaxModel.TaxResidence_Code.US,
                catala_runtime.Unit(),
            )
        else:
            tax_residence = TaxModel.TaxResidence(
                TaxModel.TaxResidence_Code.Foreign,
                catala_runtime.Unit(),
            )

        return TaxModel.Individual(tax_residence=tax_residence)

    def _money(self, amount: float):
        return catala_runtime.Money(catala_runtime.Integer(int(amount)))

    def _money_to_float(self, money) -> float:
        return float(money.value.value)

    def _find_property_asset_in_trust(self, trust_id: str):
        for asset in self.state.assets.values():
            if (
                asset.kind == AssetKind.PROPERTY
                and asset.owner_type == OwnerType.TRUST
                and asset.owner_id == trust_id
                and not asset.is_sold
            ):
                return asset

        return None

    def _find_cash_owned_by_individual(self, individual_id: str):
        for asset in self.state.assets.values():
            if (
                asset.kind == AssetKind.CASH
                and asset.owner_type == OwnerType.INDIVIDUAL
                and asset.owner_id == individual_id
                and asset.fair_market_value > 0
            ):
                return asset

        return None

    def _find_foreign_individual_id(self) -> str | None:
        for individual_id, individual in self.state.individuals.items():
            if individual.tax_residence == TaxResidence.FOREIGN:
                return individual_id

        return None

    def _individual_has_sufficient_cash(self, individual_id: str, amount: float) -> bool:
        cash = self._find_cash_owned_by_individual(individual_id)
        return cash is not None and cash.fair_market_value >= amount

    def _get_trust_id(self, idx: int) -> str:
        if idx not in self.idx_to_trust:
            raise ValueError(f"Invalid trust index: {idx}")
        return self.idx_to_trust[idx]

    def _get_asset_id(self, idx: int) -> str:
        if idx not in self.idx_to_asset:
            raise ValueError(f"Invalid asset index: {idx}")
        return self.idx_to_asset[idx]

    def _get_individual_id(self, idx: int) -> str:
        if idx not in self.idx_to_individual:
            raise ValueError(f"Invalid individual index: {idx}")
        return self.idx_to_individual[idx]

    def _refresh_indices(self) -> None:
        self.idx_to_trust = {
            i: trust_id for i, trust_id in enumerate(self.state.trusts.keys())
        }

        self.idx_to_asset = {
            i: asset_id for i, asset_id in enumerate(self.state.assets.keys())
        }

        self.idx_to_individual = {
            i: individual_id
            for i, individual_id in enumerate(self.state.individuals.keys())
        }