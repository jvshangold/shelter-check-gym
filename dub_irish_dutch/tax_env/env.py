import gymnasium as gym
from gymnasium import Space, spaces

from typing import Dict, List
from pathlib import Path

from .state import WorldState
from .render import build_graph

import sys

from graphviz import Digraph

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CATALA_TARGET_ROOT = PACKAGE_ROOT / "formalizations" / "_target" / "dutch_tax_rules"
CATALA_TARGET_PYTHON = CATALA_TARGET_ROOT / "python"

for path in (CATALA_TARGET_ROOT, CATALA_TARGET_PYTHON):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.append(path_text)

from python import TaxModel
from python import catala_runtime


class TaxEnv(gym.Env):
    def __init__(
        self,
        MAX_ENTITIES=4,
        JURISDICTIONS=5,
        MAX_STEPS=10,
        START_WITH_BERMUDA_HOLDING=True,
    ):
        super().__init__()

        self.observation_space: Space = spaces.Dict({})

        # action = [
        #   action_type,
        #   arg_1,
        #   arg_2,
        #   incorporation_jurisdiction,
        #   management_jurisdiction,
        #   company_type,
        # ]
        #
        # action_type:
        #   0 = add_child
        #   1 = rent_ip
        #   2 = transfer_ip
        #
        # company_type:
        #   0 = Holding
        #   1 = Operating
        self.action_space: Space = spaces.MultiDiscrete([
            3,
            MAX_ENTITIES,
            MAX_ENTITIES,
            JURISDICTIONS,
            JURISDICTIONS,
            2,
        ])

        self.state = None

        self.max_steps = MAX_STEPS
        self.max_entities = MAX_ENTITIES
        self.start_with_bermuda_holding = START_WITH_BERMUDA_HOLDING

        self.idx_to_entity: Dict[int, str] = {}
        self.idx_to_jurisdiction: Dict[int, str] = {
            0: "Ireland",
            1: "Netherlands",
            2: "Bermuda",
            3: "US",
            4: "Germany",
        }

        self.idx_to_company_type: Dict[int, str] = {
            0: "Holding",
            1: "Operating",
        }

        self.prev_normalized_tax_advantage = 0.0
        self.prev_normalized_raw_tax_advantage = 0.0
        self.raw_tax_advantage_reward_weight = 0.25
        self.steps = 0

    def step(self, action):
        action_type, arg_1, arg_2, arg_3, arg_4, arg_5 = action
        invalid_action = False

        try:
            if action_type == 0:
                if len(self.state.entities) >= self.max_entities:
                    raise ValueError("Maximum number of entities reached")

                parent_id = self.idx_to_entity[arg_1]
                incorporation_jurisdiction = self.idx_to_jurisdiction[arg_3]
                management_jurisdiction = self.idx_to_jurisdiction[arg_4]
                company_type = self.idx_to_company_type[arg_5]

                tax_residence = self.state.get_tax_residence(
                    incorporation_jurisdiction,
                    management_jurisdiction,
                )

                new_entity = self.state.add_child(
                    parent=parent_id,
                    incorporation_jurisdiction=incorporation_jurisdiction,
                    management_jurisdiction=management_jurisdiction,
                    tax_residence=tax_residence,
                    company_type=company_type,
                )

                new_idx = len(self.idx_to_entity)
                self.idx_to_entity[new_idx] = new_entity

            elif action_type == 1:
                licensee_id = self.idx_to_entity[arg_1]
                licensor_id = self.idx_to_entity[arg_2]

                if not self.is_valid_rent_pair(licensee_id, licensor_id):
                    raise ValueError("Invalid IP license pair.")

                self.state.rent_ip(licensee_id, licensor_id)

            elif action_type == 2:
                new_owner_id = self.idx_to_entity[arg_2]

                if new_owner_id == self.state.ip_owner:
                    raise ValueError("Cannot transfer IP to current owner.")

                self.state.transfer_ip(new_owner_id)

            else:
                raise ValueError("Unknown action type")

        except Exception as e:
            print("Action:", action)
            print("Exception:", e)
            invalid_action = True

        current_profit = self.money_to_float(self.compute_profit())
        baseline_profit = self.compute_baseline_profit()
        raw_tax_advantage = self.compute_raw_tax_advantage(
            current_profit=current_profit,
            baseline_profit=baseline_profit,
        )
        tax_advantage = self.compute_tax_advantage(
            current_profit=current_profit,
            baseline_profit=baseline_profit,
            raw_tax_advantage=raw_tax_advantage,
        )
        normalized_tax_advantage = self.normalize_tax_advantage(tax_advantage)
        normalized_raw_tax_advantage = self.normalize_tax_advantage(raw_tax_advantage)

        reward = (
            normalized_tax_advantage
            - self.prev_normalized_tax_advantage
            + self.raw_tax_advantage_reward_weight
            * (
                normalized_raw_tax_advantage
                - self.prev_normalized_raw_tax_advantage
            )
            - self.compute_complexity_penalty()
        )
        self.prev_normalized_tax_advantage = normalized_tax_advantage
        self.prev_normalized_raw_tax_advantage = normalized_raw_tax_advantage

        if invalid_action:
            reward -= 1.0

        terminated = tax_advantage > 0.0 and not invalid_action
        truncated = False

        obs = self.get_observation()
        info = {
            "invalid_action": invalid_action,
            "tax_advantage": tax_advantage,
            "raw_tax_advantage": raw_tax_advantage,
            "normalized_tax_advantage": normalized_tax_advantage,
            "normalized_raw_tax_advantage": normalized_raw_tax_advantage,
            "current_profit": current_profit,
            "baseline_profit": baseline_profit,
            "loophole_gate_complete": self.has_completed_loophole_structure(
                current_profit=current_profit,
                baseline_profit=baseline_profit,
                raw_tax_advantage=raw_tax_advantage,
            ),
        }

        self.steps += 1

        if self.steps >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = WorldState()
        self.steps = 0
        self.prev_normalized_tax_advantage = 0.0
        self.prev_normalized_raw_tax_advantage = 0.0
        self.idx_to_entity = {}

        # add US root
        incorporation_jurisdiction = self.idx_to_jurisdiction[3]
        management_jurisdiction = self.idx_to_jurisdiction[3]
        tax_residence = self.state.get_tax_residence(
            incorporation_jurisdiction,
            management_jurisdiction,
        )

        self.state.add_root(
            entity_id="root",
            incorporation_jurisdiction=incorporation_jurisdiction,
            management_jurisdiction=management_jurisdiction,
            tax_residence=tax_residence,
            company_type="Operating",
        )

        self.idx_to_entity[0] = "root"

        if self.start_with_bermuda_holding:
            bermuda = self.state.add_child(
                parent="root",
                incorporation_jurisdiction="Bermuda",
                management_jurisdiction="Bermuda",
                tax_residence="Bermuda",
                company_type="Holding",
            )
            self.idx_to_entity[1] = bermuda

        return self.get_observation(), {}

    def render(self):
        return

    def get_observation(self):
        return build_graph(state=self.state)

    def would_create_license_cycle(self, licensee_id, licensor_id):
        cur = licensor_id

        while cur in self.state.licenses:
            cur = self.state.licenses[cur]

            if cur == licensee_id:
                return True

        return False

    def is_valid_rent_pair(self, licensee_id, licensor_id):
        if licensee_id == licensor_id:
            return False

        if licensee_id in self.state.licenses:
            return False

        if not self.state.has_ip_rights(licensor_id):
            return False

        if self.would_create_license_cycle(licensee_id, licensor_id):
            return False

        return True

    def can_rent_ip(self):
        for licensee_id in self.state.entities:
            for licensor_id in self.state.entities:
                if self.is_valid_rent_pair(licensee_id, licensor_id):
                    return True

        return False

    def can_transfer_ip(self):
        if len(self.idx_to_entity) < 2:
            return False

        for _, entity_id in self.idx_to_entity.items():
            entity = self.state.entities[entity_id]

            if (
                entity_id != self.state.ip_owner
                and entity.company_type == "Holding"
            ):
                return True

        return False

    def get_action_mask(self):
        mask = [1, 1, 1]

        if len(self.state.entities) >= self.max_entities:
            mask[0] = 0

        if not self.can_rent_ip():
            mask[1] = 0

        if not self.can_transfer_ip():
            mask[2] = 0

        return mask

    def get_entity_mask(self):
        return [1 for _ in range(len(self.idx_to_entity))]

    def money_to_float(self, money):
        return float(money.value.value)

    def compute_reward(self):
        tax_advantage = self.compute_tax_advantage()
        raw_tax_advantage = self.compute_raw_tax_advantage()
        normalized_tax_advantage = self.normalize_tax_advantage(tax_advantage)
        normalized_raw_tax_advantage = self.normalize_tax_advantage(raw_tax_advantage)
        reward = (
            normalized_tax_advantage
            - self.prev_normalized_tax_advantage
            + self.raw_tax_advantage_reward_weight
            * (
                normalized_raw_tax_advantage
                - self.prev_normalized_raw_tax_advantage
            )
            - self.compute_complexity_penalty()
        )
        self.prev_normalized_tax_advantage = normalized_tax_advantage
        self.prev_normalized_raw_tax_advantage = normalized_raw_tax_advantage
        return reward

    def compute_complexity_penalty(self):
        penalty = 0.0

        for entity in self.state.entities.values():
            if entity.tax_residence != entity.incorporation_jurisdiction:
                penalty += 0.01

        return penalty

    def normalize_tax_advantage(self, tax_advantage: float) -> float:
        baseline_profit = self.compute_baseline_profit()
        if baseline_profit <= 0.0:
            return tax_advantage

        return tax_advantage / baseline_profit

    def compute_raw_tax_advantage(
        self,
        current_profit: float | None = None,
        baseline_profit: float | None = None,
    ) -> float:
        if current_profit is None:
            current_profit = self.money_to_float(self.compute_profit())
        if baseline_profit is None:
            baseline_profit = self.compute_baseline_profit()

        return max(0.0, current_profit - baseline_profit)

    def compute_tax_advantage(
        self,
        current_profit: float | None = None,
        baseline_profit: float | None = None,
        raw_tax_advantage: float | None = None,
    ) -> float:
        if current_profit is None:
            current_profit = self.money_to_float(self.compute_profit())
        if baseline_profit is None:
            baseline_profit = self.compute_baseline_profit()
        if raw_tax_advantage is None:
            raw_tax_advantage = self.compute_raw_tax_advantage(
                current_profit=current_profit,
                baseline_profit=baseline_profit,
            )

        if not self.has_completed_loophole_structure(
            current_profit=current_profit,
            baseline_profit=baseline_profit,
            raw_tax_advantage=raw_tax_advantage,
        ):
            return 0.0

        return raw_tax_advantage

    def has_completed_loophole_structure(
        self,
        current_profit: float | None = None,
        baseline_profit: float | None = None,
        raw_tax_advantage: float | None = None,
    ) -> bool:
        if current_profit is None:
            current_profit = self.money_to_float(self.compute_profit())
        if baseline_profit is None:
            baseline_profit = self.compute_baseline_profit()
        if raw_tax_advantage is None:
            raw_tax_advantage = self.compute_raw_tax_advantage(
                current_profit=current_profit,
                baseline_profit=baseline_profit,
            )

        return (
            raw_tax_advantage > baseline_profit * 0.03
            and len(self.state.entities) >= 3
            and self.has_completed_royalty_chain()
        )

    def has_completed_royalty_chain(self) -> bool:
        for start_id, start_entity in self.state.entities.items():
            if start_entity.company_type != "Operating":
                continue

            cur_id = start_id
            visited = set()
            chain_length = 0

            while cur_id in self.state.licenses:
                if cur_id in visited:
                    break

                visited.add(cur_id)
                cur_id = self.state.licenses[cur_id]
                chain_length += 1

                if (
                    chain_length >= 2
                    and cur_id == self.state.ip_owner
                    and self.state.entities[cur_id].company_type == "Holding"
                ):
                    return True

        return False

    def compute_baseline_profit(self):
        baseline_profit = 0.0

        tax_rates = {
            "Ireland": 0.125,
            "Bermuda": 0.0,
            "Netherlands": 0.258,
            "US": 0.21,
            "Germany": 0.15,
        }

        for entity_id, entity in self.state.entities.items():
            if entity.company_type != "Operating":
                continue

            if not self.state.has_ip_rights(entity_id):
                continue

            revenue = self.state.get_company_revenue(entity_id)
            tax_rate = tax_rates[entity.tax_residence]

            baseline_profit += revenue * (1.0 - tax_rate)

        return baseline_profit

    def compute_profit(self):
        entity_inputs: List[TaxModel.EntityTaxInput] = []
        payment_dict: Dict[str, List[TaxModel.Payment]] = {
            entity_id: [] for entity_id in self.state.entities
        }
        taxable_gross_revenue: Dict[str, float] = {}

        total_revenue = 0
        for entity_id in self.state.entities:
            revenue = self.state.get_company_revenue(entity_id)
            taxable_gross_revenue[entity_id] = revenue
            total_revenue += round(revenue)

        for entity_id, entity in self.state.entities.items():
            payer_catala_entity = self.make_entity(
                entity.incorporation_jurisdiction,
                entity.tax_residence,
            )

            revenue = self.state.get_company_revenue(entity_id)

            if entity_id in self.state.licenses:
                prev_id = entity_id
                prev_catala_entity = payer_catala_entity
                cur_payment = self.state.royalty_rate * revenue
                cur_licensor_id = self.state.licenses[entity_id]

                while True:
                    licensor_entity = self.state.entities[cur_licensor_id]
                    licensor_catala_entity = self.make_entity(
                        licensor_entity.incorporation_jurisdiction,
                        licensor_entity.tax_residence,
                    )

                    payment = self.make_payment(
                        prev_catala_entity,
                        licensor_catala_entity,
                        catala_runtime.Money(
                            catala_runtime.Integer(round(cur_payment))
                        ),
                    )

                    payment_dict[prev_id].append(payment)
                    taxable_gross_revenue[cur_licensor_id] += cur_payment

                    if cur_licensor_id not in self.state.licenses:
                        break

                    prev_id = cur_licensor_id
                    prev_catala_entity = licensor_catala_entity
                    cur_licensor_id = self.state.licenses[cur_licensor_id]
                    cur_payment *= self.state.royalty_rate

        for entity_id, entity in self.state.entities.items():
            catala_entity = self.make_entity(
                entity.incorporation_jurisdiction,
                entity.tax_residence,
            )

            revenue = taxable_gross_revenue[entity_id]

            entity_inputs.append(
                TaxModel.EntityTaxInput(
                    entity=catala_entity,
                    gross_revenue=catala_runtime.Money(
                        catala_runtime.Integer(round(revenue))
                    ),
                    outgoing_payments=payment_dict[entity_id],
                )
            )

        total_group_tax = TaxModel.group_tax_outcome(
            TaxModel.GroupTaxOutcomeIn(entity_inputs_in=entity_inputs)
        ).total_group_tax

        return (
            catala_runtime.Money(catala_runtime.Integer(total_revenue))
            - total_group_tax
        )

    def make_jurisdiction(self, jurisdiction: str):
        return TaxModel.Jurisdiction(
            getattr(TaxModel.Jurisdiction_Code, jurisdiction),
            catala_runtime.Unit(),
        )

    def make_entity(self, incorporation_jurisdiction: str, tax_residence: str):
        return TaxModel.Entity(
            incorporation_jurisdiction=self.make_jurisdiction(
                incorporation_jurisdiction
            ),
            tax_residence=self.make_jurisdiction(tax_residence),
        )

    def make_payment(self, payer, receiver, amount):
        return TaxModel.Payment(
            payer=payer,
            receiver=receiver,
            amount=amount,
        )

    def render_world(self, filename="world"):
        dot = Digraph()

        # nodes
        for entity_id, e in self.state.entities.items():
            label = (
                f"{entity_id}\n"
                f"{e.incorporation_jurisdiction}\n"
                f"TR: {e.tax_residence}\n"
                f"{e.company_type}"
            )

            if entity_id == self.state.ip_owner:
                dot.node(entity_id, label, style="filled", fillcolor="lightblue")
            else:
                dot.node(entity_id, label)

        # ownership edges
        for parent, child in self.state.subsidiary:
            dot.edge(parent, child, label="owns")

        # license edges
        for licensee, licensor in self.state.licenses.items():
            dot.edge(licensor, licensee, label="license", style="dashed")

        dot.render(filename, format="png", cleanup=True)
