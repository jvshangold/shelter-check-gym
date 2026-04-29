import gymnasium as gym
from gymnasium import Space, spaces

from typing import Dict, List

from .state import WorldState
from .render import build_graph

import sys

sys.path.append("formalizations/_target/tax_rules")
sys.path.append("formalizations/_build/libcatala/python")

from python import TaxModel
from python import catala_runtime


class TaxEnv(gym.Env):
    def __init__(self, MAX_ENTITIES=10, JURISDICTIONS=5):
        super().__init__()

        self.observation_space: Space = spaces.Dict({})
        self.action_space: Space = spaces.MultiDiscrete([
            3,
            MAX_ENTITIES,
            MAX_ENTITIES,
            JURISDICTIONS,
            JURISDICTIONS,
        ])

        self.state = None

        self.max_steps = 20
        self.max_entities = MAX_ENTITIES

        self.idx_to_entity: Dict[int, str] = {}
        self.idx_to_jurisdiction: Dict[int, str] = {
            0: "Ireland",
            1: "Netherlands",
            2: "Bermuda",
            3: "US",
            4: "Germany",
        }

        self.prev_profit = catala_runtime.Money(catala_runtime.Integer(0))
        self.steps = 0
    
    def step(self, action):
        action_type, arg_1, arg_2, arg_3, arg_4 = action

        try:
            if action_type == 0:
                if len(self.state.entities) >= self.max_entities:
                    raise ValueError("Maximum number of entities reached")

                parent_id = self.idx_to_entity[arg_1]
                incorporation_jurisdiction = self.idx_to_jurisdiction[arg_3]
                management_jurisdiction = self.idx_to_jurisdiction[arg_4]
                tax_residence = self.state.get_tax_residence(
                    incorporation_jurisdiction,
                    management_jurisdiction,
                )

                new_entity = self.state.add_child(
                    parent_id,
                    incorporation_jurisdiction,
                    management_jurisdiction,
                    tax_residence,
                )

                new_idx = len(self.idx_to_entity)
                self.idx_to_entity[new_idx] = new_entity

            elif action_type == 1:
                licensee_id = self.idx_to_entity[arg_1]
                licensor_id = self.idx_to_entity[arg_2]

                self.state.rent_ip(licensee_id, licensor_id)

                if licensor_id == licensee_id:
                    raise ValueError("A company cannot license IP from itself.")

            elif action_type == 2:
                new_owner_id = self.idx_to_entity[arg_2]

                if new_owner_id == self.state.ip_owner:
                    raise ValueError("Cannot transfer IP to current owner.")

                self.state.transfer_ip(new_owner_id)

            else:
                raise ValueError("Unknown action type")

            reward = self.compute_reward()

            if action_type == 0:
                reward -= 0.02
            elif action_type == 2:
                reward -= 0.05

            terminated = False
            truncated = False

        except Exception as e:
            print("Action:", action)
            print("Exception:", e)

            reward = -1.0
            terminated = False
            truncated = False

        obs = self.get_observation()
        info = {}

        self.steps += 1

        if self.steps >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = WorldState()
        self.steps = 0
        self.prev_profit = catala_runtime.Money(catala_runtime.Integer(0))
        self.idx_to_entity = {}

        self.state.add_root("root", "US", "US", "US")
        self.idx_to_entity[0] = "root"

        return self.get_observation(), {}
    
    def render(self):
        return
    
    def get_observation(self):
        return build_graph(state=self.state)
    
    def get_action_mask(self):
        mask = [1, 1, 1]

        if len(self.state.entities) >= self.max_entities:
            mask[0] = 0

        if not self.can_rent_ip():
            mask[1] = 0

        if not self.can_transfer_ip():
            mask[2] = 0

        return mask
    
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
            if entity_id != self.state.ip_owner:
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
        current_profit = self.compute_profit()
        reward = current_profit - self.prev_profit
        self.prev_profit = current_profit
        # divide by large number to make reward more stable
        return self.money_to_float(reward) / 1e10

    def compute_profit(self):
        entity_inputs: List[TaxModel.EntityTaxInput] = []
        payment_dict: Dict[str, List[TaxModel.Payment]] = {
            entity_id: [] for entity_id in self.state.entities
        }

        total_revenue = 0

        for entity_id, entity in self.state.entities.items():
            payer_catala_entity = self.make_entity(
                entity.incorporation_jurisdiction,
                entity.tax_residence
            )

            revenue = self.state.get_company_revenue(entity_id)
            total_revenue += round(revenue)

            if entity_id in self.state.licenses:
                prev_id = entity_id
                prev_catala_entity = payer_catala_entity
                cur_payment = 0.9 * revenue
                cur_licensor_id = self.state.licenses[entity_id]

                while True:
                    licensor_entity = self.state.entities[cur_licensor_id]
                    licensor_catala_entity = self.make_entity(
                        licensor_entity.incorporation_jurisdiction,
                        licensor_entity.tax_residence
                    )

                    payment = self.make_payment(
                        prev_catala_entity,
                        licensor_catala_entity,
                        catala_runtime.Money(catala_runtime.Integer(round(cur_payment))),
                        self.make_payment_kind("Royalty"),
                    )
                    payment_dict[prev_id].append(payment)

                    if cur_licensor_id not in self.state.licenses:
                        break

                    prev_id = cur_licensor_id
                    prev_catala_entity = licensor_catala_entity
                    cur_licensor_id = self.state.licenses[cur_licensor_id]
                    cur_payment *= 0.9

        for entity_id, entity in self.state.entities.items():
            catala_entity = self.make_entity(
                entity.incorporation_jurisdiction,
                entity.tax_residence
            )

            revenue = self.state.get_company_revenue(entity_id)

            entity_inputs.append(
                TaxModel.EntityTaxInput(
                    entity=catala_entity,
                    gross_revenue=catala_runtime.Money(catala_runtime.Integer(round(revenue))),
                    outgoing_payments=payment_dict[entity_id],
                )
            )

        total_group_tax = TaxModel.group_tax_outcome(
            TaxModel.GroupTaxOutcomeIn(entity_inputs_in=entity_inputs)
        ).total_group_tax

        return catala_runtime.Money(catala_runtime.Integer(total_revenue)) - total_group_tax
    
    def make_jurisdiction(self, jurisdiction: str):
        return TaxModel.Jurisdiction(
            getattr(TaxModel.Jurisdiction_Code, jurisdiction),
            catala_runtime.Unit()
        )

    def make_payment_kind(self, kind: str):
        return TaxModel.PaymentKind(
            getattr(TaxModel.PaymentKind_Code, kind),
            catala_runtime.Unit()
        )

    def make_entity(self, incorporation_jurisdiction: str, tax_residence: str):
        return TaxModel.Entity(
            incorporation_jurisdiction=self.make_jurisdiction(incorporation_jurisdiction),
            tax_residence=self.make_jurisdiction(tax_residence),
        )

    def make_payment(self, payer, receiver, amount, kind):
        return TaxModel.Payment(
            payer=payer,
            receiver=receiver,
            amount=amount,
            kind=kind,
        )
    
