import gymnasium as gym
from gymnasium import Space, spaces

from typing import Dict, List
from dataclasses import field

from .state import WorldState
from .render import build_graph

import sys

sys.path.append("formalizations/_target/tax_rules")
sys.path.append("formalizations/_build/libcatala/python")

from python import TaxModel
from python import catala_runtime


class TaxEnv(gym.Env):
    def __init__(self, hidden_dim, MAX_ENTITES=10, JURISDICTIONS=5):
        self.observation_space: Space = spaces.Discrete(hidden_dim,)
        self.action_space: Space = spaces.MultiDiscrete([3, # add_child, rent_ip, transfer_ip
                                                         MAX_ENTITES, # arg1
                                                         MAX_ENTITES, # arg2
                                                         JURISDICTIONS, # arg 3
                                                         JURISDICTIONS]) # arg 4
        self.state = None

        self.max_steps = 20
        self.max_entities = MAX_ENTITES

        self.idx_to_entity: Dict[int, str] = field(default_factory=dict)
        self.idx_to_jurisdiction: Dict[int, str] = {0: "Ireland", 
                                                    1: "Netherlands", 
                                                    2: "Bermuda", 
                                                    3: "US", 
                                                    4: "Germany"}
        self.prev_profit = catala_runtime.Money(0)
    
    def step(self, action):
        action_type, arg_1, arg_2, arg_3, arg_4 = action

        try:
            entity_1 = self.idx_to_entity[arg_1]
            entity_2 = self.idx_to_entity[arg_2]
            incorporation_jurisdiction = self.idx_to_jurisdiction[arg_3]
            management_jurisdiction = self.idx_to_jurisdiction[arg_4]
            tax_residence = self.state.get_tax_residence(incorporation_jurisdiction, management_jurisdiction)

            if action_type == 0:  
                if len(self.state.entities >= self.max_entities):
                    raise ValueError("Maximum number of entities reached")
                new_entity = self.state.add_child(
                    entity_1,
                    incorporation_jurisdiction,
                    management_jurisdiction,
                    tax_residence
                )
                self.idx_to_entity[new_entity] = len(self.state.entities)

            elif action_type == 1: 
                self.state.rent_ip(
                    entity_1,
                    entity_2
                )

            elif action_type == 2:
                self.state.transfer_ip(entity_1)
            
            else:
                raise ValueError("Unknown action type")
            
            reward = self.compute_reward()
            terminated = False
            truncated = False
        
        except:
            reward = -100.0
            terminated = False
            truncated = False
        
        obs = self.get_observation()
        info = {}
        
        self.steps += 1
        
        # check for truncation
        if self.steps >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = WorldState()
        self.steps = 0
        
        # add root
        self.state.add_root("root", "US", "US", "US")

        return self.get_observation(), {}
    
    def render(self):
        return
    
    def get_observation(self):
        return build_graph(state=self.state)
    
    def get_action_mask(self):
        '''
        Mask that tells us if we can add more children
        '''
        mask = [1, 1, 1]

        if len(self.state.entities) >= self.max_entities:
            mask[0] = 0
        
        return mask
    
    def get_license_mask(self):
        '''
        Mask that tells us what entities can license ip
        '''
        mask = []
        for i in range(self.max_entities):
            if i not in self.idx_to_entity:
                mask.append(0)
            else:
                entity_id = self.idx_to_entity[i]
                mask.append(1 if self.state.has_ip_rights(entity_id) else 0) 

        return mask     
    
    def get_entity_mask(self):
        '''
        Mask that tells us what entities can have subsidiaries
        '''
        
        mask = []
        for i in range(self.max_entities):
            if i not in self.idx_to_entity:
                mask.append(0)
            else:
                mask.append(1) 

        return mask
    
    def compute_reward(self):
        current_profit = self.compute_profit()
        reward = current_profit - self.prev_profit
        self.prev_profit = current_profit
        return reward

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
    
