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
        self.prev_profit = 0
    
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
        
        steps += 1
        
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
    
    # below is all of the catala code and helper functions to compute the reward for a new state
    
    def compute_reward(self):
        profit =  self.compute_profit() - self.prev_profit
        self.prev_profit = profit
        return profit

    def compute_profit(self):

        # list of entity inputs to be used in TaxModel.EntityTaxOutcome
        entity_inputs: List[TaxModel.EntityTaxInput] = []

        # dictionary to map entity to the payments it must make
        payment_dict: Dict[str, List[TaxModel.Payment]] = dict()

        profit = 0

        for entity_id, entity in self.state.entites.items():
            incorporation_jurisdiction = self.make_jurisdiction(entity.incorporation_jurisdiction)
            tax_residence = self.make_jurisdiction(entity.tax_residence)
            
            catala_entity = self.make_entity(entity.incorporation_jurisdiction, entity.tax_residence)
            
            # get revenue
            revenue = self.state.get_company_revenue(entity_id)

            # keep accumulating profit for later
            profit += revenue
            
            # logic to make outgoing payments list with licensing graph
            if entity_id in self.state.licenses:
                prev = catala_entity

                cur_licensor = self.state.licenses[entity_id]
                cur_payment = .9 * revenue

                while cur_licensor in self.state.licenses:
                    # get new jurisdictions for catala
                    incorporation_jurisdiction = self.make_jurisdiction(entity.incorporation_jurisdiction)
                    tax_residence = self.make_jurisdiction(entity.tax_residence)
                    cur_licensor_catala = self.make_entity(incorporation_jurisdiction, tax_residence)

                    # add the payment to outgoing payments for correct entity_id
                    amount = catala_runtime.Money(round(cur_payment))
                    payment = self.make_payment(prev, cur_licensor_catala, amount, self.make_payment_kind("Royalty"))
                    payment_dict[cur_licensor].append(payment)

                    prev = cur_licensor_catala
                    cur_payment *= .9      
            
        for entity_id, entity in self.state.entities.items():    
            outgoing_payments = payment_dict[entity_id]
            incorporation_jurisdiction = self.make_jurisdiction(entity.incorporation_jurisdiction)
            tax_residence = self.make_jurisdiction(entity.tax_residence)
            catala_entity = self.make_entity(incorporation_jurisdiction, tax_residence)
            revenue = self.state.get_company_revenue(entity_id)


            entity_tax_input = TaxModel.EntityTaxInput(catala_entity, revenue, outgoing_payments)
            
            entity_inputs.append(entity_tax_input)

        # call TaxModel.GroupTaxOutcome
        total_taxes = TaxModel.GroupTaxOutcome(entity_inputs)

        return profit - total_taxes
    
    def make_jurisdiction(self, name: str):
        return TaxModel.Jurisdiction(TaxModel.Jurisdiction_Code[name], None)
    
    def make_entity(self, incorporation_jurisdiction, tax_residence):
        return TaxModel.Entity(incorporation_jurisdiction, tax_residence)
    
    def make_payment_kind(self, name: str):
        return TaxModel.PaymentKind(TaxModel.PaymentKind_Code[name], None)
    
    def make_payment(self, payer, receiver, amount, kind):
        return TaxModel.Payment(payer, receiver, amount, kind)
    
