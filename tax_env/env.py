import gymnasium as gym
from gymnasium import Space, spaces, ActType

from typing import Dict
from dataclasses import field

from .state import WorldState, Entity
from .render import build_graph


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
    
    def step(self, action: ActType):
        action_type, arg_1, arg_2, arg_3, arg_4 = action

        try:
            entity_1 = self.idx_to_entity[arg_1]
            entity_2 = self.idx_to_entity[arg_2]
            incorporation_jurisdiction = self.idx_to_jurisdiction[arg_3]
            management_jurisdiction = self.idx_to_jurisdiction[arg_4]
            tax_residence = self.state.get_tax_residence(incorporation_jurisdiction, management_jurisdiction)

            if action_type == 0:  
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
    
    def reset(self, seed):
        super().reset(seed=seed)
        self.state = WorldState()
        self.steps = 0
        
        # add root
        self.state.entites["root"] = Entity("root", "US", "US", "US")
        self.state.ip_owner = "root"

        return self.get_observation(), {}
    
    def render():
        return
    
    def get_observation(self):
        return build_graph(state=self.state)
    
    def get_action_mask(self):
        return
    
    