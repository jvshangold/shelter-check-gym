import torch.nn as nn
import torch.functional as F
import torch

from dub_irish_dutch.tax_env.model import GNN

class PolicyValueNet(nn.Module):
    '''
    This is our model with all of the actor heads and the one value head.
    '''
    def __init__(self, embed_dim, num_action_types, num_jurisdictions):
        super().__init__()
        self.gnn = GNN(embed_dim, embed_dim)

        # actor heads
        self.action_type_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_action_types)
        )

        self.incorporation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_jurisdictions)
        )

        self.management_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_jurisdictions)
        )

        self.src_entity_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        self.dst_entity_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        self.company_type_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2)
        )

        # value head

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, data):
        entity_embeddings, graph_embedding = self.gnn(data)

        action_logits = self.action_type_head(graph_embedding)
        incorporation_logits = self.incorporation_head(graph_embedding)
        management_logits = self.management_head(graph_embedding)
        company_type_logits = self.company_type_head(graph_embedding)
        
        g = graph_embedding.expand(entity_embeddings.size(0), -1) # shape(num_entities, embed_dim)
        node_inputs = torch.cat([entity_embeddings, g], dim=-1)

        src_logits = self.src_entity_head(node_inputs).squeeze(-1)
        dst_logits = self.dst_entity_head(node_inputs).squeeze(-1)

        value = 3.0 * torch.tanh(self.value_head(graph_embedding).squeeze(-1))
        
        return {
            "action_logits": action_logits,
            "src_logits": src_logits,
            "dst_logits": dst_logits,
            "incorporation_logits": incorporation_logits,
            "management_logits": management_logits,
            "company_type_logits": company_type_logits,
            "value": value

        }