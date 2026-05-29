import torch
from torch import nn

from distressed_assets.tax_env.model import GNN


class PolicyValueNet(nn.Module):
    """
    Actor-critic network for the distressed-assets environment.
    """

    def __init__(
        self,
        embed_dim,
        num_action_types,
        max_trusts,
        max_assets,
        max_individuals,
    ):
        super().__init__()

        self.gnn = GNN(embed_dim, embed_dim)
        graph_dim = 3 * embed_dim

        self.action_type_head = nn.Sequential(
            nn.Linear(graph_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_action_types),
        )

        self.trust_head = nn.Sequential(
            nn.Linear(graph_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, max_trusts),
        )

        self.asset_head = nn.Sequential(
            nn.Linear(graph_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, max_assets),
        )

        self.individual_head = nn.Sequential(
            nn.Linear(graph_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, max_individuals),
        )

        self.value_head = nn.Sequential(
            nn.Linear(graph_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, data):
        graph_embedding = self.gnn(data)

        value = 3.0 * torch.tanh(self.value_head(graph_embedding).squeeze(-1))

        return {
            "action_logits": self.action_type_head(graph_embedding),
            "trust_logits": self.trust_head(graph_embedding),
            "asset_logits": self.asset_head(graph_embedding),
            "individual_logits": self.individual_head(graph_embedding),
            "value": value,
        }
