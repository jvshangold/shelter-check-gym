import torch
from torch import nn

from partnership_disguised_sale.tax_env.model import GNN


class PolicyValueNet(nn.Module):
    """
    Actor-critic network for the partnership disguised-sale environment.
    """

    def __init__(
        self,
        embed_dim,
        num_action_types,
        max_individuals,
        max_assets,
        num_amounts,
    ):
        super().__init__()

        self.gnn = GNN(embed_dim, embed_dim)
        graph_dim = 5 * embed_dim

        self.action_type_head = self._head(graph_dim, embed_dim, num_action_types)
        self.individual_head = self._head(graph_dim, embed_dim, max_individuals)
        self.asset_head = self._head(graph_dim, embed_dim, max_assets)
        self.amount_head = self._head(graph_dim, embed_dim, num_amounts)
        self.value_head = self._head(graph_dim, embed_dim, 1)

    def forward(self, data):
        graph_embedding = self.gnn(data)
        value = 5.0 * torch.tanh(self.value_head(graph_embedding).squeeze(-1))

        return {
            "action_logits": self.action_type_head(graph_embedding),
            "individual_logits": self.individual_head(graph_embedding),
            "asset_logits": self.asset_head(graph_embedding),
            "amount_logits": self.amount_head(graph_embedding),
            "value": value,
        }

    def _head(self, graph_dim, embed_dim, out_dim):
        return nn.Sequential(
            nn.Linear(graph_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim),
        )
