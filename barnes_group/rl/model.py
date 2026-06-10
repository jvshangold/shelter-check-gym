import torch
from torch import nn

from barnes_group.tax_env.model import GNN


class PolicyValueNet(nn.Module):
    """
    Actor-critic network for the Barnes Group environment.
    """

    def __init__(
        self,
        embed_dim,
        num_action_types,
        max_corporations,
        max_stocks,
        max_amounts,
    ):
        super().__init__()

        self.gnn = GNN(embed_dim, embed_dim)
        graph_dim = 3 * embed_dim

        self.action_type_head = self._head(graph_dim, embed_dim, num_action_types)
        self.corp_a_head = self._head(graph_dim, embed_dim, max_corporations)
        self.corp_b_head = self._head(graph_dim, embed_dim, max_corporations)
        self.stock_head = self._head(graph_dim, embed_dim, max_stocks)
        self.amount_head = self._head(graph_dim, embed_dim, max_amounts)
        self.value_head = self._head(graph_dim, embed_dim, 1)

    def forward(self, data):
        graph_embedding = self.gnn(data)
        value = 5.0 * torch.tanh(self.value_head(graph_embedding).squeeze(-1))

        return {
            "action_logits": self.action_type_head(graph_embedding),
            "corp_a_logits": self.corp_a_head(graph_embedding),
            "corp_b_logits": self.corp_b_head(graph_embedding),
            "stock_logits": self.stock_head(graph_embedding),
            "amount_logits": self.amount_head(graph_embedding),
            "value": value,
        }

    def _head(self, graph_dim, embed_dim, out_dim):
        return nn.Sequential(
            nn.Linear(graph_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim),
        )
