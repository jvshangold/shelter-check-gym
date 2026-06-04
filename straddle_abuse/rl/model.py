import torch
from torch import nn

from straddle_abuse.tax_env.model import GNN


class PolicyValueNet(nn.Module):
    """
    Actor-critic network for the straddle-abuse environment.
    """

    def __init__(
        self,
        embed_dim,
        num_action_types,
        max_straddles,
        num_fractions,
        num_individuals,
    ):
        super().__init__()

        self.gnn = GNN(embed_dim, embed_dim)
        graph_dim = 3 * embed_dim

        self.action_type_head = self._head(graph_dim, embed_dim, num_action_types)
        self.straddle_head = self._head(graph_dim, embed_dim, max_straddles)
        self.fraction_head = self._head(graph_dim, embed_dim, num_fractions)
        self.individual_head = self._head(graph_dim, embed_dim, num_individuals)
        self.value_head = self._head(graph_dim, embed_dim, 1)

    def forward(self, data):
        graph_embedding = self.gnn(data)
        value = 5.0 * torch.tanh(self.value_head(graph_embedding).squeeze(-1))

        return {
            "action_logits": self.action_type_head(graph_embedding),
            "straddle_logits": self.straddle_head(graph_embedding),
            "fraction_logits": self.fraction_head(graph_embedding),
            "individual_logits": self.individual_head(graph_embedding),
            "value": value,
        }

    def _head(self, graph_dim, embed_dim, out_dim):
        return nn.Sequential(
            nn.Linear(graph_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim),
        )
