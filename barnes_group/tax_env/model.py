from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
import torch
from torch import nn


class GNN(torch.nn.Module):
    """
    GNN encoder for the Barnes Group repatriation environment.
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.corporation_linear = nn.Linear(5, hidden_channels)
        self.cash_linear = nn.Linear(2, hidden_channels)
        self.stock_linear = nn.Linear(4, hidden_channels)

        edge_types = {
            ("corporation", "has_subsidiary", "corporation"): SAGEConv((-1, -1), hidden_channels),
            ("corporation", "owns_cash", "cash"): SAGEConv((-1, -1), hidden_channels),
            ("cash", "owned_by", "corporation"): SAGEConv((-1, -1), hidden_channels),
            ("corporation", "holds_stock", "stock"): SAGEConv((-1, -1), hidden_channels),
            ("stock", "held_by", "corporation"): SAGEConv((-1, -1), hidden_channels),
            ("corporation", "issues_stock", "stock"): SAGEConv((-1, -1), hidden_channels),
            ("stock", "issued_by", "corporation"): SAGEConv((-1, -1), hidden_channels),
            ("corporation", "contributed_cash", "cash"): SAGEConv((-1, -1), hidden_channels),
            ("cash", "cash_contributed_by", "corporation"): SAGEConv((-1, -1), hidden_channels),
            ("corporation", "contributed_stock", "stock"): SAGEConv((-1, -1), hidden_channels),
            ("stock", "stock_contributed_by", "corporation"): SAGEConv((-1, -1), hidden_channels),
        }

        self.conv1 = HeteroConv(edge_types, aggr="sum")
        self.conv2 = HeteroConv(
            {edge_type: SAGEConv((-1, -1), out_channels) for edge_type in edge_types},
            aggr="sum",
        )

    def forward(self, data):
        x_dict = {
            "corporation": self.corporation_linear(data["corporation"].x),
            "cash": self.cash_linear(data["cash"].x),
            "stock": self.stock_linear(data["stock"].x),
        }

        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: value.relu() for key, value in x_dict.items()}

        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {key: value.relu() for key, value in x_dict.items()}

        pooled = []
        for node_type in ["corporation", "cash", "stock"]:
            x = x_dict[node_type]
            if hasattr(data[node_type], "batch"):
                batch = data[node_type].batch
            else:
                batch = torch.zeros(
                    x.size(0),
                    dtype=torch.long,
                    device=x.device,
                )
            pooled.append(global_mean_pool(x, batch))

        return torch.cat(pooled, dim=-1)
