from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
import torch
from torch import nn


class GNN(torch.nn.Module):
    """
    GNN encoder for the subsidiary stock compensation environment.
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels

        self.corp_linear = nn.Linear(6, hidden_channels)
        self.cash_linear = nn.Linear(2, hidden_channels)
        self.stock_linear = nn.Linear(5, hidden_channels)
        self.market_linear = nn.Linear(1, hidden_channels)

        self.conv1 = HeteroConv({
            ("corporation", "has_cash", "cash"): SAGEConv((-1, -1), hidden_channels),
            ("corporation", "has_stock", "stock"): SAGEConv((-1, -1), hidden_channels),
            ("corporation", "has_subcorp", "corporation"): SAGEConv((-1, -1), hidden_channels),
            ("stock_market", "sells_stock", "stock"): SAGEConv((-1, -1), hidden_channels),
        }, aggr="sum")

        self.conv2 = HeteroConv({
            ("corporation", "has_cash", "cash"): SAGEConv((-1, -1), out_channels),
            ("corporation", "has_stock", "stock"): SAGEConv((-1, -1), out_channels),
            ("corporation", "has_subcorp", "corporation"): SAGEConv((-1, -1), out_channels),
            ("stock_market", "sells_stock", "stock"): SAGEConv((-1, -1), out_channels),
        }, aggr="sum")

    def forward(self, data):
        x_dict = {
            "corporation": self.corp_linear(data["corporation"].x),
            "cash": self.cash_linear(data["cash"].x),
            "stock": self.stock_linear(data["stock"].x),
            "stock_market": self.market_linear(data["stock_market"].x),
        }

        conv_out = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {
            node_type: conv_out.get(node_type, x).relu()
            for node_type, x in x_dict.items()
        }

        conv_out = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {
            node_type: conv_out.get(
                node_type,
                x.new_zeros((x.size(0), self.out_channels)),
            ).relu()
            for node_type, x in x_dict.items()
        }

        pooled = []
        for node_type in ["corporation", "cash", "stock", "stock_market"]:
            x = x_dict[node_type]
            if x.size(0) == 0:
                pooled.append(x.new_zeros((1, self.out_channels)))
                continue
            if hasattr(data[node_type], "batch"):
                batch = data[node_type].batch
            else:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            pooled.append(global_mean_pool(x, batch))

        return torch.cat(pooled, dim=-1)
