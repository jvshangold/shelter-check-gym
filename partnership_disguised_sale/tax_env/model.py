from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
import torch
from torch import nn


class GNN(torch.nn.Module):
    """
    GNN encoder for the partnership disguised-sale graph.
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.individual_linear = nn.Linear(5, hidden_channels)
        self.partnership_linear = nn.Linear(3, hidden_channels)
        self.asset_linear = nn.Linear(5, hidden_channels)
        self.cash_linear = nn.Linear(6, hidden_channels)
        self.loan_linear = nn.Linear(2, hidden_channels)
        self.source_only_linear = nn.Linear(hidden_channels, out_channels)

        self.conv1 = HeteroConv({
            ("individual", "partner_of", "partnership"): SAGEConv((-1, -1), hidden_channels),
            ("individual", "owns_asset", "asset"): SAGEConv((-1, -1), hidden_channels),
            ("partnership", "owns_asset", "asset"): SAGEConv((-1, -1), hidden_channels),
            ("individual", "holds_cash", "cash"): SAGEConv((-1, -1), hidden_channels),
            ("partnership", "holds_cash", "cash"): SAGEConv((-1, -1), hidden_channels),
            ("partnership", "borrower_of", "loan"): SAGEConv((-1, -1), hidden_channels),
            ("individual", "guarantees", "loan"): SAGEConv((-1, -1), hidden_channels),
        }, aggr="sum")

        self.conv2 = HeteroConv({
            ("individual", "partner_of", "partnership"): SAGEConv((-1, -1), out_channels),
            ("individual", "owns_asset", "asset"): SAGEConv((-1, -1), out_channels),
            ("partnership", "owns_asset", "asset"): SAGEConv((-1, -1), out_channels),
            ("individual", "holds_cash", "cash"): SAGEConv((-1, -1), out_channels),
            ("partnership", "holds_cash", "cash"): SAGEConv((-1, -1), out_channels),
            ("partnership", "borrower_of", "loan"): SAGEConv((-1, -1), out_channels),
            ("individual", "guarantees", "loan"): SAGEConv((-1, -1), out_channels),
        }, aggr="sum")

    def forward(self, data):
        x_dict = {
            "individual": self.individual_linear(data["individual"].x),
            "partnership": self.partnership_linear(data["partnership"].x),
            "asset": self.asset_linear(data["asset"].x),
            "cash": self.cash_linear(data["cash"].x),
            "loan": self.loan_linear(data["loan"].x),
        }

        conv1_out = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {
            k: conv1_out.get(k, v).relu()
            for k, v in x_dict.items()
        }

        conv2_out = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {
            k: conv2_out.get(k, self.source_only_linear(v)).relu()
            for k, v in x_dict.items()
        }

        pooled = []
        for node_type in ["individual", "partnership", "asset", "cash", "loan"]:
            x = x_dict[node_type]
            if x.size(0) == 0:
                pooled.append(torch.zeros(
                    (1, x.size(1)),
                    dtype=x.dtype,
                    device=x.device,
                ))
                continue

            if hasattr(data[node_type], "batch"):
                batch = data[node_type].batch
            else:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

            pooled.append(global_mean_pool(x, batch))

        return torch.cat(pooled, dim=-1)
