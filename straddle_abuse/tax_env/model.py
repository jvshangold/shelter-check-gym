from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
import torch
from torch import nn


class GNN(torch.nn.Module):
    """
    GNN encoder for the common trust fund straddle graph.
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.individual_linear = nn.Linear(6, hidden_channels)
        self.ctf_linear = nn.Linear(3, hidden_channels)
        self.leg_linear = nn.Linear(7, hidden_channels)

        self.conv1 = HeteroConv({
            ("individual", "invested_in", "ctf"): SAGEConv((-1, -1), hidden_channels),
            ("ctf", "has_investor", "individual"): SAGEConv((-1, -1), hidden_channels),
            ("individual", "not_invested_in", "ctf"): SAGEConv((-1, -1), hidden_channels),
            ("ctf", "missing_investor", "individual"): SAGEConv((-1, -1), hidden_channels),
            ("ctf", "owns_leg", "leg"): SAGEConv((-1, -1), hidden_channels),
            ("leg", "held_by_ctf", "ctf"): SAGEConv((-1, -1), hidden_channels),
            ("leg", "offsets", "leg"): SAGEConv((-1, -1), hidden_channels),
        }, aggr="sum")

        self.conv2 = HeteroConv({
            ("individual", "invested_in", "ctf"): SAGEConv((-1, -1), out_channels),
            ("ctf", "has_investor", "individual"): SAGEConv((-1, -1), out_channels),
            ("individual", "not_invested_in", "ctf"): SAGEConv((-1, -1), out_channels),
            ("ctf", "missing_investor", "individual"): SAGEConv((-1, -1), out_channels),
            ("ctf", "owns_leg", "leg"): SAGEConv((-1, -1), out_channels),
            ("leg", "held_by_ctf", "ctf"): SAGEConv((-1, -1), out_channels),
            ("leg", "offsets", "leg"): SAGEConv((-1, -1), out_channels),
        }, aggr="sum")

    def forward(self, data):
        x_dict = {
            "individual": self.individual_linear(data["individual"].x),
            "ctf": self.ctf_linear(data["ctf"].x),
            "leg": self.leg_linear(data["leg"].x),
        }

        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}

        pooled = []

        for node_type in ["individual", "ctf", "leg"]:
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
                batch = torch.zeros(
                    x.size(0),
                    dtype=torch.long,
                    device=x.device,
                )

            pooled.append(global_mean_pool(x, batch))

        return torch.cat(pooled, dim=-1)
