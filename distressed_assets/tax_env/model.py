from torch_geometric.nn import SAGEConv, HeteroConv, global_mean_pool
import torch
from torch import nn


class GNN(torch.nn.Module):
    """
    GNN encoder for the distressed assets trust graph.
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.trust_linear = nn.Linear(3, hidden_channels)
        self.asset_linear = nn.Linear(5, hidden_channels)
        self.individual_linear = nn.Linear(2, hidden_channels)

        self.conv1 = HeteroConv({
            ("trust", "has_subtrust", "trust"): SAGEConv((-1, -1), hidden_channels),
            ("trust", "is_subtrust_of", "trust"): SAGEConv((-1, -1), hidden_channels),

            ("individual", "vesting_power", "trust"): SAGEConv((-1, -1), hidden_channels),
            ("trust", "controlled_by", "individual"): SAGEConv((-1, -1), hidden_channels),

            ("trust", "owns", "asset"): SAGEConv((-1, -1), hidden_channels),
            ("asset", "owned_by_trust", "trust"): SAGEConv((-1, -1), hidden_channels),

            ("individual", "owns", "asset"): SAGEConv((-1, -1), hidden_channels),
            ("asset", "owned_by_individual", "individual"): SAGEConv((-1, -1), hidden_channels),
        }, aggr="sum")

        self.conv2 = HeteroConv({
            ("trust", "has_subtrust", "trust"): SAGEConv((-1, -1), out_channels),
            ("trust", "is_subtrust_of", "trust"): SAGEConv((-1, -1), out_channels),

            ("individual", "vesting_power", "trust"): SAGEConv((-1, -1), out_channels),
            ("trust", "controlled_by", "individual"): SAGEConv((-1, -1), out_channels),

            ("trust", "owns", "asset"): SAGEConv((-1, -1), out_channels),
            ("asset", "owned_by_trust", "trust"): SAGEConv((-1, -1), out_channels),

            ("individual", "owns", "asset"): SAGEConv((-1, -1), out_channels),
            ("asset", "owned_by_individual", "individual"): SAGEConv((-1, -1), out_channels),
        }, aggr="sum")

    def forward(self, data):
        x_dict = {
            "trust": self.trust_linear(data["trust"].x),
            "asset": self.asset_linear(data["asset"].x),
            "individual": self.individual_linear(data["individual"].x),
        }

        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}

        pooled = []

        for node_type in ["trust", "asset", "individual"]:
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