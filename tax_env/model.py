import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

class GNN(torch.nn.Module):
    '''
    GNN-encoder that will encode state to get an embedding vector
    to be passed into the RL pipeline
    '''
    def __init__(self, hidden_channels, out_channels):    
        super().__init__()
        
        self.entity_linear = nn.Linear(1, hidden_channels)
        self.jurisdiction_linear = nn.Linear(1, hidden_channels)
        
        self.conv1 = HeteroConv({
            ("entity", "has_subsidiary", "entity"): SAGEConv((-1, -1), hidden_channels),
            ("entity", "licenses_from", "entity"): SAGEConv((-1, -1), hidden_channels),
            ("entity", "incorporated_in", "entity"): SAGEConv((-1, -1), hidden_channels),
            ("entity", "tax_resident_of", "entity"): SAGEConv((-1, -1), hidden_channels),
            ("entity", "managed_from", "entity"): SAGEConv((-1, -1), hidden_channels)
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ("entity", "has_subsidiary", "entity"): SAGEConv((-1, -1), out_channels),
            ("entity", "licenses_from", "entity"): SAGEConv((-1, -1), out_channels),
            ("entity", "incorporated_in", "entity"): SAGEConv((-1, -1), out_channels),
            ("entity", "tax_resident_of", "entity"): SAGEConv((-1, -1), out_channels),
            ("entity", "managed_from", "entity"): SAGEConv((-1, -1), out_channels)
        }, aggr='sum')
        
    def forward(self, data):
        x_dict = {
            "entity": self.entity_linear(data["entity"].x),
            "jurisdiction": self.jurisdiction_linear(data["jurisdiction"].x)}
    
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}

        # add moving to device later for GPU opt.
        batch = torch.zeros(x_dict["entity"].size(0), dtype=torch.long)
        embedding = global_mean_pool(x_dict["entity"], batch)
        return embedding