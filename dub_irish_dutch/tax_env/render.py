from torch_geometric.data import HeteroData
import torch

from .state import WorldState


REVENUE_SCALE = 1_000_000_000.0


def one_hot(index: int, width: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(width)]


def build_graph(state: WorldState):
    '''
    method meant to take in WorldState and translate it to a 'HeteroData'
    graph from which we extract an embedding.
    @return heterodata to be used by GNN
    '''
    # instantiate graph that will represent state
    data = HeteroData()

    jurisdiction_vocab = {
        "Ireland": 0,
        "Netherlands": 1,
        "Bermuda": 2,
        "US": 3,
        "Germany": 4,
    }
    
    entity_ids = list(state.entities.keys())

    # dictionary to retrieve index from string id
    entity_index = {eid: i for i, eid in enumerate(entity_ids)}

    entity_x = [] # node feature matrix
    has_subsidiary = [] # entity x -> has subsidiary y
    licenses_from = [] # licensee -> owner

    for eid in entity_ids:
        ent = state.entities[eid]
        entity_i = entity_index[eid] # cur entity idx so we don't retrieve multiple times

        revenue = state.get_company_revenue(eid) / REVENUE_SCALE

        # is_holding, is_operating one_hot
        is_holding = 1.0 if ent.company_type == "Holding" else 0.0
        is_operating = 1.0 if ent.company_type == "Operating" else 0.0
        is_ip_owner = 1.0 if eid == state.ip_owner else 0.0
        has_license = 1.0 if eid in state.licenses else 0.0

        entity_x.append([
            revenue,
            is_holding,
            is_operating,
            is_ip_owner,
            has_license,
            *one_hot(jurisdiction_vocab[ent.incorporation_jurisdiction], 5),
            *one_hot(jurisdiction_vocab[ent.management_jurisdiction], 5),
            *one_hot(jurisdiction_vocab[ent.tax_residence], 5),
        ])

        # subsidiary graph connectivity matrix
        if ent.parent_id is not None and ent.parent_id in entity_index:
            has_subsidiary.append([entity_index[ent.parent_id], entity_i])

        # sub-licensing connectivity matrix
        owner = state.licenses.get(eid)
        if owner is not None:
            licenses_from.append([entity_i, entity_index[owner]])
        
    data["entity"].x = torch.tensor(entity_x, dtype=torch.float)

    # load subsidiary graph connectivity matrix
    if has_subsidiary:
        data["entity", "has_subsidiary", "entity"].edge_index = torch.tensor(has_subsidiary, dtype=torch.long).t().contiguous()
    else:
        data["entity", "has_subsidiary", "entity"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    if licenses_from:
        data["entity", "licenses_from", "entity"].edge_index = torch.tensor(licenses_from, dtype=torch.long).t().contiguous()
    else:
        data["entity", "licenses_from", "entity"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    return data
