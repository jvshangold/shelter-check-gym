from torch_geometric.data import HeteroData
import torch

from state import WorldState

def build_graph(state: WorldState):
    '''
    method meant to take in WorldState and translate it to a 'HeteroData'
    graph from which we extract an embedding.
    @return heterodata to be used by GNN
    '''
    # instantiate graph that will represent state
    data = HeteroData()

    jurisdiction_vocab = {"Ireland": 0,
                          "Netherlands": 1,
                          "Bermuda": 2,
                          "US": 3,
                          "Germany": 4}
    
    entity_ids = list(state.entities.keys())

    # dictionary to retrieve index from string id
    entity_index = {eid: i for i, eid in enumerate(entity_ids)}

    entity_x = [] # node feature matrix
    has_subsidiary = [] # entity x -> has subsidiary y
    licenses_from = [] # licensee -> owner
    incorporated_in = [] # x is incorporated in J
    tax_residence = [] # x pays taxes in J

    for eid in entity_ids:
        ent = state.entities[eid]
        entity_i = entity_index[eid] # cur entity idx so we don't retrieve multiple times

        # node feature matrix
        revenue = state.get_company_revenue(eid)
        entity_x.append([revenue])

        # subsidiary graph connectivity matrix
        if ent.parent_id is not None and ent.parent_id in entity_index:
            has_subsidiary.append([entity_index[ent.parent_id], entity_i])

        # sub-licensing connectivity matrix
        owner = state.licenses.get(eid)
        if owner is not None:
            licenses_from.append([entity_i, entity_index[owner]])
        
        # incorporated connectivity
        incorporated_in.append([entity_i, jurisdiction_vocab[ent.incorporation_jurisdiction]])

        # tax_residence connectivity
        tax_residence.append([entity_i, jurisdiction_vocab[ent.tax_residence]])


    
    data["entity"].x = torch.tensor(entity_x, dtype=torch.float)

    data["jurisidction"].x = torch.arange(len(jurisdiction_vocab), dtype=torch.float).unsqueeze(1)

    # load subsidiary graph connectivity matrix
    if has_subsidiary:
        data["entity", "has_subsidiary", "entity"].edge_index = torch.tensor(has_subsidiary, dtype=torch.long).t().contiguous()
    else:

        data["entity", "has_subsidiary", "entity"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    if licenses_from:
        data["entity", "licenses_from", "entity"].edge_index = torch.tensor(licenses_from, dtype=torch.long).t().contiguous()
    else:
        data["entity", "licenses_from", "entity"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    data["entity", "incorporated_in", "jurisdiction"].edge_index = torch.tensor(incorporated_in, dtype=torch.long).t().contiguous()
    data["entity", "tax_resident_of", "jurisdiction"].edge_index = torch.tensor(tax_residence, dtype=torch.long).t().contiguous()

    return data
