from torch_geometric.data import HeteroData
import torch

from .state import WorldState, OwnerType, AssetKind, TaxResidence


def build_graph(state: WorldState):
    data = HeteroData()

    trust_ids = list(state.trusts.keys())
    asset_ids = list(state.assets.keys())
    individual_ids = list(state.individuals.keys())

    trust_index = {tid: i for i, tid in enumerate(trust_ids)}
    asset_index = {aid: i for i, aid in enumerate(asset_ids)}
    individual_index = {iid: i for i, iid in enumerate(individual_ids)}

    trust_x = []
    asset_x = []
    individual_x = []

    has_subtrust = []
    is_subtrust_of = []

    vesting_power = []
    controlled_by = []

    trust_owns = []
    asset_owned_by_trust = []

    individual_owns = []
    asset_owned_by_individual = []

    for tid in trust_ids:
        trust = state.trusts[tid]
        trust_i = trust_index[tid]

        trust_x.append([
            1.0 if trust.parent_trust_id is None else 0.0,
            1.0 if trust.section_678_power_holder_id else 0.0,
            float(len(state.assets_owned_by_trust(tid))),
        ])

        if trust.parent_trust_id is not None:
            parent_i = trust_index[trust.parent_trust_id]
            has_subtrust.append([parent_i, trust_i])
            is_subtrust_of.append([trust_i, parent_i])

        if trust.section_678_power_holder_id is not None:
            holder_i = individual_index[trust.section_678_power_holder_id]
            vesting_power.append([holder_i, trust_i])
            controlled_by.append([trust_i, holder_i])

    for aid in asset_ids:
        asset = state.assets[aid]
        asset_i = asset_index[aid]

        asset_kind = 1.0 if asset.kind == AssetKind.PROPERTY else 0.0
        sold = 1.0 if asset.sale_price is not None else 0.0
        sale_price = asset.sale_price if asset.sale_price is not None else 0.0

        asset_x.append([
            asset.basis,
            asset.fair_market_value,
            sale_price,
            sold,
            asset_kind,
        ])

        if asset.owner_type == OwnerType.TRUST:
            trust_i = trust_index[asset.owner_id]
            trust_owns.append([trust_i, asset_i])
            asset_owned_by_trust.append([asset_i, trust_i])

        elif asset.owner_type == OwnerType.INDIVIDUAL:
            individual_i = individual_index[asset.owner_id]
            individual_owns.append([individual_i, asset_i])
            asset_owned_by_individual.append([asset_i, individual_i])

    for iid in individual_ids:
        individual = state.individuals[iid]

        individual_x.append([
            1.0 if individual.tax_residence == TaxResidence.US else 0.0,
            1.0 if iid == state.taxpayer_id else 0.0,
        ])

    data["trust"].x = torch.tensor(trust_x, dtype=torch.float)
    data["asset"].x = torch.tensor(asset_x, dtype=torch.float)
    data["individual"].x = torch.tensor(individual_x, dtype=torch.float)

    data["trust", "has_subtrust", "trust"].edge_index = edge_index(has_subtrust)
    data["trust", "is_subtrust_of", "trust"].edge_index = edge_index(is_subtrust_of)

    data["individual", "vesting_power", "trust"].edge_index = edge_index(vesting_power)
    data["trust", "controlled_by", "individual"].edge_index = edge_index(controlled_by)

    data["trust", "owns", "asset"].edge_index = edge_index(trust_owns)
    data["asset", "owned_by_trust", "trust"].edge_index = edge_index(asset_owned_by_trust)

    data["individual", "owns", "asset"].edge_index = edge_index(individual_owns)
    data["asset", "owned_by_individual", "individual"].edge_index = edge_index(
        asset_owned_by_individual
    )

    return data


def edge_index(edges: list[list[int]]) -> torch.Tensor:
    if edges:
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.zeros((2, 0), dtype=torch.long)