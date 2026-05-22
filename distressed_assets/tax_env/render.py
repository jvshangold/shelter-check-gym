from torch_geometric.data import HeteroData
import torch

from .state import WorldState, OwnerType, AssetKind


def build_graph(state: WorldState):
    data = HeteroData()

    trust_ids = list(state.trusts.keys())
    asset_ids = list(state.assets.keys())
    individual_ids = list(state.individuals.keys())

    trust_index = {tid: i for i, tid in enumerate(trust_ids)}
    asset_index = {aid: i for i, aid in enumerate(asset_ids)}
    individual_index = {iid: i for i, iid in enumerate(individual_ids)}

    trust_x = [[0.0] for _ in trust_ids]
    asset_x = []
    individual_x = []

    has_subtrust = []
    vesting_power = []
    trust_owns = []
    individual_owns = []

    for tid in trust_ids:
        trust = state.trusts[tid]
        trust_i = trust_index[tid]

        if trust.parent_trust_id is not None:
            has_subtrust.append([trust_index[trust.parent_trust_id], trust_i])

        if trust.section_678_power_holder_id is not None:
            vesting_power.append([
                individual_index[trust.section_678_power_holder_id],
                trust_i,
            ])

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
            trust_owns.append([trust_index[asset.owner_id], asset_i])
        else:
            individual_owns.append([individual_index[asset.owner_id], asset_i])

    for iid in individual_ids:
        individual = state.individuals[iid]
        individual_x.append([individual.income])

    data["trust"].x = torch.tensor(trust_x, dtype=torch.float)
    data["individual"].x = torch.tensor(individual_x, dtype=torch.float)
    data["asset"].x = torch.tensor(asset_x, dtype=torch.float)

    data["trust", "has_subtrust", "trust"].edge_index = edge_index(has_subtrust)
    data["individual", "vesting_power", "trust"].edge_index = edge_index(vesting_power)
    data["trust", "owns", "asset"].edge_index = edge_index(trust_owns)
    data["individual", "owns", "asset"].edge_index = edge_index(individual_owns)

    return data


def edge_index(edges: list[list[int]]) -> torch.Tensor:
    if edges:
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.zeros((2, 0), dtype=torch.long)