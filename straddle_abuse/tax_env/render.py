from torch_geometric.data import HeteroData
import torch

from .state import StraddleLegKind, WorldState


def build_graph(state: WorldState) -> HeteroData:
    data = HeteroData()

    ctf_ids = [state.ctf.id]
    individual_ids = list(state.individuals.keys())
    leg_ids = list(state.straddle_legs.keys())

    ctf_index = {ctf_id: i for i, ctf_id in enumerate(ctf_ids)}
    individual_index = {
        individual_id: i for i, individual_id in enumerate(individual_ids)
    }
    leg_index = {leg_id: i for i, leg_id in enumerate(leg_ids)}

    ctf_x = [[
        float(state.ctf.participant_count),
        state.ctf.total_contributions,
        1.0 if state.ctf.is_common_trust_fund else 0.0,
    ]]

    individual_x = []
    leg_x = []

    invested_in = []
    has_investor = []
    not_invested_in = []
    missing_investor = []
    owns_leg = []
    held_by_ctf = []
    offsets = []

    ctf_i = ctf_index[state.ctf.id]

    for individual_id in individual_ids:
        individual = state.individuals[individual_id]
        individual_i = individual_index[individual_id]
        contribution = state.ctf.contribution_of(individual_id)

        individual_x.append([
            individual.cash,
            contribution,
            individual.tax_rate,
            state.allocated_gain_for(individual_id),
            state.allocated_loss_for(individual_id),
            1.0 if individual_id == state.taxpayer_id else 0.0,
        ])

        if contribution > 0.0:
            invested_in.append([individual_i, ctf_i])
            has_investor.append([ctf_i, individual_i])
        else:
            not_invested_in.append([individual_i, ctf_i])
            missing_investor.append([ctf_i, individual_i])

    for leg_id in leg_ids:
        leg = state.straddle_legs[leg_id]
        leg_i = leg_index[leg_id]

        leg_x.append([
            1.0 if leg.kind == StraddleLegKind.GAIN else 0.0,
            1.0 if leg.kind == StraddleLegKind.LOSS else 0.0,
            float(leg.straddle_id),
            leg.built_in_amount,
            leg.realized_amount,
            leg.remaining_amount,
            1.0 if leg.is_fully_realized else 0.0,
        ])

        owns_leg.append([ctf_i, leg_i])
        held_by_ctf.append([leg_i, ctf_i])

    for gain_leg_id, gain_leg in state.straddle_legs.items():
        if gain_leg.kind != StraddleLegKind.GAIN:
            continue

        loss_leg_id = f"straddle_{gain_leg.straddle_id}_loss"
        if loss_leg_id not in leg_index:
            continue

        gain_i = leg_index[gain_leg_id]
        loss_i = leg_index[loss_leg_id]
        offsets.append([gain_i, loss_i])
        offsets.append([loss_i, gain_i])

    data["ctf"].x = node_features(ctf_x, 3)
    data["individual"].x = node_features(individual_x, 6)
    data["leg"].x = node_features(leg_x, 7)

    data["individual", "invested_in", "ctf"].edge_index = edge_index(invested_in)
    data["ctf", "has_investor", "individual"].edge_index = edge_index(has_investor)

    data["individual", "not_invested_in", "ctf"].edge_index = edge_index(
        not_invested_in
    )
    data["ctf", "missing_investor", "individual"].edge_index = edge_index(
        missing_investor
    )

    data["ctf", "owns_leg", "leg"].edge_index = edge_index(owns_leg)
    data["leg", "held_by_ctf", "ctf"].edge_index = edge_index(held_by_ctf)

    data["leg", "offsets", "leg"].edge_index = edge_index(offsets)

    return data


def edge_index(edges: list[list[int]]) -> torch.Tensor:
    if edges:
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.zeros((2, 0), dtype=torch.long)


def node_features(rows: list[list[float]], width: int) -> torch.Tensor:
    if rows:
        return torch.tensor(rows, dtype=torch.float)
    return torch.zeros((0, width), dtype=torch.float)
