from torch_geometric.data import HeteroData
import torch

from .state import CashSource, OwnerType, WorldState


MONEY_SCALE = 100.0


def scaled_money(amount: float) -> float:
    return amount / MONEY_SCALE


def build_graph(state: WorldState) -> HeteroData:
    data = HeteroData()

    individual_ids = list(state.individuals.keys())
    partnership_ids = list(state.partnerships.keys())
    asset_ids = list(state.assets.keys())
    cash_ids = list(state.cash_lots.keys())
    loan_ids = list(state.loans.keys())

    individual_index = {node_id: i for i, node_id in enumerate(individual_ids)}
    partnership_index = {node_id: i for i, node_id in enumerate(partnership_ids)}
    asset_index = {node_id: i for i, node_id in enumerate(asset_ids)}
    cash_index = {node_id: i for i, node_id in enumerate(cash_ids)}
    loan_index = {node_id: i for i, node_id in enumerate(loan_ids)}

    individual_x = []
    partnership_x = []
    asset_x = []
    cash_x = []
    loan_x = []

    partner_of = []
    individual_owns_asset = []
    partnership_owns_asset = []
    individual_holds_cash = []
    partnership_holds_cash = []
    borrower_of = []
    guarantees = []

    for individual_id in individual_ids:
        individual = state.individuals[individual_id]
        individual_x.append([
            1.0 if individual_id == state.taxpayer_id else 0.0,
            individual.tax_rate,
            scaled_money(state.individual_cash(individual_id)),
            scaled_money(state.cash_distributed_to(individual_id)),
        ])

    for partnership_id in partnership_ids:
        partnership = state.partnerships[partnership_id]
        partnership_x.append([
            float(len(partnership.partner_ids)) / 10.0,
            scaled_money(state.partnership_cash(partnership_id)),
            scaled_money(sum(
                loan.principal
                for loan in state.loans.values()
                if loan.borrower_partnership_id == partnership_id
            )),
        ])
        partnership_i = partnership_index[partnership_id]
        for partner_id in partnership.partner_ids:
            partner_of.append([individual_index[partner_id], partnership_i])

    for asset_id in asset_ids:
        asset = state.assets[asset_id]
        asset_i = asset_index[asset_id]
        asset_x.append([
            scaled_money(asset.basis),
            scaled_money(asset.fair_market_value),
            scaled_money(asset.built_in_gain),
            1.0 if asset.contributed_by_id == state.taxpayer_id else 0.0,
            1.0 if asset.is_contributed_to_partnership else 0.0,
        ])

        if asset.owner_type == OwnerType.INDIVIDUAL:
            individual_owns_asset.append([individual_index[asset.owner_id], asset_i])
        else:
            partnership_owns_asset.append([partnership_index[asset.owner_id], asset_i])

    for cash_id in cash_ids:
        cash = state.cash_lots[cash_id]
        cash_i = cash_index[cash_id]
        cash_x.append([
            scaled_money(cash.amount),
            1.0 if cash.source == CashSource.INITIAL else 0.0,
            1.0 if cash.source == CashSource.CONTRIBUTION else 0.0,
            1.0 if cash.source == CashSource.LOAN_PROCEEDS else 0.0,
            1.0 if cash.source == CashSource.DISTRIBUTION else 0.0,
            1.0 if cash.loan_id is not None else 0.0,
        ])

        if cash.owner_type == OwnerType.INDIVIDUAL:
            individual_holds_cash.append([individual_index[cash.owner_id], cash_i])
        else:
            partnership_holds_cash.append([partnership_index[cash.owner_id], cash_i])

    for loan_id in loan_ids:
        loan = state.loans[loan_id]
        loan_i = loan_index[loan_id]
        loan_x.append([
            scaled_money(loan.principal),
            1.0 if loan.guarantor_id == state.taxpayer_id else 0.0,
        ])
        borrower_of.append([partnership_index[loan.borrower_partnership_id], loan_i])
        guarantees.append([individual_index[loan.guarantor_id], loan_i])

    data["individual"].x = node_features(individual_x, 4)
    data["partnership"].x = node_features(partnership_x, 3)
    data["asset"].x = node_features(asset_x, 5)
    data["cash"].x = node_features(cash_x, 6)
    data["loan"].x = node_features(loan_x, 2)

    data["individual", "partner_of", "partnership"].edge_index = edge_index(partner_of)
    data["individual", "owns_asset", "asset"].edge_index = edge_index(
        individual_owns_asset
    )
    data["partnership", "owns_asset", "asset"].edge_index = edge_index(
        partnership_owns_asset
    )
    data["individual", "holds_cash", "cash"].edge_index = edge_index(
        individual_holds_cash
    )
    data["partnership", "holds_cash", "cash"].edge_index = edge_index(
        partnership_holds_cash
    )
    data["partnership", "borrower_of", "loan"].edge_index = edge_index(borrower_of)
    data["individual", "guarantees", "loan"].edge_index = edge_index(guarantees)

    return data


def edge_index(edges: list[list[int]]) -> torch.Tensor:
    if edges:
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.zeros((2, 0), dtype=torch.long)


def node_features(rows: list[list[float]], width: int) -> torch.Tensor:
    if rows:
        return torch.tensor(rows, dtype=torch.float)
    return torch.zeros((0, width), dtype=torch.float)
