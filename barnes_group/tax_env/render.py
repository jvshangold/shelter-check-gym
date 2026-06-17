from torch_geometric.data import HeteroData
import torch

from .state import TaxResidence, WorldState


def build_graph(state: WorldState):
    data = HeteroData()

    corp_ids = list(state.corporations.keys())
    cash_ids = list(state.cash.keys())
    stock_ids = list(state.stock.keys())

    corp_index = {corp_id: i for i, corp_id in enumerate(corp_ids)}
    cash_index = {cash_id: i for i, cash_id in enumerate(cash_ids)}
    stock_index = {stock_id: i for i, stock_id in enumerate(stock_ids)}

    corp_x = []
    cash_x = []
    stock_x = []

    corp_owns_cash = []
    cash_owned_by_corp = []
    corp_holds_stock = []
    stock_held_by_corp = []

    for corp_id in corp_ids:
        corporation = state.corporations[corp_id]
        parent_id = corporation.parent_id
        corp_x.append([
            1.0 if corporation.tax_residence == TaxResidence.US else 0.0,
            1.0 if corp_id == state.taxpayer_id else 0.0,
            1.0 if state.is_cfc(corp_id) else 0.0,
            state.cash_amount(corp_id),
            state.ownership_percent(state.taxpayer_id, corp_id),
            1.0 if parent_id is not None else 0.0,
            1.0 if parent_id == state.taxpayer_id else 0.0,
            1.0 if parent_id is not None and state.is_cfc(parent_id) else 0.0,
        ])

    for cash_id in cash_ids:
        cash = state.cash[cash_id]
        cash_i = cash_index[cash_id]
        owner_i = corp_index[cash.owner_id]
        contributor_id = cash.contributed_by_id

        cash_x.append([
            cash.amount,
            cash.basis,
            1.0 if contributor_id is not None else 0.0,
            1.0 if contributor_id is not None and state.corporations[contributor_id].is_domestic else 0.0,
            1.0 if contributor_id == state.taxpayer_id else 0.0,
            1.0 if contributor_id is not None and state.is_cfc(contributor_id) else 0.0,
            1.0 if cash.exchanged_for_stock_id is not None else 0.0,
        ])
        corp_owns_cash.append([owner_i, cash_i])
        cash_owned_by_corp.append([cash_i, owner_i])

    for stock_id in stock_ids:
        stock = state.stock[stock_id]
        stock_i = stock_index[stock_id]
        holder_i = corp_index[stock.holder_id]
        issuer = state.corporations[stock.issuer_id]
        issuer_parent_id = issuer.parent_id
        contributor_id = stock.contributed_by_id

        stock_x.append([
            stock.percent,
            stock.basis,
            1.0 if stock.issuer_id == stock.holder_id else 0.0,
            1.0 if issuer.tax_residence == TaxResidence.US else 0.0,
            1.0 if stock.issuer_id == state.taxpayer_id else 0.0,
            1.0 if state.is_cfc(stock.issuer_id) else 0.0,
            1.0 if issuer_parent_id == state.taxpayer_id else 0.0,
            1.0 if contributor_id is not None else 0.0,
            1.0 if contributor_id == state.taxpayer_id else 0.0,
            1.0 if contributor_id is not None and state.is_cfc(contributor_id) else 0.0,
            1.0 if stock.exchanged_for_stock_id is not None else 0.0,
        ])

        corp_holds_stock.append([holder_i, stock_i])
        stock_held_by_corp.append([stock_i, holder_i])

    data["corporation"].x = torch.tensor(corp_x, dtype=torch.float)
    data["cash"].x = torch.tensor(cash_x, dtype=torch.float)
    data["stock"].x = torch.tensor(stock_x, dtype=torch.float)

    data["corporation", "owns_cash", "cash"].edge_index = edge_index(corp_owns_cash)
    data["cash", "owned_by", "corporation"].edge_index = edge_index(cash_owned_by_corp)
    data["corporation", "holds_stock", "stock"].edge_index = edge_index(corp_holds_stock)
    data["stock", "held_by", "corporation"].edge_index = edge_index(stock_held_by_corp)

    return data


def edge_index(edges: list[list[int]]) -> torch.Tensor:
    if edges:
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.zeros((2, 0), dtype=torch.long)
