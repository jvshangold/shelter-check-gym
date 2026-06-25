from torch_geometric.data import HeteroData
import torch

from .state import WorldState


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
    market_x = [[1.0]]

    has_cash = []
    held_by = []
    issues_stock = []
    has_subcorp = []
    stock_listed_on_market = []

    for corp_id in corp_ids:
        corporation = state.corporations[corp_id]
        corp_x.append([
            1.0 if corp_id == state.taxpayer_id else 0.0,
            1.0 if corp_id == state.subsidiary_id else 0.0,
            state.cash_amount(corp_id),
            state.ownership_percent(state.taxpayer_id, corp_id),
            state.ledger.ordinary_deductions.get(corp_id, 0.0),
            state.ledger.capital_losses.get(corp_id, 0.0),
        ])

        if corporation.parent_id is not None:
            has_subcorp.append([corp_index[corporation.parent_id], corp_index[corp_id]])
        if corp_id == state.taxpayer_id:
            stock_listed_on_market.append([corp_index[corp_id], 0])

    for cash_id in cash_ids:
        cash = state.cash[cash_id]
        cash_x.append([cash.amount, cash.basis])
        has_cash.append([corp_index[cash.owner_id], cash_index[cash_id]])

    for stock_id in stock_ids:
        stock = state.stock[stock_id]
        stock_x.append([
            stock.fmv,
            stock.basis,
            stock.percent,
            1.0 if stock.issuer_id == state.taxpayer_id else 0.0,
            1.0 if stock.issuer_id == state.subsidiary_id else 0.0,
        ])
        held_by.append([stock_index[stock_id], corp_index[stock.holder_id]])
        issues_stock.append([corp_index[stock.issuer_id], stock_index[stock_id]])

    data["corporation"].x = node_features(corp_x, 6)
    data["cash"].x = node_features(cash_x, 2)
    data["stock"].x = node_features(stock_x, 5)
    data["stock_market"].x = torch.tensor(market_x, dtype=torch.float)

    data["corporation", "has_cash", "cash"].edge_index = edge_index(has_cash)
    data["stock", "held_by", "corporation"].edge_index = edge_index(held_by)
    data["corporation", "issues_stock", "stock"].edge_index = edge_index(issues_stock)
    data["corporation", "has_subcorp", "corporation"].edge_index = edge_index(has_subcorp)
    data["corporation", "stock_listed_on", "stock_market"].edge_index = edge_index(
        stock_listed_on_market
    )

    return data


def node_features(rows: list[list[float]], width: int) -> torch.Tensor:
    if rows:
        return torch.tensor(rows, dtype=torch.float)
    return torch.zeros((0, width), dtype=torch.float)


def edge_index(edges: list[list[int]]) -> torch.Tensor:
    if edges:
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.zeros((2, 0), dtype=torch.long)
