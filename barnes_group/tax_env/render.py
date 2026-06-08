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

    corp_has_subsidiary = []
    corp_owns_cash = []
    cash_owned_by_corp = []
    corp_holds_stock = []
    stock_held_by_corp = []
    corp_issues_stock = []
    stock_issued_by_corp = []
    corp_contributed_cash = []
    cash_contributed_by_corp = []
    corp_contributed_stock = []
    stock_contributed_by_corp = []

    for corp_id in corp_ids:
        corporation = state.corporations[corp_id]
        corp_x.append([
            1.0 if corporation.tax_residence == TaxResidence.US else 0.0,
            1.0 if corp_id == state.taxpayer_id else 0.0,
            1.0 if state.is_cfc(corp_id) else 0.0,
            state.cash_amount(corp_id),
            state.ownership_percent(state.taxpayer_id, corp_id),
        ])

        if corporation.parent_id is not None:
            parent_i = corp_index[corporation.parent_id]
            corp_has_subsidiary.append([parent_i, corp_index[corp_id]])

    for cash_id in cash_ids:
        cash = state.cash[cash_id]
        cash_i = cash_index[cash_id]
        owner_i = corp_index[cash.owner_id]

        cash_x.append([cash.amount, cash.basis])
        corp_owns_cash.append([owner_i, cash_i])
        cash_owned_by_corp.append([cash_i, owner_i])

        if cash.contributed_by_id is not None:
            contributor_i = corp_index[cash.contributed_by_id]
            corp_contributed_cash.append([contributor_i, cash_i])
            cash_contributed_by_corp.append([cash_i, contributor_i])

    for stock_id in stock_ids:
        stock = state.stock[stock_id]
        stock_i = stock_index[stock_id]
        holder_i = corp_index[stock.holder_id]
        issuer_i = corp_index[stock.issuer_id]

        stock_x.append([
            stock.percent,
            stock.basis,
            1.0 if stock.issuer_id == stock.holder_id else 0.0,
            1.0 if state.corporations[stock.issuer_id].tax_residence == TaxResidence.US else 0.0,
        ])

        corp_holds_stock.append([holder_i, stock_i])
        stock_held_by_corp.append([stock_i, holder_i])
        corp_issues_stock.append([issuer_i, stock_i])
        stock_issued_by_corp.append([stock_i, issuer_i])

        if stock.contributed_by_id is not None:
            contributor_i = corp_index[stock.contributed_by_id]
            corp_contributed_stock.append([contributor_i, stock_i])
            stock_contributed_by_corp.append([stock_i, contributor_i])

    data["corporation"].x = torch.tensor(corp_x, dtype=torch.float)
    data["cash"].x = torch.tensor(cash_x, dtype=torch.float)
    data["stock"].x = torch.tensor(stock_x, dtype=torch.float)

    data["corporation", "has_subsidiary", "corporation"].edge_index = edge_index(corp_has_subsidiary)
    data["corporation", "owns_cash", "cash"].edge_index = edge_index(corp_owns_cash)
    data["cash", "owned_by", "corporation"].edge_index = edge_index(cash_owned_by_corp)
    data["corporation", "holds_stock", "stock"].edge_index = edge_index(corp_holds_stock)
    data["stock", "held_by", "corporation"].edge_index = edge_index(stock_held_by_corp)
    data["corporation", "issues_stock", "stock"].edge_index = edge_index(corp_issues_stock)
    data["stock", "issued_by", "corporation"].edge_index = edge_index(stock_issued_by_corp)
    data["corporation", "contributed_cash", "cash"].edge_index = edge_index(corp_contributed_cash)
    data["cash", "cash_contributed_by", "corporation"].edge_index = edge_index(cash_contributed_by_corp)
    data["corporation", "contributed_stock", "stock"].edge_index = edge_index(corp_contributed_stock)
    data["stock", "stock_contributed_by", "corporation"].edge_index = edge_index(stock_contributed_by_corp)

    return data


def edge_index(edges: list[list[int]]) -> torch.Tensor:
    if edges:
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.zeros((2, 0), dtype=torch.long)
