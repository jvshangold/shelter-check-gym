import sys

sys.path.insert(0, "subsidiary_stock_comp/formalizations/_target/subsidiary_stock_comp_tax_rules")
sys.path.insert(0, "subsidiary_stock_comp/formalizations/_build/libcatala/python")

from python import SubsidiaryStockCompTaxModel as TaxModel
from python import catala_runtime


def make_money(amount: int):
    return catala_runtime.Money(catala_runtime.Integer(amount))


def money_value(money) -> int:
    return int(money.value.value)


def test_notice_2000_60_pattern_computes_double_tax_advantage():
    result = TaxModel.subsidiary_stock_comp_computation(
        TaxModel.SubsidiaryStockCompComputationIn(
            transaction_in=TaxModel.StockCompLiquidationInput(
                stock_compensation_fair_market_value=make_money(9900),
                shareholder_stock_basis=make_money(10000),
                liquidation_distribution_fair_market_value=make_money(100),
                shareholder_ownership_percent=catala_runtime.Decimal("0.5"),
            )
        )
    )

    assert money_value(result.ordinary_deduction) == 9900
    assert money_value(result.capital_loss) == 9900
    assert money_value(result.total_tax_advantage) == 19800


def test_80_percent_shareholder_gets_no_331_capital_loss():
    result = TaxModel.subsidiary_stock_comp_computation(
        TaxModel.SubsidiaryStockCompComputationIn(
            transaction_in=TaxModel.StockCompLiquidationInput(
                stock_compensation_fair_market_value=make_money(9900),
                shareholder_stock_basis=make_money(10000),
                liquidation_distribution_fair_market_value=make_money(100),
                shareholder_ownership_percent=catala_runtime.Decimal("0.8"),
            )
        )
    )

    assert money_value(result.ordinary_deduction) == 9900
    assert money_value(result.capital_loss) == 0
    assert money_value(result.total_tax_advantage) == 9900
