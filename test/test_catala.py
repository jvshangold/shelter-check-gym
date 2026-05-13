import sys

sys.path.append("formalizations/_target/tax_rules")
sys.path.append("formalizations/_build/libcatala/python")

from python import TaxModel
from python import catala_runtime


def make_jurisdiction(name: str):
    return TaxModel.Jurisdiction(
        getattr(TaxModel.Jurisdiction_Code, name),
        catala_runtime.Unit(),
    )


def make_entity(incorporation: str, tax_residence: str):
    return TaxModel.Entity(
        incorporation_jurisdiction=make_jurisdiction(incorporation),
        tax_residence=make_jurisdiction(tax_residence),
    )


def make_money(amount: int):
    return catala_runtime.Money(catala_runtime.Integer(amount))


def make_payment(payer, receiver, amount: int):
    return TaxModel.Payment(
        payer=payer,
        receiver=receiver,
        amount=make_money(amount),
    )


def test_royalty_deductibility_ireland_to_netherlands():
    payer = make_entity("Ireland", "Ireland")
    receiver = make_entity("Netherlands", "Netherlands")
    payment = make_payment(payer, receiver, 9000)

    result = TaxModel.royalty_deductibility(
        TaxModel.RoyaltyDeductibilityIn(payment_in=payment)
    )

    assert result.deductible_amount == make_money(9000)


def test_royalty_deductibility_ireland_to_bermuda_not_deductible():
    payer = make_entity("Ireland", "Ireland")
    receiver = make_entity("Bermuda", "Bermuda")
    payment = make_payment(payer, receiver, 9000)

    result = TaxModel.royalty_deductibility(
        TaxModel.RoyaltyDeductibilityIn(payment_in=payment)
    )

    assert result.deductible_amount == make_money(0)


def test_withholding_ireland_to_netherlands_royalty_zero():
    payer = make_entity("Ireland", "Ireland")
    receiver = make_entity("Netherlands", "Netherlands")
    payment = make_payment(payer, receiver, 9000)

    result = TaxModel.withholding_tax_computation(
        TaxModel.WithholdingTaxComputationIn(payment_in=payment)
    )

    assert result.withholding_tax_due == make_money(0)


def test_withholding_ireland_to_bermuda_royalty_20_percent():
    payer = make_entity("Ireland", "Ireland")
    receiver = make_entity("Bermuda", "Bermuda")
    payment = make_payment(payer, receiver, 9000)

    result = TaxModel.withholding_tax_computation(
        TaxModel.WithholdingTaxComputationIn(payment_in=payment)
    )

    assert result.withholding_tax_due == make_money(1800)


def test_withholding_us_to_bermuda_royalty_10_percent():
    payer = make_entity("US", "US")
    receiver = make_entity("Bermuda", "Bermuda")
    payment = make_payment(payer, receiver, 9000)

    result = TaxModel.withholding_tax_computation(
        TaxModel.WithholdingTaxComputationIn(payment_in=payment)
    )

    assert result.withholding_tax_due == make_money(900)


def test_withholding_us_to_netherlands_royalty_zero():
    payer = make_entity("US", "US")
    receiver = make_entity("Netherlands", "Netherlands")
    payment = make_payment(payer, receiver, 9000)

    result = TaxModel.withholding_tax_computation(
        TaxModel.WithholdingTaxComputationIn(payment_in=payment)
    )

    assert result.withholding_tax_due == make_money(0)


def test_corporate_tax_ireland_100_less_90_royalty():
    ireland = make_entity("Ireland", "Ireland")
    dutch = make_entity("Netherlands", "Netherlands")
    payment = make_payment(ireland, dutch, 9000)

    result = TaxModel.corporate_tax_computation(
        TaxModel.CorporateTaxComputationIn(
            entity_in=ireland,
            gross_revenue_in=make_money(10000),
            outgoing_payments_in=[payment],
        )
    )

    assert result.tax_due == make_money(125)


def test_corporate_tax_ireland_to_bermuda_no_deduction():
    ireland = make_entity("Ireland", "Ireland")
    bermuda = make_entity("Bermuda", "Bermuda")
    payment = make_payment(ireland, bermuda, 9000)

    result = TaxModel.corporate_tax_computation(
        TaxModel.CorporateTaxComputationIn(
            entity_in=ireland,
            gross_revenue_in=make_money(10000),
            outgoing_payments_in=[payment],
        )
    )

    # Ireland -> Bermuda royalty is not deductible under RoyaltyDeductibility.
    # So tax base remains 10000.
    # 10000 * 12.5% = 1250.
    assert result.tax_due == make_money(1250)


def test_entity_tax_outcome_ireland_to_netherlands():
    ireland = make_entity("Ireland", "Ireland")
    dutch = make_entity("Netherlands", "Netherlands")
    payment = make_payment(ireland, dutch, 9000)

    result = TaxModel.entity_tax_outcome(
        TaxModel.EntityTaxOutcomeIn(
            entity_in=ireland,
            gross_revenue_in=make_money(10000),
            outgoing_payments_in=[payment],
        )
    )

    # Corporate tax:
    # gross revenue 10000 - deductible royalty 9000 = 1000
    # 1000 * 12.5% = 125
    #
    # Withholding:
    # Ireland -> Netherlands = 0
    assert result.total_tax == make_money(125)


def test_entity_tax_outcome_ireland_to_bermuda():
    ireland = make_entity("Ireland", "Ireland")
    bermuda = make_entity("Bermuda", "Bermuda")
    payment = make_payment(ireland, bermuda, 9000)

    result = TaxModel.entity_tax_outcome(
        TaxModel.EntityTaxOutcomeIn(
            entity_in=ireland,
            gross_revenue_in=make_money(10000),
            outgoing_payments_in=[payment],
        )
    )

    # Corporate tax:
    # Ireland -> Bermuda royalty is not deductible.
    # tax base = 10000
    # 10000 * 12.5% = 1250
    #
    # Withholding:
    # Ireland -> Bermuda = 20%
    # 9000 * 20% = 1800
    #
    # total = 1250 + 1800 = 3050
    assert result.total_tax == make_money(3050)


def test_group_tax_outcome_single_entity():
    ireland = make_entity("Ireland", "Ireland")
    dutch = make_entity("Netherlands", "Netherlands")
    payment = make_payment(ireland, dutch, 9000)

    entity_input = TaxModel.EntityTaxInput(
        entity=ireland,
        gross_revenue=make_money(10000),
        outgoing_payments=[payment],
    )

    result = TaxModel.group_tax_outcome(
        TaxModel.GroupTaxOutcomeIn(entity_inputs_in=[entity_input])
    )

    assert result.total_group_tax == make_money(125)


def test_dutch_sandwich_beats_direct_payment():
    ireland = make_entity("Ireland", "Ireland")
    netherlands = make_entity("Netherlands", "Netherlands")
    bermuda = make_entity("Bermuda", "Bermuda")

    # Direct Ireland -> Bermuda
    direct_payment = make_payment(ireland, bermuda, 9000)

    direct = TaxModel.entity_tax_outcome(
        TaxModel.EntityTaxOutcomeIn(
            entity_in=ireland,
            gross_revenue_in=make_money(10000),
            outgoing_payments_in=[direct_payment],
        )
    )

    # Ireland -> Netherlands -> Bermuda
    p1 = make_payment(ireland, netherlands, 9000)
    p2 = make_payment(netherlands, bermuda, 8100)

    sandwich = TaxModel.group_tax_outcome(
        TaxModel.GroupTaxOutcomeIn(
            entity_inputs_in=[
                TaxModel.EntityTaxInput(
                    entity=ireland,
                    gross_revenue=make_money(10000),
                    outgoing_payments=[p1],
                ),
                TaxModel.EntityTaxInput(
                    entity=netherlands,
                    gross_revenue=make_money(9000),
                    outgoing_payments=[p2],
                ),
            ]
        )
    )

    assert sandwich.total_group_tax < direct.total_tax