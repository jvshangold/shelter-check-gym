import sys

sys.path.append("formalizations/_target/tax_rules")
sys.path.append("formalizations/_build/libcatala/python")

from python import TaxModel
import catala_runtime


def make_jurisdiction(name: str):
    return TaxModel.Jurisdiction(
        getattr(TaxModel.Jurisdiction_Code, name),
        catala_runtime.Unit(),
    )


def make_payment_kind(name: str):
    return TaxModel.PaymentKind(
        getattr(TaxModel.PaymentKind_Code, name),
        catala_runtime.Unit(),
    )


def make_entity(incorporation: str, tax_residence: str):
    return TaxModel.Entity(
        incorporation_jurisdiction=make_jurisdiction(incorporation),
        tax_residence=make_jurisdiction(tax_residence),
    )


def make_money(amount: int):
    return catala_runtime.Money(catala_runtime.Integer(amount))


def make_payment(payer, receiver, amount: int, kind: str):
    return TaxModel.Payment(
        payer=payer,
        receiver=receiver,
        amount=make_money(amount),
        kind=make_payment_kind(kind),
    )


def test_withholding_ireland_to_netherlands_royalty_zero():
    payer = make_entity("Ireland", "Ireland")
    receiver = make_entity("Netherlands", "Netherlands")
    payment = make_payment(payer, receiver, 9000, "Royalty")

    result = TaxModel.withholding_tax_computation(
        TaxModel.WithholdingTaxComputationIn(payment_in=payment)
    )

    assert result.withholding_tax_due == make_money(0)


def test_withholding_ireland_to_bermuda_royalty_20_percent():
    payer = make_entity("Ireland", "Ireland")
    receiver = make_entity("Bermuda", "Bermuda")
    payment = make_payment(payer, receiver, 9000, "Royalty")

    result = TaxModel.withholding_tax_computation(
        TaxModel.WithholdingTaxComputationIn(payment_in=payment)
    )

    assert result.withholding_tax_due == make_money(1800)


def test_withholding_us_royalty_10_percent():
    payer = make_entity("US", "US")
    receiver = make_entity("Bermuda", "Bermuda")
    payment = make_payment(payer, receiver, 9000, "Royalty")

    result = TaxModel.withholding_tax_computation(
        TaxModel.WithholdingTaxComputationIn(payment_in=payment)
    )

    assert result.withholding_tax_due == make_money(900)


def test_corporate_tax_ireland_100_less_90_royalty():
    ireland = make_entity("Ireland", "Ireland")
    dutch = make_entity("Netherlands", "Netherlands")
    payment = make_payment(ireland, dutch, 9000, "Royalty")

    result = TaxModel.corporate_tax_computation(
        TaxModel.CorporateTaxComputationIn(
            entity_in=ireland,
            gross_revenue_in=make_money(10000),
            outgoing_payments_in=[payment],
        )
    )

    assert result.tax_due == make_money(125)


def test_entity_tax_outcome_ireland_to_netherlands():
    ireland = make_entity("Ireland", "Ireland")
    dutch = make_entity("Netherlands", "Netherlands")
    payment = make_payment(ireland, dutch, 9000, "Royalty")

    result = TaxModel.entity_tax_outcome(
        TaxModel.EntityTaxOutcomeIn(
            entity_in=ireland,
            gross_revenue_in=make_money(10000),
            outgoing_payments_in=[payment],
        )
    )

    assert result.total_tax == make_money(125)


def test_entity_tax_outcome_ireland_to_bermuda():
    ireland = make_entity("Ireland", "Ireland")
    bermuda = make_entity("Bermuda", "Bermuda")
    payment = make_payment(ireland, bermuda, 9000, "Royalty")

    result = TaxModel.entity_tax_outcome(
        TaxModel.EntityTaxOutcomeIn(
            entity_in=ireland,
            gross_revenue_in=make_money(10000),
            outgoing_payments_in=[payment],
        )
    )

    assert result.total_tax == make_money(1925)


def test_group_tax_outcome_single_entity():
    ireland = make_entity("Ireland", "Ireland")
    dutch = make_entity("Netherlands", "Netherlands")
    payment = make_payment(ireland, dutch, 9000, "Royalty")

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

    # direct Ireland -> Bermuda
    direct_payment = make_payment(ireland, bermuda, 9000, "Royalty")

    direct = TaxModel.entity_tax_outcome(
        TaxModel.EntityTaxOutcomeIn(
            entity_in=ireland,
            gross_revenue_in=make_money(10000),
            outgoing_payments_in=[direct_payment],
        )
    )

    # Ireland -> Netherlands -> Bermuda
    p1 = make_payment(ireland, netherlands, 9000, "Royalty")
    p2 = make_payment(netherlands, bermuda, 8100, "Royalty")

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