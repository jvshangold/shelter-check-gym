import pytest

from dub_irish_dutch.tax_env.render import build_graph
from dub_irish_dutch.tax_env.state import WorldState, Entity
from dub_irish_dutch.tax_env.model import GNN
from dub_irish_dutch.tax_env.env import TaxEnv


def make_root(state: WorldState):
    state.entities["root"] = Entity(
        id="root",
        incorporation_jurisdiction="US",
        management_jurisdiction="US",
        tax_residence="US",
        company_type="Operating",
    )
    state.ip_owner = "root"


# write test to add root and test revenue
def test_revenue():
    state = WorldState()
    make_root(state)

    print(state.get_company_revenue("root"))
    assert state.get_company_revenue("root") > 0


# test add_child
def test_add_child():
    state = WorldState()
    make_root(state)

    state.add_child(
        parent="root",
        incorporation_jurisdiction="Bermuda",
        management_jurisdiction="Ireland",
        tax_residence="Bermuda",
        company_type="Holding",
    )

    child_id = "company_1"

    assert child_id in state.entities
    assert state.entities[child_id].company_type == "Holding"


# test to get more complex state (sandwich)
def test_sandwich():
    state = WorldState()
    make_root(state)

    # create subsidiary hierarchy
    state.add_child("root", "Bermuda", "Ireland", "Bermuda", "Holding")          # company_1
    state.add_child("root", "Ireland", "Ireland", "Ireland", "Operating")       # company_2
    state.add_child("root", "Netherlands", "Netherlands", "Netherlands", "Holding")  # company_3

    # create licensing structure
    state.transfer_ip("company_1")  # give ip to tax haven
    state.rent_ip("company_3", "company_1")
    state.rent_ip("company_2", "company_3")

    assert state.has_irish_sandwich()


# write test to check no licensing cycles
def test_no_license_cycle():
    state = WorldState()
    make_root(state)

    state.add_child("root", "Bermuda", "Ireland", "Bermuda", "Holding")     # company_1
    state.add_child("root", "Ireland", "Ireland", "Ireland", "Operating")  # company_2

    state.rent_ip("company_1", "root")
    state.rent_ip("company_2", "company_1")

    with pytest.raises(ValueError, match="This would create a licensing cycle"):
        state.rent_ip("root", "company_2")


# write test to put complex state into build_graph
def test_build_graph():
    state = WorldState()
    make_root(state)

    # create subsidiary hierarchy
    state.add_child("root", "Bermuda", "Ireland", "Bermuda", "Holding")          # company_1
    state.add_child("root", "Ireland", "Ireland", "Ireland", "Operating")       # company_2
    state.add_child("root", "Netherlands", "Netherlands", "Netherlands", "Holding")  # company_3

    # create licensing structure
    state.transfer_ip("company_1")  # give ip to tax haven
    state.rent_ip("company_3", "company_1")
    state.rent_ip("company_2", "company_3")

    build_graph(state=state)


def test_GNN_dims():
    state = WorldState()
    make_root(state)

    # create subsidiary hierarchy
    state.add_child("root", "Bermuda", "Ireland", "Bermuda", "Holding")          # company_1
    state.add_child("root", "Ireland", "Ireland", "Ireland", "Operating")       # company_2
    state.add_child("root", "Netherlands", "Netherlands", "Netherlands", "Holding")  # company_3

    # create licensing structure
    state.transfer_ip("company_1")  # give ip to tax haven
    state.rent_ip("company_3", "company_1")
    state.rent_ip("company_2", "company_3")

    data = build_graph(state=state)
    model = GNN(64, 32)
    model(data)


def test_initialize_env():
    env = TaxEnv()
    env.reset()


def test_take_step():
    env = TaxEnv()
    env.reset()

    # action = (
    #   action_type,
    #   arg_1,
    #   arg_2,
    #   incorporation_jurisdiction,
    #   management_jurisdiction,
    #   company_type,
    # )
    #
    # company_type:
    #   0 = Holding
    #   1 = Operating

    # add child:
    # parent = root
    # incorporation = Bermuda
    # management = Ireland
    # company_type = Holding
    action = (0, 0, 0, 2, 0, 0)
    env.step(action=action)

    assert "company_1" in env.state.entities
    assert env.state.entities["company_1"].company_type == "Holding"


def test_compute_profit():
    env = TaxEnv()
    env.reset()

    # add child:
    # parent = root
    # incorporation = Bermuda
    # management = Ireland
    # company_type = Holding
    env.step((0, 0, 0, 2, 0, 0))

    # license root from child:
    # licensee = root
    # licensor = company_1
    #
    # Since action_type = 1 ignores jurisdiction/company type args,
    # the last three values are just filler.
    env.step((1, 0, 1, 0, 0, 0))

    profit = env.compute_profit()

    assert profit is not None