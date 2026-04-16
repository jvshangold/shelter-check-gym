import pytest

from tax_env.render import build_graph
from tax_env.state import WorldState, Entity
from tax_env.model import GNN

# write test to add root and test revenue
def test_revenue():
    state = WorldState()
    state.entities["root"] = Entity("root", "US", "US", "US")
    state.ip_owner = "root"

    print(state.get_company_revenue("root"))
    assert state.get_company_revenue("root") > 0

# test add_child
def test_add_child():
    state = WorldState()
    state.entities["root"] = Entity("root", "US", "US", "US")
    state.ip_owner = "root"

    state.add_child("root", "Bermuda", "Ireland", "Bermuda")

    child_id = "company_1"

    assert child_id in state.entities

# test to get more complex state (sandwich)
def test_sandwich():
    state = WorldState()
    state.entities["root"] = Entity("root", "US", "US", "US")
    state.ip_owner = "root"

    # create subsidiary hierarchy
    state.add_child("root", "Bermuda", "Ireland", "Bermuda") # company_1
    state.add_child("root", "Ireland", "Ireland", "Ireland") # company_2
    state.add_child("root", "Netherlands", "Netherlands", "Netherlands") # company_3

    # create licensing structure
    state.transfer_ip("company_1") # give ip to tax haven
    state.rent_ip("company_3", "company_1")
    state.rent_ip("company_2", "company_3")


# write test to check no licensing cycles
def test_no_license_cycle():
    state = WorldState()
    state.entities["root"] = Entity("root", "US", "US", "US")
    state.ip_owner = "root"

    state.add_child("root", "Bermuda", "Ireland", "Bermuda") # company_1
    state.add_child("root", "Ireland", "Ireland", "Ireland") # company_2

    state.rent_ip("company_1", "root")
    state.rent_ip("company_2", "company_1")
    with pytest.raises(ValueError, match="This would create a licensing cycle"):
        state.rent_ip("root", "company_2")

# write test to put complex state into build_graph
def test_build_graph():
    state = WorldState()
    state.entities["root"] = Entity("root", "US", "US", "US")
    state.ip_owner = "root"

    # create subsidiary hierarchy
    state.add_child("root", "Bermuda", "Ireland", "Bermuda") # company_1
    state.add_child("root", "Ireland", "Ireland", "Ireland") # company_2
    state.add_child("root", "Netherlands", "Netherlands", "Netherlands") # company_3

    # create licensing structure
    state.transfer_ip("company_1") # give ip to tax haven
    state.rent_ip("company_3", "company_1")
    state.rent_ip("company_2", "company_3")

    build_graph(state=state)

def test_GNN_dims():
    state = WorldState()
    state.entities["root"] = Entity("root", "US", "US", "US")
    state.ip_owner = "root"

    # create subsidiary hierarchy
    state.add_child("root", "Bermuda", "Ireland", "Bermuda") # company_1
    state.add_child("root", "Ireland", "Ireland", "Ireland") # company_2
    state.add_child("root", "Netherlands", "Netherlands", "Netherlands") # company_3

    # create licensing structure
    state.transfer_ip("company_1") # give ip to tax haven
    state.rent_ip("company_3", "company_1")
    state.rent_ip("company_2", "company_3")

    data = build_graph(state=state)
    model = GNN(64, 32)
    model(data)