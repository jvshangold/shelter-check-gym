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


# write test to check no licensing cycles
def test_no_license_cycle():
    state = WorldState()
    make_root(state)

    state.add_child("root", "Bermuda", "Ireland", "Bermuda", "Holding")     # company_1
    state.add_child("root", "Ireland", "Ireland", "Ireland", "Operating")  # company_2

    state.transfer_ip("company_1")
    state.rent_ip("company_2", "company_1")

    with pytest.raises(ValueError, match="This would create a licensing cycle"):
        state.rent_ip("company_1", "company_2")


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

    data = build_graph(state=state)

    assert data["entity"].x.shape == (4, 20)
    assert data["entity"].x[:, 0].max().item() <= 1.0
    assert data["entity", "has_subsidiary", "entity"].edge_index.shape == (2, 3)
    assert data["entity", "licenses_from", "entity"].edge_index.shape == (2, 2)


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

    assert env.idx_to_entity == {
        0: "root",
        1: "company_1",
    }
    assert env.state.entities["company_1"].parent_id == "root"
    assert env.state.entities["company_1"].tax_residence == "Bermuda"
    assert env.state.entities["company_1"].company_type == "Holding"


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
    # incorporation = Ireland
    # management = Ireland
    # company_type = Operating
    action = (0, 0, 0, 0, 0, 1)
    env.step(action=action)

    assert "company_2" in env.state.entities
    assert env.state.entities["company_2"].company_type == "Operating"


def test_default_entity_limit_allows_initial_bermuda_and_two_more_children():
    env = TaxEnv()
    env.reset()

    env.step((0, 0, 0, 0, 0, 1))
    env.step((0, 0, 0, 1, 1, 0))

    assert len(env.state.entities) == 4
    assert env.get_action_mask()[0] == 0


def test_compute_profit():
    env = TaxEnv()
    env.reset()

    profit = env.compute_profit()

    assert profit is not None


def test_direct_parent_child_license_is_invalid():
    env = TaxEnv()
    env.reset()

    env.step((2, 0, 1, 0, 0, 0))  # transfer IP to Bermuda
    _, reward, terminated, _, info = env.step((1, 0, 1, 0, 0, 0))  # root rents from Bermuda

    assert info["invalid_action"]
    assert info["tax_advantage"] == 0.0
    assert not info["loophole_gate_complete"]
    assert reward < 0.0
    assert not terminated


def test_transfer_ip_requires_holding_company_for_root_too():
    env = TaxEnv()
    env.reset()

    env.step((2, 0, 1, 0, 0, 0))  # transfer IP to Bermuda
    _, reward, _, _, info = env.step((2, 0, 0, 0, 0, 0))  # try transfer back to root

    assert info["invalid_action"]
    assert reward < 0.0
    assert env.state.ip_owner == "company_1"
    assert env.state.entities["root"].company_type == "Operating"


def test_inbound_royalty_is_taxable_to_recipient():
    env = TaxEnv()
    env.reset()

    env.step((0, 0, 0, 1, 1, 0))  # company_2: Netherlands holding
    env.step((0, 0, 0, 4, 4, 1))  # company_3: Germany operating
    env.step((2, 0, 2, 0, 0, 0))  # transfer IP to Netherlands
    _, _, _, _, info = env.step((1, 3, 2, 0, 0, 0))  # Germany rents from Netherlands

    assert info["current_profit"] < info["baseline_profit"]
    assert info["raw_tax_advantage"] == 0.0


def test_invalid_parent_child_rent_ip_does_not_change_company_types():
    env = TaxEnv()
    env.reset()

    _, reward, _, _, info = env.step((1, 1, 0, 0, 0, 0))  # Bermuda rents from root

    assert info["invalid_action"]
    assert env.state.entities["root"].company_type == "Operating"
    assert env.state.entities["company_1"].company_type == "Holding"
    assert reward < 0.0


def test_transfer_ip_requires_holding_company():
    env = TaxEnv()
    env.reset()
    env.step((0, 0, 0, 0, 0, 1))  # company_2: Ireland operating

    _, reward, _, _, info = env.step((2, 0, 2, 0, 0, 0))

    assert info["invalid_action"]
    assert reward < 0.0
    assert env.state.ip_owner == "root"
    assert env.state.entities["company_2"].company_type == "Operating"


def test_sibling_royalty_chain_can_produce_tax_advantage():
    env = TaxEnv()
    env.reset()

    env.step((0, 0, 0, 0, 0, 1))  # company_2: Ireland operating
    env.step((0, 0, 0, 1, 1, 0))  # company_3: Netherlands holding
    env.step((2, 0, 1, 0, 0, 0))  # transfer IP to Bermuda
    env.step((1, 3, 1, 0, 0, 0))  # Netherlands rents from Bermuda
    _, reward, terminated, _, info = env.step((1, 2, 3, 0, 0, 0))  # Ireland rents from Netherlands

    assert info["raw_tax_advantage"] > 0.0
    assert info["tax_advantage"] == info["raw_tax_advantage"]
    assert info["normalized_tax_advantage"] > 0.0
    assert info["loophole_gate_complete"]
    assert reward > 0.0
    assert terminated


def test_market_revenue_middle_company_shapes_but_does_not_complete():
    env = TaxEnv()
    env.reset()

    env.step((0, 0, 0, 1, 1, 1))  # company_2: Netherlands operating
    env.step((2, 0, 1, 0, 0, 0))  # transfer IP to Bermuda holding
    _, _, terminated, _, info = env.step(
        (1, 2, 1, 0, 0, 0)
    )  # Netherlands operating rents from Bermuda

    assert info["raw_tax_advantage"] > 0.0
    assert info["tax_advantage"] == 0.0
    assert not info["loophole_gate_complete"]
    assert not terminated


def test_completion_gate_is_not_irish_specific():
    env = TaxEnv()
    env.reset()

    env.step((0, 0, 0, 4, 4, 1))  # company_2: Germany operating
    env.step((0, 0, 0, 1, 1, 0))  # company_3: Netherlands holding
    env.step((2, 0, 1, 0, 0, 0))  # transfer IP to Bermuda
    env.step((1, 3, 1, 0, 0, 0))  # Netherlands rents from Bermuda
    _, reward, terminated, _, info = env.step((1, 2, 3, 0, 0, 0))  # Germany rents from Netherlands

    assert info["raw_tax_advantage"] > 0.0
    assert info["tax_advantage"] == info["raw_tax_advantage"]
    assert info["loophole_gate_complete"]
    assert reward > 0.0
    assert terminated
