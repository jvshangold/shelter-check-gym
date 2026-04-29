import torch

from tax_env.env import TaxEnv
from rl.ppo import masked_categorical, ppo_update
from rl.rollout import RolloutBuffer
from rl.model import PolicyValueNet


def entity_to_idx(env, entity_id):
    for idx, eid in env.idx_to_entity.items():
        if eid == entity_id:
            return idx
    raise KeyError(entity_id)

def make_rent_licensee_mask(env, device):
    mask = []

    for licensee_idx, licensee_id in env.idx_to_entity.items():
        valid = any(
            env.is_valid_rent_pair(licensee_id, licensor_id)
            for licensor_id in env.state.entities
        )
        mask.append(valid)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_rent_licensor_mask(env, licensee_idx, device):
    licensee_id = env.idx_to_entity[licensee_idx]

    mask = []

    for licensor_idx, licensor_id in env.idx_to_entity.items():
        mask.append(env.is_valid_rent_pair(licensee_id, licensor_id))

    return torch.tensor(mask, dtype=torch.bool, device=device)

def make_entity_mask(env, device):
    return torch.tensor(env.get_entity_mask(), dtype=torch.bool, device=device)


def make_rent_dst_mask(env, src_idx, device):
    mask = []

    for i in range(len(env.idx_to_entity)):
        dst_id = env.idx_to_entity[i]

        valid = True

        if i == src_idx:
            valid = False

        if dst_id in env.state.licenses:
            valid = False

        mask.append(valid)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_transfer_dst_mask(env, device):
    owner_idx = entity_to_idx(env, env.state.ip_owner)

    mask = []
    for i in range(len(env.idx_to_entity)):
        mask.append(i != owner_idx)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def sample_action(model, env, obs, device):
    out = model(obs)

    log_prob = 0.0
    entropy = 0.0

    action_mask = torch.tensor(env.get_action_mask(), dtype=torch.bool, device=device)

    dist_action = masked_categorical(out["action_logits"], action_mask)
    action_type = dist_action.sample()

    log_prob += dist_action.log_prob(action_type)
    entropy += dist_action.entropy()

    action_type_int = action_type.item()

    action = {
        "action_type": action_type,
        "src": None,
        "dst": None,
        "incorporation": None,
        "management": None,
    }

    used_masks = {
        "action": action_mask,
    }

    if action_type_int == 0:
        # add_child: src = parent, incorporation/management used
        parent_mask = make_entity_mask(env, device)

        dist_parent = masked_categorical(out["src_logits"], parent_mask)
        parent = dist_parent.sample()

        dist_incorp = masked_categorical(out["incorporation_logits"])
        incorporation = dist_incorp.sample()

        dist_mgmt = masked_categorical(out["management_logits"])
        management = dist_mgmt.sample()

        log_prob += dist_parent.log_prob(parent)
        log_prob += dist_incorp.log_prob(incorporation)
        log_prob += dist_mgmt.log_prob(management)

        entropy += dist_parent.entropy()
        entropy += dist_incorp.entropy()
        entropy += dist_mgmt.entropy()

        action["src"] = parent
        action["incorporation"] = incorporation
        action["management"] = management

        used_masks["src"] = parent_mask

    elif action_type_int == 1:
        # rent_ip: src = licensee, dst = licensor / IP holder

        licensee_mask = make_rent_licensee_mask(env, device)

        dist_src = masked_categorical(out["src_logits"], licensee_mask)
        src = dist_src.sample()

        licensor_mask = make_rent_licensor_mask(env, src.item(), device)

        dist_dst = masked_categorical(out["dst_logits"], licensor_mask)
        dst = dist_dst.sample()

        log_prob += dist_src.log_prob(src)
        log_prob += dist_dst.log_prob(dst)

        entropy += dist_src.entropy()
        entropy += dist_dst.entropy()

        action["src"] = src
        action["dst"] = dst

        used_masks["src"] = licensee_mask
        used_masks["dst"] = licensor_mask

    elif action_type_int == 2:
        # transfer_ip: dst = new IP owner
        dst_mask = make_transfer_dst_mask(env, device)

        dist_dst = masked_categorical(out["dst_logits"], dst_mask)
        dst = dist_dst.sample()

        log_prob += dist_dst.log_prob(dst)
        entropy += dist_dst.entropy()

        action["dst"] = dst

        used_masks["dst"] = dst_mask

    else:
        raise RuntimeError(f"Unknown action type: {action_type_int}")      

    return action, log_prob, entropy, out["value"], used_masks


def action_dict_to_env_action(action):
    return (
        int(action["action_type"].item()),
        int(action["src"].item()) if action["src"] is not None else 0,
        int(action["dst"].item()) if action["dst"] is not None else 0,
        int(action["incorporation"].item()) if action["incorporation"] is not None else 0,
        int(action["management"].item()) if action["management"] is not None else 0,
    )


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TaxEnv()

    embed_dim = 128
    num_action_types = 3
    num_jurisdictions = 5

    model = PolicyValueNet(
        embed_dim=embed_dim,
        num_action_types=num_action_types,
        num_jurisdictions=num_jurisdictions,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    buffer = RolloutBuffer()

    total_updates = 1000
    rollout_steps = 256

    for update in range(total_updates):
        obs, _ = env.reset()
        obs = obs.to(device)

        buffer.clear()
        cumulative_reward = 0.0

        action_counts = {0: 0, 1: 0, 2: 0}

        for _ in range(rollout_steps):
            with torch.no_grad():
                
                action, log_prob, entropy, value, masks = sample_action(
                    model=model,
                    env=env,
                    obs=obs,
                    device=device,
                )

            env_action = action_dict_to_env_action(action)

            # for debugging
            action_counts[env_action[0]] += 1

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            next_obs = next_obs.to(device)

            buffer.add(
                obs=obs,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
                masks=masks,
            )

            cumulative_reward += reward

            if done:
                obs, _ = env.reset()
                obs = obs.to(device)
            else:
                obs = next_obs

        with torch.no_grad():
            last_value = model(obs)["value"]

        buffer.compute_advantages(last_value=last_value)

        loss = ppo_update(
            model=model,
            optimizer=optimizer,
            rollout=buffer,
        )

        print(f"update: {update}; loss: {loss}; cumulative_reward: {cumulative_reward}")
        print("action_counts:", action_counts)


if __name__ == "__main__":
    train()