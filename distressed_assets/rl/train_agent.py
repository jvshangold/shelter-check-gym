import torch

from distressed_assets.tax_env.env import (
    GIVE_VESTING_POWER,
    MAKE_SUBTRUST,
    MOVE_ASSET,
    SELL_ASSET,
    TaxEnv,
)
from distressed_assets.tax_env.state import AssetKind, OwnerType
from distressed_assets.rl.model import PolicyValueNet
from distressed_assets.rl.ppo import masked_categorical, ppo_update
from distressed_assets.rl.rollout import RolloutBuffer


def make_action_mask(env, device):
    mask = [False, False, False, False]

    mask[MAKE_SUBTRUST] = len(env.state.trusts) < env.max_trusts
    mask[MOVE_ASSET] = any(
        _can_move_asset(env, asset_id)
        for asset_id in _indexed_values(env.idx_to_asset, env.max_assets)
    )
    mask[SELL_ASSET] = any(
        _can_sell_asset(env, asset_id)
        for asset_id in _indexed_values(env.idx_to_asset, env.max_assets)
    )
    mask[GIVE_VESTING_POWER] = any(
        _can_give_vesting_power(env, trust_id, individual_id)
        for trust_id in _indexed_values(env.idx_to_trust, env.max_trusts)
        for individual_id in _indexed_values(
            env.idx_to_individual,
            env.max_individuals,
        )
    )

    return torch.tensor(mask, dtype=torch.bool, device=device)


def _indexed_values(index, max_count):
    return [index[i] for i in range(max_count) if i in index]


def make_trust_mask(env, action_type, device):
    mask = []

    for i in range(env.max_trusts):
        trust_id = env.idx_to_trust.get(i)

        if trust_id is None:
            mask.append(False)
        elif action_type == MAKE_SUBTRUST:
            mask.append(len(env.state.trusts) < env.max_trusts)
        elif action_type == MOVE_ASSET:
            mask.append(True)
        elif action_type == GIVE_VESTING_POWER:
            mask.append(
                any(
                    _can_give_vesting_power(env, trust_id, individual_id)
                    for individual_id in env.state.individuals
                )
            )
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_asset_mask(env, action_type, device):
    mask = []

    for i in range(env.max_assets):
        asset_id = env.idx_to_asset.get(i)

        if asset_id is None:
            mask.append(False)
        elif action_type == MOVE_ASSET:
            mask.append(_can_move_asset(env, asset_id))
        elif action_type == SELL_ASSET:
            mask.append(_can_sell_asset(env, asset_id))
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_individual_mask(env, trust_idx, device):
    trust_id = env.idx_to_trust[trust_idx]
    mask = []

    for i in range(env.max_individuals):
        individual_id = env.idx_to_individual.get(i)

        if individual_id is None:
            mask.append(False)
        else:
            mask.append(_can_give_vesting_power(env, trust_id, individual_id))

    return torch.tensor(mask, dtype=torch.bool, device=device)


def _can_move_asset(env, asset_id):
    return not env.state.assets[asset_id].is_sold and bool(env.state.trusts)


def _can_sell_asset(env, asset_id):
    asset = env.state.assets[asset_id]
    return (
        asset.kind == AssetKind.PROPERTY
        and not asset.is_sold
        and asset.owner_type == OwnerType.TRUST
    )


def _can_give_vesting_power(env, trust_id, individual_id):
    trust = env.state.trusts[trust_id]

    if trust.section_678_power_holder_id is not None:
        return False

    if trust.parent_trust_id is None:
        return False

    asset = env._find_property_asset_in_trust(trust_id)
    if asset is None:
        return False

    if not env._individual_has_sufficient_cash(individual_id, asset.fair_market_value):
        return False

    return env._find_foreign_individual_id() is not None


def sample_action(model, env, obs, device):
    out = model(obs)

    log_prob = 0.0
    entropy = 0.0

    action_mask = make_action_mask(env, device)
    dist_action = masked_categorical(out["action_logits"], action_mask)
    action_type = dist_action.sample()

    log_prob = log_prob + dist_action.log_prob(action_type)
    entropy = entropy + dist_action.entropy()

    action_type_int = action_type.item()
    action = {
        "action_type": action_type,
        "trust": None,
        "asset": None,
        "individual": None,
    }
    used_masks = {"action": action_mask}

    if action_type_int == MAKE_SUBTRUST:
        trust_mask = make_trust_mask(env, action_type_int, device)
        dist_trust = masked_categorical(out["trust_logits"], trust_mask)
        trust = dist_trust.sample()

        log_prob = log_prob + dist_trust.log_prob(trust)
        entropy = entropy + dist_trust.entropy()

        action["trust"] = trust
        used_masks["trust"] = trust_mask

    elif action_type_int == MOVE_ASSET:
        trust_mask = make_trust_mask(env, action_type_int, device)
        asset_mask = make_asset_mask(env, action_type_int, device)

        dist_trust = masked_categorical(out["trust_logits"], trust_mask)
        dist_asset = masked_categorical(out["asset_logits"], asset_mask)

        trust = dist_trust.sample()
        asset = dist_asset.sample()

        log_prob = log_prob + dist_trust.log_prob(trust)
        log_prob = log_prob + dist_asset.log_prob(asset)
        entropy = entropy + dist_trust.entropy()
        entropy = entropy + dist_asset.entropy()

        action["trust"] = trust
        action["asset"] = asset
        used_masks["trust"] = trust_mask
        used_masks["asset"] = asset_mask

    elif action_type_int == SELL_ASSET:
        asset_mask = make_asset_mask(env, action_type_int, device)
        dist_asset = masked_categorical(out["asset_logits"], asset_mask)
        asset = dist_asset.sample()

        log_prob = log_prob + dist_asset.log_prob(asset)
        entropy = entropy + dist_asset.entropy()

        action["asset"] = asset
        used_masks["asset"] = asset_mask

    elif action_type_int == GIVE_VESTING_POWER:
        trust_mask = make_trust_mask(env, action_type_int, device)
        dist_trust = masked_categorical(out["trust_logits"], trust_mask)
        trust = dist_trust.sample()

        individual_mask = make_individual_mask(env, trust.item(), device)
        dist_individual = masked_categorical(
            out["individual_logits"],
            individual_mask,
        )
        individual = dist_individual.sample()

        log_prob = log_prob + dist_trust.log_prob(trust)
        log_prob = log_prob + dist_individual.log_prob(individual)
        entropy = entropy + dist_trust.entropy()
        entropy = entropy + dist_individual.entropy()

        action["trust"] = trust
        action["individual"] = individual
        used_masks["trust"] = trust_mask
        used_masks["individual"] = individual_mask

    else:
        raise RuntimeError(f"Unknown action type: {action_type_int}")

    return action, log_prob, entropy, out["value"], used_masks


def action_dict_to_env_action(action):
    return (
        int(action["action_type"].item()),
        int(action["trust"].item()) if action["trust"] is not None else 0,
        int(action["asset"].item()) if action["asset"] is not None else 0,
        int(action["individual"].item()) if action["individual"] is not None else 0,
    )


def reset_training_env(env, device):
    obs, _ = env.reset()
    return obs.to(device)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TaxEnv(
        RANDOM_FOREIGN_PARTY_PROB=0.75,
        MAX_RANDOM_FOREIGN_PARTIES=2,
    )

    embed_dim = 128
    num_action_types = 4

    model = PolicyValueNet(
        embed_dim=embed_dim,
        num_action_types=num_action_types,
        max_trusts=env.max_trusts,
        max_assets=env.max_assets,
        max_individuals=env.max_individuals,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    buffer = RolloutBuffer()

    total_updates = 1000
    rollout_steps = 256
    max_success_images = 12
    saved_success_image_count = 0

    for update in range(total_updates):
        obs = reset_training_env(env, device)

        buffer.clear()
        cumulative_reward = 0.0
        reward_by_action = {i: [] for i in range(num_action_types)}
        invalid_count = 0
        positive_reward_count = 0
        saved_success_image = False

        for _ in range(rollout_steps):
            with torch.no_grad():
                action, log_prob, entropy, value, masks = sample_action(
                    model=model,
                    env=env,
                    obs=obs,
                    device=device,
                )

            env_action = action_dict_to_env_action(action)
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            next_obs = next_obs.to(device)

            reward_by_action[env_action[0]].append(reward)
            invalid_count += int(info.get("invalid_action", False))
            positive_reward_count += int(reward > 0.0)

            if (
                terminated
                and not saved_success_image
                and update % 25 == 0
                and saved_success_image_count < max_success_images
            ):
                env.render_world(filename=f"distressed_world_update_{update}")
                saved_success_image = True
                saved_success_image_count += 1

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
                obs = reset_training_env(env, device)
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

        if update % 25 == 0:
            print(
                "update: "
                f"{update}; loss: {loss}; "
                f"cumulative_reward: {cumulative_reward}; "
                f"invalid_count: {invalid_count}; "
                f"positive_reward_count: {positive_reward_count}"
            )
            for k, v in reward_by_action.items():
                if v:
                    print(k, sum(v) / len(v), len(v))


if __name__ == "__main__":
    train()
