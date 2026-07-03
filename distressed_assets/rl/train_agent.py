import copy
import os
from pathlib import Path

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


ACTION_NAMES = {
    MAKE_SUBTRUST: "make_subtrust",
    MOVE_ASSET: "move_asset",
    SELL_ASSET: "sell_asset",
    GIVE_VESTING_POWER: "give_vesting_power",
}


def resolve_device():
    requested = os.environ.get("SHELTER_CHECK_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("requested_device_unavailable: cuda; using cpu", flush=True)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_action_mask(env, device):
    mask = [False, False, False, False]

    mask[MAKE_SUBTRUST] = len(env.state.trusts) < env.max_trusts
    mask[MOVE_ASSET] = any(
        _can_move_asset(env, asset_id, trust_id)
        for asset_id in _indexed_values(env.idx_to_asset, env.max_assets)
        for trust_id in _indexed_values(env.idx_to_trust, env.max_trusts)
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


def make_trust_mask(env, action_type, device, asset_idx=None):
    mask = []
    asset_id = None
    if asset_idx is not None:
        asset_id = env.idx_to_asset.get(asset_idx)

    for i in range(env.max_trusts):
        trust_id = env.idx_to_trust.get(i)

        if trust_id is None:
            mask.append(False)
        elif action_type == MAKE_SUBTRUST:
            mask.append(len(env.state.trusts) < env.max_trusts)
        elif action_type == MOVE_ASSET:
            if asset_id is None:
                mask.append(
                    any(
                        _can_move_asset(env, candidate_asset_id, trust_id)
                        for candidate_asset_id in _indexed_values(
                            env.idx_to_asset,
                            env.max_assets,
                        )
                    )
                )
            else:
                mask.append(_can_move_asset(env, asset_id, trust_id))
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
            mask.append(
                any(
                    _can_move_asset(env, asset_id, trust_id)
                    for trust_id in _indexed_values(env.idx_to_trust, env.max_trusts)
                )
            )
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


def _can_move_asset(env, asset_id, dst_trust_id=None):
    asset = env.state.assets[asset_id]

    if asset.kind != AssetKind.PROPERTY:
        return False

    if asset.is_sold:
        return False

    if not env.state.trusts:
        return False

    if dst_trust_id is None:
        return True

    if dst_trust_id not in env.state.trusts:
        return False

    return not (
        asset.owner_type == OwnerType.TRUST
        and asset.owner_id == dst_trust_id
    )


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


def masked_uniform(mask):
    valid = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        raise ValueError(f"All actions masked out. mask={mask}")

    choice = valid[torch.randint(valid.numel(), (1,), device=mask.device)].squeeze(0)
    log_prob = torch.log(torch.tensor(
        1.0 / valid.numel(),
        dtype=torch.float32,
        device=mask.device,
    ))
    entropy = torch.log(torch.tensor(
        float(valid.numel()),
        dtype=torch.float32,
        device=mask.device,
    ))

    return choice, log_prob, entropy


def sample_from_logits_or_uniform(logits, mask, force_uniform):
    if force_uniform:
        return masked_uniform(mask)

    dist = masked_categorical(logits, mask)
    sample = dist.sample()
    return sample, dist.log_prob(sample), dist.entropy()


def sample_action(model, env, obs, device, random_action_prob=0.0):
    out = model(obs)

    log_prob = 0.0
    entropy = 0.0
    force_uniform = torch.rand((), device=device).item() < random_action_prob

    action_mask = make_action_mask(env, device)
    action_type, action_log_prob, action_entropy = sample_from_logits_or_uniform(
        out["action_logits"],
        action_mask,
        force_uniform,
    )

    log_prob = log_prob + action_log_prob
    entropy = entropy + action_entropy

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
        trust, trust_log_prob, trust_entropy = sample_from_logits_or_uniform(
            out["trust_logits"],
            trust_mask,
            force_uniform,
        )

        log_prob = log_prob + trust_log_prob
        entropy = entropy + trust_entropy

        action["trust"] = trust
        used_masks["trust"] = trust_mask

    elif action_type_int == MOVE_ASSET:
        asset_mask = make_asset_mask(env, action_type_int, device)
        asset, asset_log_prob, asset_entropy = sample_from_logits_or_uniform(
            out["asset_logits"],
            asset_mask,
            force_uniform,
        )

        trust_mask = make_trust_mask(
            env,
            action_type_int,
            device,
            asset_idx=asset.item(),
        )
        trust, trust_log_prob, trust_entropy = sample_from_logits_or_uniform(
            out["trust_logits"],
            trust_mask,
            force_uniform,
        )

        log_prob = log_prob + asset_log_prob
        entropy = entropy + asset_entropy
        log_prob = log_prob + trust_log_prob
        entropy = entropy + trust_entropy

        action["trust"] = trust
        action["asset"] = asset
        used_masks["trust"] = trust_mask
        used_masks["asset"] = asset_mask

    elif action_type_int == SELL_ASSET:
        asset_mask = make_asset_mask(env, action_type_int, device)
        asset, asset_log_prob, asset_entropy = sample_from_logits_or_uniform(
            out["asset_logits"],
            asset_mask,
            force_uniform,
        )

        log_prob = log_prob + asset_log_prob
        entropy = entropy + asset_entropy

        action["asset"] = asset
        used_masks["asset"] = asset_mask

    elif action_type_int == GIVE_VESTING_POWER:
        trust_mask = make_trust_mask(env, action_type_int, device)
        trust, trust_log_prob, trust_entropy = sample_from_logits_or_uniform(
            out["trust_logits"],
            trust_mask,
            force_uniform,
        )

        individual_mask = make_individual_mask(env, trust.item(), device)
        individual, individual_log_prob, individual_entropy = sample_from_logits_or_uniform(
            out["individual_logits"],
            individual_mask,
            force_uniform,
        )

        log_prob = log_prob + trust_log_prob
        log_prob = log_prob + individual_log_prob
        entropy = entropy + trust_entropy
        entropy = entropy + individual_entropy

        action["trust"] = trust
        action["individual"] = individual
        used_masks["trust"] = trust_mask
        used_masks["individual"] = individual_mask

    else:
        raise RuntimeError(f"Unknown action type: {action_type_int}")

    return action, log_prob, entropy, out["value"], used_masks, force_uniform


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


def has_valid_action(env, device) -> bool:
    return bool(make_action_mask(env, device).any().item())


def save_success_snapshot(env, snapshot, output_dir, update, snapshot_count):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stem = output_dir / f"success_structure_update_{update:04d}"
    old_state = env.state

    try:
        env.state = snapshot["state"]
        env.render_world(filename=str(image_stem))
    finally:
        env.state = old_state

    metadata_path = image_stem.with_suffix(".txt")
    with metadata_path.open("w") as metadata:
        metadata.write(f"update: {update}\n")
        metadata.write(f"snapshot_index: {snapshot_count}\n")
        metadata.write(f"tax_advantage: {snapshot['tax_advantage']}\n")
        metadata.write(f"raw_tax_savings: {snapshot['raw_tax_savings']}\n")
        metadata.write(f"terminated: {snapshot['terminated']}\n")
        metadata.write(f"truncated: {snapshot['truncated']}\n")
        metadata.write(f"invalid_action: {snapshot['invalid_action']}\n")


def train(total_updates=500, rollout_steps=256, save_snapshots=True, log_interval=25):
    device = resolve_device()

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

    snapshot_dir = Path("distressed_assets/rl_snapshots")
    max_success_images = 12
    saved_success_image_count = 0

    for update in range(total_updates):
        obs = reset_training_env(env, device)

        buffer.clear()
        cumulative_reward = 0.0
        reward_by_action = {i: [] for i in range(num_action_types)}
        invalid_count = 0
        positive_reward_count = 0
        success_count = 0
        episode_count = 0
        random_exploration_count = 0
        saved_success_image = False
        random_action_prob = max(0.02, 0.25 * (0.995 ** update))

        if update % log_interval == 0:
            print(
                f"update_start: {update}; "
                f"random_action_prob: {random_action_prob:.4f}",
                flush=True,
            )

        for _ in range(rollout_steps):
            if not has_valid_action(env, device):
                obs = reset_training_env(env, device)

            with torch.no_grad():
                action, log_prob, entropy, value, masks, used_uniform = sample_action(
                    model=model,
                    env=env,
                    obs=obs,
                    device=device,
                    random_action_prob=random_action_prob,
                )

            env_action = action_dict_to_env_action(action)
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            next_obs = next_obs.to(device)

            reward_by_action[env_action[0]].append(reward)
            invalid_count += int(info.get("invalid_action", False))
            positive_reward_count += int(reward > 0.0)
            success_count += int(terminated)
            random_exploration_count += int(used_uniform)

            if (
                terminated
                and save_snapshots
                and not saved_success_image
                and update % log_interval == 0
                and saved_success_image_count < max_success_images
            ):
                save_success_snapshot(
                    env=env,
                    snapshot={
                        "state": copy.deepcopy(env.state),
                        "tax_advantage": info.get("tax_advantage", 0.0),
                        "raw_tax_savings": info.get("raw_tax_savings", 0.0),
                        "terminated": terminated,
                        "truncated": truncated,
                        "invalid_action": info.get("invalid_action", False),
                    },
                    output_dir=snapshot_dir,
                    update=update,
                    snapshot_count=saved_success_image_count,
                )
                saved_success_image = True
                saved_success_image_count += 1
                print(
                    "snapshot_saved: "
                    f"{(snapshot_dir / f'success_structure_update_{update:04d}.png').resolve()}",
                    flush=True,
                )

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
                episode_count += 1
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

        if update % log_interval == 0:
            print(
                "update_complete: "
                f"{update}; loss: {loss}; "
                f"cumulative_reward: {cumulative_reward}; "
                f"invalid_count: {invalid_count}; "
                f"positive_reward_count: {positive_reward_count}; "
                f"success_count: {success_count}; "
                f"episode_count: {episode_count}; "
                f"random_exploration_count: {random_exploration_count}; "
                f"random_action_prob: {random_action_prob:.4f}",
                flush=True,
            )
            for k, v in reward_by_action.items():
                if v:
                    print(ACTION_NAMES.get(k, k), sum(v) / len(v), len(v), flush=True)


if __name__ == "__main__":
    train()
