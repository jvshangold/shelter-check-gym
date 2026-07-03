import copy
import os
from pathlib import Path

import torch

from barnes_group.rl.model import PolicyValueNet
from barnes_group.rl.ppo import masked_categorical, ppo_update
from barnes_group.rl.rollout import RolloutBuffer
from barnes_group.tax_env.env import (
    CONTRIBUTE_FOR_STOCK,
    MAKE_SUBCORPORATION,
    TRANSFER_CASH,
    TaxEnv,
)


ACTION_NAMES = {
    MAKE_SUBCORPORATION: "make_subcorporation",
    TRANSFER_CASH: "transfer_cash",
    CONTRIBUTE_FOR_STOCK: "contribute_for_stock",
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
    mask = [False, False, False]

    mask[MAKE_SUBCORPORATION] = len(env.state.corporations) < env.max_corporations
    mask[TRANSFER_CASH] = any(
        _can_transfer_cash(env, from_id, to_id, amount)
        for from_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
        for to_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
        for amount in env.cash_amounts
    )
    mask[CONTRIBUTE_FOR_STOCK] = (
        len(env.state.stock) < env.max_stocks
        and any(
            _can_contribute_for_stock(env, issuer_id, contributor_id, stock_id, amount_idx)
            for issuer_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
            for contributor_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
            for stock_id in _indexed_values(env.idx_to_stock, env.max_stocks)
            for amount_idx in range(len(env.cash_amounts))
        )
    )

    return torch.tensor(mask, dtype=torch.bool, device=device)


def _indexed_values(index, max_count):
    return [index[i] for i in range(max_count) if i in index]


def make_corp_a_mask(env, action_type, device):
    mask = []

    for i in range(env.max_corporations):
        corp_id = env.idx_to_corporation.get(i)

        if corp_id is None:
            mask.append(False)
        elif action_type == MAKE_SUBCORPORATION:
            mask.append(len(env.state.corporations) < env.max_corporations)
        elif action_type == TRANSFER_CASH:
            mask.append(
                any(
                    _can_transfer_cash(env, corp_id, to_id, amount)
                    for to_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
                    for amount in env.cash_amounts
                )
            )
        elif action_type == CONTRIBUTE_FOR_STOCK:
            mask.append(
                len(env.state.stock) < env.max_stocks
                and any(
                    _can_contribute_for_stock(env, corp_id, contributor_id, stock_id, amount_idx)
                    for contributor_id in _indexed_values(
                        env.idx_to_corporation,
                        env.max_corporations,
                    )
                    for stock_id in _indexed_values(env.idx_to_stock, env.max_stocks)
                    for amount_idx in range(len(env.cash_amounts))
                )
            )
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_corp_b_mask(env, action_type, device, corp_a_idx=None, stock_idx=None):
    mask = []
    corp_a_id = None if corp_a_idx is None else env.idx_to_corporation.get(corp_a_idx)
    stock_id = None if stock_idx is None else env.idx_to_stock.get(stock_idx)

    for i in range(env.max_corporations):
        corp_id = env.idx_to_corporation.get(i)

        if corp_id is None:
            mask.append(False)
        elif action_type == TRANSFER_CASH:
            if corp_a_id is None:
                mask.append(
                    any(
                        _can_transfer_cash(env, from_id, corp_id, amount)
                        for from_id in _indexed_values(
                            env.idx_to_corporation,
                            env.max_corporations,
                        )
                        for amount in env.cash_amounts
                    )
                )
            else:
                mask.append(
                    any(
                        _can_transfer_cash(env, corp_a_id, corp_id, amount)
                        for amount in env.cash_amounts
                    )
                )
        elif action_type == CONTRIBUTE_FOR_STOCK:
            if corp_a_id is None:
                mask.append(
                    len(env.state.stock) < env.max_stocks
                    and any(
                        _can_contribute_for_stock(env, issuer_id, corp_id, stock_id, amount_idx)
                        for issuer_id in _indexed_values(
                            env.idx_to_corporation,
                            env.max_corporations,
                        )
                        for stock_id in _indexed_values(env.idx_to_stock, env.max_stocks)
                        for amount_idx in range(len(env.cash_amounts))
                    )
                )
            else:
                mask.append(
                    len(env.state.stock) < env.max_stocks
                    and any(
                        _can_contribute_for_stock(env, corp_a_id, corp_id, stock_id, amount_idx)
                        for stock_id in _indexed_values(env.idx_to_stock, env.max_stocks)
                        for amount_idx in range(len(env.cash_amounts))
                    )
                )
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_stock_mask(env, action_type, device, corp_a_idx=None, corp_b_idx=None):
    mask = []
    corp_a_id = None if corp_a_idx is None else env.idx_to_corporation.get(corp_a_idx)
    corp_b_id = None if corp_b_idx is None else env.idx_to_corporation.get(corp_b_idx)

    for i in range(env.max_stocks):
        stock_id = env.idx_to_stock.get(i)

        if stock_id is None:
            mask.append(False)
        elif action_type == CONTRIBUTE_FOR_STOCK:
            if corp_a_id is None or corp_b_id is None:
                mask.append(
                    any(
                        _can_contribute_for_stock(env, issuer_id, contributor_id, stock_id, amount_idx)
                        for issuer_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
                        for contributor_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
                        for amount_idx in range(len(env.cash_amounts))
                    )
                )
            else:
                mask.append(
                    any(
                        _can_contribute_for_stock(env, corp_a_id, corp_b_id, stock_id, amount_idx)
                        for amount_idx in range(len(env.cash_amounts))
                    )
                )
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_amount_mask(env, action_type, device, corp_a_idx=None, corp_b_idx=None, stock_idx=None):
    max_amounts = max(len(env.cash_amounts), len(env.stock_percents))
    mask = []
    corp_a_id = None if corp_a_idx is None else env.idx_to_corporation.get(corp_a_idx)
    corp_b_id = None if corp_b_idx is None else env.idx_to_corporation.get(corp_b_idx)
    stock_id = None if stock_idx is None else env.idx_to_stock.get(stock_idx)

    for i in range(max_amounts):
        if action_type == TRANSFER_CASH:
            if i >= len(env.cash_amounts):
                mask.append(False)
            elif corp_a_id is None:
                amount = env.cash_amounts[i]
                mask.append(
                    any(
                        _has_cash_lot(env, from_id, amount)
                        for from_id in _indexed_values(
                            env.idx_to_corporation,
                            env.max_corporations,
                        )
                    )
                )
            else:
                mask.append(_has_cash_lot(env, corp_a_id, env.cash_amounts[i]))
        elif action_type == CONTRIBUTE_FOR_STOCK:
            if i >= len(env.cash_amounts) or i >= len(env.stock_percents):
                mask.append(False)
            elif corp_a_id is None or corp_b_id is None or stock_id is None:
                mask.append(False)
            else:
                mask.append(_can_contribute_for_stock(env, corp_a_id, corp_b_id, stock_id, i))
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def _can_transfer_cash(env, from_id, to_id, amount):
    if from_id == to_id:
        return False
    if from_id not in env.state.corporations or to_id not in env.state.corporations:
        return False
    if not _has_cash_lot(env, from_id, amount):
        return False

    from_foreign = env.state.corporations[from_id].is_foreign
    to_domestic = env.state.corporations[to_id].is_domestic
    direct_repatriation = (
        from_foreign
        and to_id == env.state.taxpayer_id
    )
    domestic_child_to_taxpayer = (
        env.state.corporations[from_id].is_domestic
        and to_id == env.state.taxpayer_id
        and env.state.is_subcorporation_of(from_id, to_id)
    )
    if not direct_repatriation and not domestic_child_to_taxpayer:
        return False

    if direct_repatriation and to_domestic:
        tax_due = min(
            env.state.ledger.direct_repatriated_cash + amount,
            env.state.applicable_earnings,
        ) * env.state.tax_rate
        incremental_tax = tax_due - env.state.ledger.tax_paid
        if incremental_tax > 0 and not _has_cash_lot_after_transfer(
            env,
            from_id,
            to_id,
            amount,
            env.state.taxpayer_id,
            incremental_tax,
        ):
            return False

    return True


def _has_cash_lot(env, owner_id, amount):
    return any(
        cash.owner_id == owner_id and cash.amount >= amount
        for cash in env.state.cash.values()
    )


def _has_cash_lot_after_transfer(env, from_id, to_id, transfer_amount, owner_id, amount):
    for cash in env.state.cash.values():
        if cash.owner_id == from_id and cash.amount >= transfer_amount:
            remaining_amount = cash.amount - transfer_amount
            if cash.owner_id == owner_id and remaining_amount >= amount:
                return True

        if cash.owner_id == owner_id and cash.amount >= amount:
            return True

    if to_id == owner_id and transfer_amount >= amount:
        return True

    return False


def _can_contribute_for_stock(env, issuer_id, contributor_id, stock_id, amount_idx):
    if issuer_id == contributor_id:
        return False
    if issuer_id not in env.state.corporations or contributor_id not in env.state.corporations:
        return False
    if stock_id not in env.state.stock:
        return False
    if amount_idx >= len(env.cash_amounts) or amount_idx >= len(env.stock_percents):
        return False
    if len(env.state.stock) >= env.max_stocks:
        return False

    stock = env.state.stock[stock_id]
    if stock.holder_id != contributor_id:
        return False
    if stock.issuer_id != contributor_id:
        return False
    if stock.percent <= 0.0:
        return False
    if not _has_cash_lot(env, contributor_id, env.cash_amounts[amount_idx]):
        return False
    return True


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
    entropy = torch.log(torch.tensor(float(valid.numel()), dtype=torch.float32, device=mask.device))
    return choice, log_prob, entropy


def sample_from_logits_or_uniform(logits, mask, force_uniform):
    if force_uniform:
        return masked_uniform(mask)

    dist = masked_categorical(logits, mask)
    sample = dist.sample()
    return sample.squeeze(), dist.log_prob(sample).squeeze(), dist.entropy().squeeze()


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
    action_type_int = action_type.item()

    log_prob = log_prob + action_log_prob
    entropy = entropy + action_entropy

    action = {
        "action_type": action_type,
        "corp_a": None,
        "corp_b": None,
        "stock": None,
        "amount": None,
    }
    used_masks = {"action": action_mask}

    if action_type_int in {MAKE_SUBCORPORATION, TRANSFER_CASH, CONTRIBUTE_FOR_STOCK}:
        corp_a_mask = make_corp_a_mask(env, action_type_int, device)
        corp_a, corp_a_log_prob, corp_a_entropy = sample_from_logits_or_uniform(
            out["corp_a_logits"],
            corp_a_mask,
            force_uniform,
        )

        log_prob = log_prob + corp_a_log_prob
        entropy = entropy + corp_a_entropy
        action["corp_a"] = corp_a
        used_masks["corp_a"] = corp_a_mask

    if action_type_int in {TRANSFER_CASH, CONTRIBUTE_FOR_STOCK}:
        corp_b_mask = make_corp_b_mask(
            env,
            action_type_int,
            device,
            corp_a_idx=action["corp_a"].item(),
        )
        corp_b, corp_b_log_prob, corp_b_entropy = sample_from_logits_or_uniform(
            out["corp_b_logits"],
            corp_b_mask,
            force_uniform,
        )

        log_prob = log_prob + corp_b_log_prob
        entropy = entropy + corp_b_entropy
        action["corp_b"] = corp_b
        used_masks["corp_b"] = corp_b_mask

    if action_type_int == CONTRIBUTE_FOR_STOCK:
        stock_mask = make_stock_mask(
            env,
            action_type_int,
            device,
            corp_a_idx=action["corp_a"].item(),
            corp_b_idx=action["corp_b"].item(),
        )
        stock, stock_log_prob, stock_entropy = sample_from_logits_or_uniform(
            out["stock_logits"],
            stock_mask,
            force_uniform,
        )

        log_prob = log_prob + stock_log_prob
        entropy = entropy + stock_entropy
        action["stock"] = stock
        used_masks["stock"] = stock_mask

    if action_type_int in {TRANSFER_CASH, CONTRIBUTE_FOR_STOCK}:
        corp_a_idx = None
        if action["corp_a"] is not None:
            corp_a_idx = action["corp_a"].item()
        corp_b_idx = None
        if action["corp_b"] is not None:
            corp_b_idx = action["corp_b"].item()
        stock_idx = None
        if action["stock"] is not None:
            stock_idx = action["stock"].item()

        amount_mask = make_amount_mask(
            env,
            action_type_int,
            device,
            corp_a_idx=corp_a_idx,
            corp_b_idx=corp_b_idx,
            stock_idx=stock_idx,
        )
        amount, amount_log_prob, amount_entropy = sample_from_logits_or_uniform(
            out["amount_logits"],
            amount_mask,
            force_uniform,
        )

        log_prob = log_prob + amount_log_prob
        entropy = entropy + amount_entropy
        action["amount"] = amount
        used_masks["amount"] = amount_mask

    return (
        action,
        torch.as_tensor(log_prob, device=device).squeeze(),
        torch.as_tensor(entropy, device=device).squeeze(),
        out["value"],
        used_masks,
        force_uniform,
    )


def action_dict_to_env_action(action):
    return (
        int(action["action_type"].item()),
        int(action["corp_a"].item()) if action["corp_a"] is not None else 0,
        int(action["corp_b"].item()) if action["corp_b"] is not None else 0,
        int(action["stock"].item()) if action["stock"] is not None else 0,
        int(action["amount"].item()) if action["amount"] is not None else 0,
    )


def reset_training_env(env, device):
    obs, _ = env.reset()
    return obs.to(device)


def has_valid_action(env, device):
    return bool(make_action_mask(env, device).any().item())


def describe_env_action(env, env_action):
    action_type, corp_a_idx, corp_b_idx, stock_idx, amount_idx = env_action
    action_name = ACTION_NAMES.get(action_type, f"unknown_{action_type}")

    if action_type == MAKE_SUBCORPORATION:
        parent_id = env.idx_to_corporation.get(corp_a_idx, f"idx_{corp_a_idx}")
        return f"{action_name}(parent={parent_id})"

    if action_type == TRANSFER_CASH:
        from_id = env.idx_to_corporation.get(corp_a_idx, f"idx_{corp_a_idx}")
        to_id = env.idx_to_corporation.get(corp_b_idx, f"idx_{corp_b_idx}")
        amount = env.cash_amounts[amount_idx]
        return f"{action_name}(from={from_id}, to={to_id}, amount={amount})"

    if action_type == CONTRIBUTE_FOR_STOCK:
        issuer_id = env.idx_to_corporation.get(corp_a_idx, f"idx_{corp_a_idx}")
        contributor_id = env.idx_to_corporation.get(corp_b_idx, f"idx_{corp_b_idx}")
        stock_id = env.idx_to_stock.get(stock_idx, f"idx_{stock_idx}")
        amount = env.cash_amounts[amount_idx]
        percent = env.stock_percents[amount_idx]
        return (
            f"{action_name}(issuer={issuer_id}, contributor={contributor_id}, "
            f"stock={stock_id}, cash={amount}, percent={percent})"
        )

    return str(env_action)


def save_best_snapshot(env, snapshot, output_dir, update, snapshot_count):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stem = output_dir / f"best_structure_update_{update:04d}"
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
        metadata.write(f"terminated: {snapshot['terminated']}\n")
        metadata.write(f"truncated: {snapshot['truncated']}\n")
        metadata.write(f"invalid_action: {snapshot['invalid_action']}\n")
        metadata.write("\nactions:\n")
        for i, action_description in enumerate(snapshot["actions"], start=1):
            metadata.write(f"{i}. {action_description}\n")


def train(total_updates=500, rollout_steps=256, save_snapshots=True, log_interval=25):
    device = resolve_device()

    env = TaxEnv(
        MAX_CORPORATIONS=5,
        MAX_STOCKS=8,
        MAX_STEPS=10,
    )

    embed_dim = 128
    num_action_types = 3
    max_amounts = max(len(env.cash_amounts), len(env.stock_percents))

    model = PolicyValueNet(
        embed_dim=embed_dim,
        num_action_types=num_action_types,
        max_corporations=env.max_corporations,
        max_stocks=env.max_stocks,
        max_amounts=max_amounts,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    buffer = RolloutBuffer()

    snapshot_dir = Path("barnes_group/rl_snapshots")
    saved_snapshot_count = 0

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
        entropy_total = 0.0
        best_tax_advantage = float("-inf")
        best_snapshot = None
        current_episode_actions = []

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
                current_episode_actions = []

            with torch.no_grad():
                action, log_prob, entropy, value, masks, used_uniform = sample_action(
                    model=model,
                    env=env,
                    obs=obs,
                    device=device,
                    random_action_prob=random_action_prob,
                )

            env_action = action_dict_to_env_action(action)
            action_description = describe_env_action(env, env_action)
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            next_obs = next_obs.to(device)

            current_episode_actions.append(action_description)
            reward_by_action[env_action[0]].append(reward)
            invalid_count += int(info.get("invalid_action", False))
            positive_reward_count += int(reward > 0.0)
            success_count += int(terminated)
            random_exploration_count += int(used_uniform)
            entropy_total += float(entropy.detach())

            tax_advantage = info.get("tax_advantage", 0.0)
            longer_equal_advantage_trace = (
                best_snapshot is not None
                and tax_advantage == best_tax_advantage
                and len(current_episode_actions) > len(best_snapshot["actions"])
            )
            if (
                best_snapshot is None
                or tax_advantage > best_tax_advantage
                or longer_equal_advantage_trace
            ):
                best_tax_advantage = tax_advantage
                best_snapshot = {
                    "state": copy.deepcopy(env.state),
                    "tax_advantage": tax_advantage,
                    "terminated": terminated,
                    "truncated": truncated,
                    "invalid_action": info.get("invalid_action", False),
                    "actions": list(current_episode_actions),
                }

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
                current_episode_actions = []
            else:
                obs = next_obs

        with torch.no_grad():
            last_value = model(obs)["value"]

        buffer.compute_advantages(last_value=last_value)
        loss = ppo_update(model=model, optimizer=optimizer, rollout=buffer)

        if (
            best_snapshot is not None
            and update % log_interval == 0
            and save_snapshots
        ):
            save_best_snapshot(
                env=env,
                snapshot=best_snapshot,
                output_dir=snapshot_dir,
                update=update,
                snapshot_count=saved_snapshot_count,
            )
            saved_snapshot_count += 1
            print(
                "snapshot_saved: "
                f"{(snapshot_dir / f'best_structure_update_{update:04d}.png').resolve()}",
                flush=True,
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
                f"avg_entropy: {entropy_total / rollout_steps:.4f}; "
                f"best_tax_advantage: {best_tax_advantage}; "
                f"random_action_prob: {random_action_prob:.4f}",
                flush=True,
            )
            for action_type, rewards in reward_by_action.items():
                if rewards:
                    print(
                        ACTION_NAMES[action_type],
                        sum(rewards) / len(rewards),
                        len(rewards),
                        flush=True,
                    )


if __name__ == "__main__":
    train()
