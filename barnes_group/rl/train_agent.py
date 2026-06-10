import copy
from pathlib import Path

import torch

from barnes_group.rl.model import PolicyValueNet
from barnes_group.rl.ppo import masked_categorical, ppo_update
from barnes_group.rl.rollout import RolloutBuffer
from barnes_group.tax_env.env import (
    ISSUE_STOCK,
    MAKE_SUBCORPORATION,
    TRANSFER_CASH,
    TRANSFER_STOCK,
    TaxEnv,
)


ACTION_NAMES = {
    MAKE_SUBCORPORATION: "make_subcorporation",
    TRANSFER_CASH: "transfer_cash",
    TRANSFER_STOCK: "transfer_stock",
    ISSUE_STOCK: "issue_stock",
}


def make_action_mask(env, device):
    mask = [False, False, False, False]

    mask[MAKE_SUBCORPORATION] = len(env.state.corporations) < env.max_corporations
    mask[TRANSFER_CASH] = any(
        _can_transfer_cash(env, from_id, to_id, amount)
        for from_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
        for to_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
        for amount in env.cash_amounts
    )
    mask[TRANSFER_STOCK] = any(
        _can_transfer_stock(env, stock_id, to_id)
        for stock_id in _indexed_values(env.idx_to_stock, env.max_stocks)
        for to_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
    )
    mask[ISSUE_STOCK] = (
        len(env.state.stock) < env.max_stocks
        and any(
            _can_issue_stock(env, issuer_id, recipient_id)
            for issuer_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
            for recipient_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
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
        elif action_type == ISSUE_STOCK:
            mask.append(
                len(env.state.stock) < env.max_stocks
                and any(
                    _can_issue_stock(env, corp_id, recipient_id)
                    for recipient_id in _indexed_values(
                        env.idx_to_corporation,
                        env.max_corporations,
                    )
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
        elif action_type == TRANSFER_STOCK:
            if stock_id is None:
                mask.append(
                    any(
                        _can_transfer_stock(env, candidate_stock_id, corp_id)
                        for candidate_stock_id in _indexed_values(
                            env.idx_to_stock,
                            env.max_stocks,
                        )
                    )
                )
            else:
                mask.append(_can_transfer_stock(env, stock_id, corp_id))
        elif action_type == ISSUE_STOCK:
            if corp_a_id is None:
                mask.append(
                    len(env.state.stock) < env.max_stocks
                    and any(
                        _can_issue_stock(env, issuer_id, corp_id)
                        for issuer_id in _indexed_values(
                            env.idx_to_corporation,
                            env.max_corporations,
                        )
                    )
                )
            else:
                mask.append(
                    len(env.state.stock) < env.max_stocks
                    and _can_issue_stock(env, corp_a_id, corp_id)
                )
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_stock_mask(env, action_type, device):
    mask = []

    for i in range(env.max_stocks):
        stock_id = env.idx_to_stock.get(i)

        if stock_id is None:
            mask.append(False)
        elif action_type == TRANSFER_STOCK:
            mask.append(
                any(
                    _can_transfer_stock(env, stock_id, to_id)
                    for to_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
                )
            )
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_amount_mask(env, action_type, device, corp_a_idx=None, corp_b_idx=None):
    max_amounts = max(len(env.cash_amounts), len(env.stock_percents))
    mask = []
    corp_a_id = None if corp_a_idx is None else env.idx_to_corporation.get(corp_a_idx)
    corp_b_id = None if corp_b_idx is None else env.idx_to_corporation.get(corp_b_idx)

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
            elif corp_b_id is None:
                amount = env.cash_amounts[i]
                mask.append(
                    any(
                        _can_transfer_cash(env, corp_a_id, to_id, amount)
                        for to_id in _indexed_values(
                            env.idx_to_corporation,
                            env.max_corporations,
                        )
                    )
                )
            else:
                mask.append(
                    _can_transfer_cash(env, corp_a_id, corp_b_id, env.cash_amounts[i])
                )
        elif action_type in {TRANSFER_STOCK, ISSUE_STOCK}:
            mask.append(i < len(env.stock_percents))
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

    if (
        env.state.corporations[from_id].is_domestic
        and env.state.corporations[to_id].is_domestic
    ):
        if not env.state.is_subcorporation_of(from_id, to_id):
            return False
        if (
            to_id == env.state.taxpayer_id
            and _transfer_uses_foreign_contributed_cash(env, from_id, amount)
            and not env.state.has_qualifying_zero_basis_cfc_stock(from_id)
        ):
            return False

    from_foreign = env.state.corporations[from_id].is_foreign
    to_domestic = env.state.corporations[to_id].is_domestic
    direct_to_taxpayer = (
        from_foreign
        and to_domestic
        and to_id == env.state.taxpayer_id
    )
    if direct_to_taxpayer:
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


def _transfer_uses_foreign_contributed_cash(env, from_id, amount):
    for cash in env.state.cash.values():
        if cash.owner_id != from_id or cash.amount < amount:
            continue
        contributor_id = cash.contributed_by_id
        if contributor_id is None:
            return False
        return env.state.corporations[contributor_id].is_foreign

    return False


def _can_transfer_stock(env, stock_id, to_id):
    if stock_id not in env.state.stock or to_id not in env.state.corporations:
        return False

    stock = env.state.stock[stock_id]
    return stock.percent > 0.0 and stock.holder_id != to_id


def _can_issue_stock(env, issuer_id, recipient_id):
    if issuer_id == recipient_id:
        return False
    if issuer_id not in env.state.corporations or recipient_id not in env.state.corporations:
        return False
    if not env.state.has_contributed_property(issuer_id, recipient_id):
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

    if action_type_int in {MAKE_SUBCORPORATION, TRANSFER_CASH, ISSUE_STOCK}:
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

    if action_type_int == TRANSFER_STOCK:
        stock_mask = make_stock_mask(env, action_type_int, device)
        stock, stock_log_prob, stock_entropy = sample_from_logits_or_uniform(
            out["stock_logits"],
            stock_mask,
            force_uniform,
        )

        log_prob = log_prob + stock_log_prob
        entropy = entropy + stock_entropy
        action["stock"] = stock
        used_masks["stock"] = stock_mask

    if action_type_int in {TRANSFER_CASH, ISSUE_STOCK}:
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

    if action_type_int == TRANSFER_STOCK:
        corp_b_mask = make_corp_b_mask(
            env,
            action_type_int,
            device,
            stock_idx=action["stock"].item(),
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

    if action_type_int in {TRANSFER_CASH, TRANSFER_STOCK, ISSUE_STOCK}:
        corp_a_idx = None
        if action["corp_a"] is not None:
            corp_a_idx = action["corp_a"].item()
        corp_b_idx = None
        if action["corp_b"] is not None:
            corp_b_idx = action["corp_b"].item()

        amount_mask = make_amount_mask(
            env,
            action_type_int,
            device,
            corp_a_idx=corp_a_idx,
            corp_b_idx=corp_b_idx,
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

    if action_type == TRANSFER_STOCK:
        stock_id = env.idx_to_stock.get(stock_idx, f"idx_{stock_idx}")
        to_id = env.idx_to_corporation.get(corp_b_idx, f"idx_{corp_b_idx}")
        percent = env.stock_percents[amount_idx]
        return f"{action_name}(stock={stock_id}, to={to_id}, percent={percent})"

    if action_type == ISSUE_STOCK:
        issuer_id = env.idx_to_corporation.get(corp_a_idx, f"idx_{corp_a_idx}")
        recipient_id = env.idx_to_corporation.get(corp_b_idx, f"idx_{corp_b_idx}")
        percent = env.stock_percents[amount_idx]
        return f"{action_name}(issuer={issuer_id}, recipient={recipient_id}, percent={percent})"

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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TaxEnv(
        MAX_CORPORATIONS=5,
        MAX_STOCKS=8,
        MAX_STEPS=10,
    )

    embed_dim = 128
    num_action_types = 4
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

    total_updates = 1000
    rollout_steps = 256
    snapshot_dir = Path("barnes_group/rl_snapshots")
    max_snapshots = 15
    saved_snapshot_count = 0

    for update in range(total_updates):
        obs = reset_training_env(env, device)

        buffer.clear()
        cumulative_reward = 0.0
        reward_by_action = {i: [] for i in range(num_action_types)}
        invalid_count = 0
        positive_reward_count = 0
        success_count = 0
        random_exploration_count = 0
        entropy_total = 0.0
        best_tax_advantage = 0.0
        best_snapshot = None
        current_episode_actions = []

        random_action_prob = max(0.02, 0.25 * (0.995 ** update))

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

            if info.get("tax_advantage", 0.0) > best_tax_advantage:
                best_tax_advantage = info["tax_advantage"]
                best_snapshot = {
                    "state": copy.deepcopy(env.state),
                    "tax_advantage": info["tax_advantage"],
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
            and best_snapshot["tax_advantage"] > 0.0
            and update % 25 == 0
            and saved_snapshot_count < max_snapshots
        ):
            save_best_snapshot(
                env=env,
                snapshot=best_snapshot,
                output_dir=snapshot_dir,
                update=update,
                snapshot_count=saved_snapshot_count,
            )
            saved_snapshot_count += 1

        if update % 25 == 0:
            print(
                "update: "
                f"{update}; loss: {loss}; "
                f"cumulative_reward: {cumulative_reward}; "
                f"invalid_count: {invalid_count}; "
                f"positive_reward_count: {positive_reward_count}; "
                f"success_count: {success_count}; "
                f"random_exploration_count: {random_exploration_count}; "
                f"avg_entropy: {entropy_total / rollout_steps:.4f}; "
                f"best_tax_advantage: {best_tax_advantage}; "
                f"random_action_prob: {random_action_prob:.4f}"
            )
            for action_type, rewards in reward_by_action.items():
                if rewards:
                    print(
                        ACTION_NAMES[action_type],
                        sum(rewards) / len(rewards),
                        len(rewards),
                    )


if __name__ == "__main__":
    train()
