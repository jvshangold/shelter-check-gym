import copy
from datetime import datetime
from pathlib import Path

import torch

from subsidiary_stock_comp.rl.model import PolicyValueNet
from subsidiary_stock_comp.rl.ppo import masked_categorical, ppo_update
from subsidiary_stock_comp.rl.rollout import RolloutBuffer
from subsidiary_stock_comp.tax_env.env import (
    BUY_STOCK,
    COMPENSATE_EMPLOYEES,
    FORM_SUBCORP,
    LIQUIDATE_CORP,
    TaxEnv,
)


ACTION_NAMES = {
    FORM_SUBCORP: "form_subcorp",
    BUY_STOCK: "buy_stock",
    COMPENSATE_EMPLOYEES: "compensate_employees",
    LIQUIDATE_CORP: "liquidate_corp",
}


def make_action_mask(env, device):
    mask = [False, False, False, False]

    mask[FORM_SUBCORP] = any(
        _can_form_subcorp(env, contributor_a_id, contributor_b_id, split_idx)
        for contributor_a_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
        for contributor_b_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
        for split_idx in range(len(env.formation_splits))
    )
    mask[BUY_STOCK] = any(
        _can_buy_stock(env, buyer_id, issuer_id, amount_idx)
        for buyer_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
        for issuer_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
        for amount_idx in range(len(env.cash_amounts))
    )
    mask[COMPENSATE_EMPLOYEES] = any(
        _can_compensate_employees(env, holder_id, service_recipient_id, stock_id, amount_idx)
        for holder_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
        for service_recipient_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
        for stock_id in _indexed_values(env.idx_to_stock, env.max_stocks)
        for amount_idx in range(len(env.cash_amounts))
    )
    mask[LIQUIDATE_CORP] = any(
        _can_liquidate_corp(env, corporation_id)
        for corporation_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
    )

    return torch.tensor(mask, dtype=torch.bool, device=device)


def _indexed_values(index, max_count):
    return [index[i] for i in range(max_count) if i in index]


def make_corp_a_mask(env, action_type, device):
    mask = []

    for i in range(env.max_corporations):
        corporation_id = env.idx_to_corporation.get(i)

        if corporation_id is None:
            mask.append(False)
        elif action_type == FORM_SUBCORP:
            mask.append(
                any(
                    _can_form_subcorp(env, corporation_id, other_id, split_idx)
                    for other_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
                    for split_idx in range(len(env.formation_splits))
                )
            )
        elif action_type == BUY_STOCK:
            mask.append(
                any(
                    _can_buy_stock(env, corporation_id, issuer_id, amount_idx)
                    for issuer_id in _indexed_values(env.idx_to_corporation, env.max_corporations)
                    for amount_idx in range(len(env.cash_amounts))
                )
            )
        elif action_type == COMPENSATE_EMPLOYEES:
            mask.append(
                any(
                    _can_compensate_employees(
                        env,
                        corporation_id,
                        service_recipient_id,
                        stock_id,
                        amount_idx,
                    )
                    for service_recipient_id in _indexed_values(
                        env.idx_to_corporation,
                        env.max_corporations,
                    )
                    for stock_id in _indexed_values(env.idx_to_stock, env.max_stocks)
                    for amount_idx in range(len(env.cash_amounts))
                )
            )
        elif action_type == LIQUIDATE_CORP:
            mask.append(_can_liquidate_corp(env, corporation_id))
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_corp_b_mask(env, action_type, corp_a_idx, stock_idx, device):
    corp_a_id = env.idx_to_corporation.get(corp_a_idx)
    stock_id = env.idx_to_stock.get(stock_idx)
    mask = []

    for i in range(env.max_corporations):
        corporation_id = env.idx_to_corporation.get(i)

        if corporation_id is None or corp_a_id is None:
            mask.append(False)
        elif action_type == FORM_SUBCORP:
            mask.append(
                any(
                    _can_form_subcorp(env, corp_a_id, corporation_id, split_idx)
                    for split_idx in range(len(env.formation_splits))
                )
            )
        elif action_type == BUY_STOCK:
            mask.append(
                any(
                    _can_buy_stock(env, corp_a_id, corporation_id, amount_idx)
                    for amount_idx in range(len(env.cash_amounts))
                )
            )
        elif action_type == COMPENSATE_EMPLOYEES:
            if stock_id is None:
                mask.append(
                    any(
                        _can_compensate_employees(
                            env,
                            corp_a_id,
                            corporation_id,
                            candidate_stock_id,
                            amount_idx,
                        )
                        for candidate_stock_id in _indexed_values(
                            env.idx_to_stock,
                            env.max_stocks,
                        )
                        for amount_idx in range(len(env.cash_amounts))
                    )
                )
            else:
                mask.append(
                    any(
                        _can_compensate_employees(
                            env,
                            corp_a_id,
                            corporation_id,
                            stock_id,
                            amount_idx,
                        )
                        for amount_idx in range(len(env.cash_amounts))
                    )
                )
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_stock_mask(env, action_type, holder_idx, service_recipient_idx, device):
    holder_id = env.idx_to_corporation.get(holder_idx)
    service_recipient_id = env.idx_to_corporation.get(service_recipient_idx)
    mask = []

    for i in range(env.max_stocks):
        stock_id = env.idx_to_stock.get(i)

        if stock_id is None or action_type != COMPENSATE_EMPLOYEES:
            mask.append(False)
        else:
            mask.append(
                any(
                    _can_compensate_employees(
                        env,
                        holder_id,
                        service_recipient_id,
                        stock_id,
                        amount_idx,
                    )
                    for amount_idx in range(len(env.cash_amounts))
                )
            )

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_amount_mask(env, action_type, corp_a_idx, corp_b_idx, stock_idx, device):
    corp_a_id = env.idx_to_corporation.get(corp_a_idx)
    corp_b_id = env.idx_to_corporation.get(corp_b_idx)
    stock_id = env.idx_to_stock.get(stock_idx)
    mask = []

    for i in range(max(len(env.cash_amounts), len(env.formation_splits))):
        if action_type == FORM_SUBCORP:
            mask.append(
                i < len(env.formation_splits)
                and _can_form_subcorp(env, corp_a_id, corp_b_id, i)
            )
        elif action_type == BUY_STOCK:
            mask.append(
                i < len(env.cash_amounts)
                and _can_buy_stock(env, corp_a_id, corp_b_id, i)
            )
        elif action_type == COMPENSATE_EMPLOYEES:
            mask.append(
                i < len(env.cash_amounts)
                and _can_compensate_employees(
                    env,
                    corp_a_id,
                    corp_b_id,
                    stock_id,
                    i,
                )
            )
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def _can_form_subcorp(env, contributor_a_id, contributor_b_id, split_idx):
    if contributor_a_id is None or contributor_b_id is None:
        return False
    if len(env.state.corporations) >= env.max_corporations:
        return False
    if contributor_a_id == contributor_b_id:
        return False
    if split_idx >= len(env.formation_splits):
        return False
    contribution_a, contribution_b = env.formation_splits[split_idx]
    return (
        env.state.cash_amount(contributor_a_id) >= contribution_a
        and env.state.cash_amount(contributor_b_id) >= contribution_b
    )


def _can_buy_stock(env, buyer_id, issuer_id, amount_idx):
    if buyer_id is None or issuer_id is None:
        return False
    if issuer_id != env.state.taxpayer_id:
        return False
    if buyer_id == issuer_id:
        return False
    if _outside_shareholder_count(env, buyer_id) < 2:
        return False
    if len(env.state.stock) >= env.max_stocks:
        return False
    if amount_idx >= len(env.cash_amounts):
        return False
    return env.state.cash_amount(buyer_id) == env.cash_amounts[amount_idx]


def _can_compensate_employees(
    env,
    holder_id,
    service_recipient_id,
    stock_id,
    amount_idx,
):
    if holder_id is None or service_recipient_id is None or stock_id is None:
        return False
    if amount_idx >= len(env.cash_amounts):
        return False
    if stock_id not in env.state.stock:
        return False
    if service_recipient_id != env.state.taxpayer_id:
        return False
    if _outside_shareholder_count(env, holder_id) < 2:
        return False

    stock = env.state.stock[stock_id]
    amount = env.cash_amounts[amount_idx]
    return (
        stock.holder_id == holder_id
        and stock.issuer_id == service_recipient_id
        and amount < stock.fmv
    )


def _can_liquidate_corp(env, corporation_id):
    if corporation_id is None:
        return False
    if corporation_id in {env.state.taxpayer_id, env.state.subsidiary_id}:
        return False
    return any(
        stock.issuer_id == corporation_id and stock.holder_id != corporation_id
        for stock in env.state.stock.values()
    )


def _outside_shareholder_count(env, corporation_id):
    return len({
        stock.holder_id
        for stock in env.state.stock.values()
        if stock.issuer_id == corporation_id and stock.holder_id != corporation_id
    })


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

    if action_type_int in {FORM_SUBCORP, BUY_STOCK, COMPENSATE_EMPLOYEES, LIQUIDATE_CORP}:
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

    if action_type_int in {FORM_SUBCORP, BUY_STOCK}:
        corp_b_mask = make_corp_b_mask(
            env,
            action_type_int,
            corp_a_idx=corp_a.item(),
            stock_idx=0,
            device=device,
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

    if action_type_int == COMPENSATE_EMPLOYEES:
        stock_mask = make_stock_mask(
            env,
            action_type_int,
            holder_idx=corp_a.item(),
            service_recipient_idx=0,
            device=device,
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

        corp_b_mask = make_corp_b_mask(
            env,
            action_type_int,
            corp_a_idx=corp_a.item(),
            stock_idx=stock.item(),
            device=device,
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

    if action_type_int in {FORM_SUBCORP, BUY_STOCK, COMPENSATE_EMPLOYEES}:
        amount_mask = make_amount_mask(
            env,
            action_type_int,
            corp_a_idx=corp_a.item(),
            corp_b_idx=action["corp_b"].item(),
            stock_idx=action["stock"].item() if action["stock"] is not None else 0,
            device=device,
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


def has_available_action(env, device):
    return bool(make_action_mask(env, device).any().item())


def describe_env_action(env, env_action):
    action_type, corp_a_idx, corp_b_idx, stock_idx, amount_idx = env_action
    action_name = ACTION_NAMES.get(action_type, f"unknown_{action_type}")

    if action_type == FORM_SUBCORP:
        contributor_a = env.idx_to_corporation.get(corp_a_idx, f"idx_{corp_a_idx}")
        contributor_b = env.idx_to_corporation.get(corp_b_idx, f"idx_{corp_b_idx}")
        contribution_a, contribution_b = env.formation_splits[amount_idx]
        return (
            f"{action_name}("
            f"{contributor_a}={contribution_a}, {contributor_b}={contribution_b})"
        )

    if action_type == BUY_STOCK:
        buyer = env.idx_to_corporation.get(corp_a_idx, f"idx_{corp_a_idx}")
        issuer = env.idx_to_corporation.get(corp_b_idx, f"idx_{corp_b_idx}")
        amount = env.cash_amounts[amount_idx]
        return f"{action_name}(buyer={buyer}, issuer={issuer}, amount={amount})"

    if action_type == COMPENSATE_EMPLOYEES:
        holder = env.idx_to_corporation.get(corp_a_idx, f"idx_{corp_a_idx}")
        service_recipient = env.idx_to_corporation.get(corp_b_idx, f"idx_{corp_b_idx}")
        stock_id = env.idx_to_stock.get(stock_idx, f"idx_{stock_idx}")
        amount = env.cash_amounts[amount_idx]
        return (
            f"{action_name}("
            f"holder={holder}, service_recipient={service_recipient}, "
            f"stock={stock_id}, amount={amount})"
        )

    if action_type == LIQUIDATE_CORP:
        corporation = env.idx_to_corporation.get(corp_a_idx, f"idx_{corp_a_idx}")
        return f"{action_name}(corporation={corporation})"

    return str(env_action)


def save_best_snapshot(snapshot, output_dir, update, snapshot_count):
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / (
        f"best_structure_{snapshot_count:04d}_update_{update:04d}.txt"
    )

    with metadata_path.open("w") as metadata:
        metadata.write(f"update: {update}\n")
        metadata.write(f"snapshot_index: {snapshot_count}\n")
        metadata.write(f"tax_advantage: {snapshot['tax_advantage']}\n")
        metadata.write(f"ordinary_deductions: {snapshot['ordinary_deductions']}\n")
        metadata.write(f"capital_losses: {snapshot['capital_losses']}\n")
        metadata.write(f"terminated: {snapshot['terminated']}\n")
        metadata.write(f"truncated: {snapshot['truncated']}\n")
        metadata.write(f"invalid_action: {snapshot['invalid_action']}\n")
        metadata.write("\nactions:\n")
        for i, action_description in enumerate(snapshot["actions"], start=1):
            metadata.write(f"{i}. {action_description}\n")


def train(total_updates=1000, rollout_steps=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TaxEnv(
        MAX_CORPORATIONS=5,
        MAX_STOCKS=8,
        MAX_STEPS=8,
    )

    embed_dim = 128
    num_action_types = 4

    model = PolicyValueNet(
        embed_dim=embed_dim,
        num_action_types=num_action_types,
        max_corporations=env.max_corporations,
        max_stocks=env.max_stocks,
        num_amounts=max(len(env.cash_amounts), len(env.formation_splits)),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    buffer = RolloutBuffer()
    dead_end_penalty = 0.25

    package_root = Path(__file__).resolve().parents[1]
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    snapshot_dir = package_root / "rl_snapshots" / run_id
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
        dead_end_count = 0
        entropy_total = 0.0
        best_tax_advantage = 0.0
        best_snapshot = None
        current_episode_actions = []
        random_action_prob = max(0.02, 0.25 * (0.995 ** update))

        for _ in range(rollout_steps):
            if not has_available_action(env, device):
                dead_end_count += 1
                obs = reset_training_env(env, device)
                current_episode_actions = []

            with torch.no_grad():
                action, log_prob, entropy, value, masks, used_random_exploration = sample_action(
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

            if not done and not has_available_action(env, device):
                reward -= dead_end_penalty
                done = True
                dead_end_count += 1
                info["dead_end"] = True

            reward_by_action[env_action[0]].append(reward)
            invalid_count += int(info.get("invalid_action", False))
            positive_reward_count += int(reward > 0.0)
            success_count += int(terminated)
            random_exploration_count += int(used_random_exploration)
            entropy_total += float(entropy.detach().cpu().item())

            tax_advantage = float(info.get("tax_advantage", 0.0))
            if tax_advantage >= best_tax_advantage or best_snapshot is None:
                best_tax_advantage = tax_advantage
                best_snapshot = {
                    "state": copy.deepcopy(env.state),
                    "tax_advantage": tax_advantage,
                    "ordinary_deductions": info["ordinary_deductions"],
                    "capital_losses": info["capital_losses"],
                    "terminated": terminated,
                    "truncated": truncated,
                    "invalid_action": bool(info.get("invalid_action", False)),
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

        loss = ppo_update(
            model=model,
            optimizer=optimizer,
            rollout=buffer,
        )

        if update % 25 == 0:
            if best_snapshot is not None:
                save_best_snapshot(
                    snapshot=best_snapshot,
                    output_dir=snapshot_dir,
                    update=update,
                    snapshot_count=saved_snapshot_count,
                )
                saved_snapshot_count += 1

            print(
                "update: "
                f"{update}; loss: {loss}; "
                f"cumulative_reward: {cumulative_reward}; "
                f"invalid_count: {invalid_count}; "
                f"positive_reward_count: {positive_reward_count}; "
                f"success_count: {success_count}; "
                f"dead_end_count: {dead_end_count}; "
                f"best_tax_advantage: {best_tax_advantage}; "
                f"avg_entropy: {entropy_total / rollout_steps}; "
                f"random_action_prob: {random_action_prob:.4f}; "
                f"random_exploration_count: {random_exploration_count}; "
                f"saved_snapshots: {saved_snapshot_count}"
            )
            for action_type, rewards in reward_by_action.items():
                if rewards:
                    print(action_type, sum(rewards) / len(rewards), len(rewards))


if __name__ == "__main__":
    train()
