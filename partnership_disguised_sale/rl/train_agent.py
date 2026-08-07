import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partnership_disguised_sale.rl.model import PolicyValueNet
from partnership_disguised_sale.rl.ppo import masked_categorical, ppo_update
from partnership_disguised_sale.rl.rollout import RolloutBuffer
from partnership_disguised_sale.tax_env.env import (
    CONTRIBUTE_ASSET,
    CONTRIBUTE_CASH,
    DISTRIBUTE_CASH,
    TAKE_OUT_LOAN,
    TaxEnv,
)
from partnership_disguised_sale.tax_env.hard_env import HardTaxEnv
from partnership_disguised_sale.tax_env.state import OwnerType


ACTION_NAMES = {
    TAKE_OUT_LOAN: "take_out_loan",
    CONTRIBUTE_ASSET: "contribute_asset",
    DISTRIBUTE_CASH: "distribute_cash",
    CONTRIBUTE_CASH: "contribute_cash",
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


def resolve_env_class(env_variant=None):
    variant = (
        env_variant
        or os.environ.get("SHELTER_CHECK_ENV_VARIANT")
        or os.environ.get("ENV_VARIANT")
        or "easy"
    ).strip().lower()

    if variant in {"easy", "default", "scaffolded"}:
        return TaxEnv
    if variant in {"hard", "unscaffolded"}:
        return HardTaxEnv
    raise ValueError(f"Unknown env variant: {env_variant}")


def make_action_mask(env, device):
    mask = [False, False, False, False]

    mask[TAKE_OUT_LOAN] = len(env.state.loans) < env.max_loans
    mask[CONTRIBUTE_ASSET] = any(
        _can_contribute_asset(env, individual_id, asset_id)
        for individual_id in _indexed_values(env.idx_to_individual, env.max_individuals)
        for asset_id in _indexed_values(env.idx_to_asset, env.max_assets)
    )
    mask[DISTRIBUTE_CASH] = any(
        env.state.partnership_cash(env.state.partnership_id) >= amount
        for amount in env.amount_buckets
    )
    mask[CONTRIBUTE_CASH] = any(
        env.state.individual_cash(individual_id) >= amount
        for individual_id in _indexed_values(env.idx_to_individual, env.max_individuals)
        for amount in env.amount_buckets
    )

    return torch.tensor(mask, dtype=torch.bool, device=device)


def _indexed_values(index, max_count):
    return [index[i] for i in range(max_count) if i in index]


def make_individual_mask(env, action_type, device):
    mask = []

    for i in range(env.max_individuals):
        individual_id = env.idx_to_individual.get(i)

        if individual_id is None:
            mask.append(False)
        elif action_type == TAKE_OUT_LOAN:
            mask.append(_is_partner(env, individual_id))
        elif action_type == CONTRIBUTE_ASSET:
            mask.append(
                any(
                    _can_contribute_asset(env, individual_id, asset_id)
                    for asset_id in _indexed_values(env.idx_to_asset, env.max_assets)
                )
            )
        elif action_type == DISTRIBUTE_CASH:
            mask.append(_is_partner(env, individual_id))
        elif action_type == CONTRIBUTE_CASH:
            mask.append(
                any(
                    env.state.individual_cash(individual_id) >= amount
                    for amount in env.amount_buckets
                )
            )
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_asset_mask(env, action_type, individual_idx, device):
    individual_id = env.idx_to_individual.get(individual_idx)
    mask = []

    for i in range(env.max_assets):
        asset_id = env.idx_to_asset.get(i)

        if asset_id is None or individual_id is None:
            mask.append(False)
        elif action_type == CONTRIBUTE_ASSET:
            mask.append(_can_contribute_asset(env, individual_id, asset_id))
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_amount_mask(env, action_type, individual_idx, device):
    individual_id = env.idx_to_individual.get(individual_idx)
    mask = []

    for amount in env.amount_buckets:
        if action_type == TAKE_OUT_LOAN:
            mask.append(len(env.state.loans) < env.max_loans)
        elif action_type == DISTRIBUTE_CASH:
            mask.append(env.state.partnership_cash(env.state.partnership_id) >= amount)
        elif action_type == CONTRIBUTE_CASH and individual_id is not None:
            mask.append(env.state.individual_cash(individual_id) >= amount)
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool, device=device)


def _is_partner(env, individual_id):
    return individual_id in env.state.partnerships[env.state.partnership_id].partner_ids


def _can_contribute_asset(env, individual_id, asset_id):
    if not _is_partner(env, individual_id):
        return False

    asset = env.state.assets[asset_id]
    return (
        asset.owner_type == OwnerType.INDIVIDUAL
        and asset.owner_id == individual_id
        and not asset.is_contributed_to_partnership
    )


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
    action_type_int = action_type.item()

    log_prob = log_prob + action_log_prob
    entropy = entropy + action_entropy

    action = {
        "action_type": action_type,
        "individual": None,
        "asset": None,
        "amount": None,
    }
    used_masks = {"action": action_mask}

    if action_type_int in {
        TAKE_OUT_LOAN,
        CONTRIBUTE_ASSET,
        DISTRIBUTE_CASH,
        CONTRIBUTE_CASH,
    }:
        individual_mask = make_individual_mask(env, action_type_int, device)
        individual, individual_log_prob, individual_entropy = sample_from_logits_or_uniform(
            out["individual_logits"],
            individual_mask,
            force_uniform,
        )

        log_prob = log_prob + individual_log_prob
        entropy = entropy + individual_entropy
        action["individual"] = individual
        used_masks["individual"] = individual_mask

    if action_type_int == CONTRIBUTE_ASSET:
        asset_mask = make_asset_mask(
            env,
            action_type_int,
            individual_idx=individual.item(),
            device=device,
        )
        asset, asset_log_prob, asset_entropy = sample_from_logits_or_uniform(
            out["asset_logits"],
            asset_mask,
            force_uniform,
        )

        log_prob = log_prob + asset_log_prob
        entropy = entropy + asset_entropy
        action["asset"] = asset
        used_masks["asset"] = asset_mask

    if action_type_int in {TAKE_OUT_LOAN, DISTRIBUTE_CASH, CONTRIBUTE_CASH}:
        amount_mask = make_amount_mask(
            env,
            action_type_int,
            individual_idx=individual.item(),
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

    return action, log_prob, entropy, out["value"], used_masks, force_uniform


def action_dict_to_env_action(action):
    return (
        int(action["action_type"].item()),
        int(action["individual"].item()) if action["individual"] is not None else 0,
        int(action["asset"].item()) if action["asset"] is not None else 0,
        0,
        int(action["amount"].item()) if action["amount"] is not None else 0,
    )


def reset_training_env(env, device):
    obs, _ = env.reset()
    return obs.to(device)


def has_valid_action(env, device):
    return bool(make_action_mask(env, device).any().item())


def describe_env_action(env, env_action):
    action_type, individual_idx, asset_idx, loan_idx, amount_idx = env_action
    action_name = ACTION_NAMES.get(action_type, f"unknown_{action_type}")

    if action_type == TAKE_OUT_LOAN:
        guarantor_id = env.idx_to_individual.get(individual_idx, f"idx_{individual_idx}")
        amount = env.amount_buckets[amount_idx]
        return f"{action_name}(guarantor={guarantor_id}, amount={amount})"

    if action_type == CONTRIBUTE_ASSET:
        partner_id = env.idx_to_individual.get(individual_idx, f"idx_{individual_idx}")
        asset_id = env.idx_to_asset.get(asset_idx, f"idx_{asset_idx}")
        return f"{action_name}(partner={partner_id}, asset={asset_id})"

    if action_type == DISTRIBUTE_CASH:
        recipient_id = env.idx_to_individual.get(individual_idx, f"idx_{individual_idx}")
        amount = env.amount_buckets[amount_idx]
        return f"{action_name}(recipient={recipient_id}, amount={amount})"

    if action_type == CONTRIBUTE_CASH:
        partner_id = env.idx_to_individual.get(individual_idx, f"idx_{individual_idx}")
        amount = env.amount_buckets[amount_idx]
        return f"{action_name}(partner={partner_id}, amount={amount})"

    return str(env_action)


def save_best_snapshot(env, snapshot, output_dir, update, snapshot_count):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stem = output_dir / (
        f"best_structure_{snapshot_count:04d}_update_{update:04d}"
    )
    old_state = env.state

    try:
        env.state = snapshot["state"]
        env.render_world(filename=str(image_stem))
    finally:
        env.state = old_state

    metadata_path = image_stem.with_suffix(".txt")
    tax = snapshot["tax_computation"]

    with metadata_path.open("w") as metadata:
        metadata.write(f"update: {update}\n")
        metadata.write(f"snapshot_index: {snapshot_count}\n")
        metadata.write(f"tax_advantage: {snapshot['tax_advantage']}\n")
        metadata.write(f"recognized_gain: {tax.recognized_gain}\n")
        metadata.write(f"baseline_gain: {tax.baseline_gain}\n")
        metadata.write(f"deferred_gain: {tax.deferred_gain}\n")
        metadata.write(f"raw_tax_savings: {tax.tax_savings}\n")
        metadata.write(f"economic_sale_complete: {snapshot['economic_sale_complete']}\n")
        metadata.write(
            "non_transferor_positive_capital: "
            f"{snapshot['non_transferor_positive_capital']}\n"
        )
        metadata.write(f"terminated: {snapshot['terminated']}\n")
        metadata.write(f"truncated: {snapshot['truncated']}\n")
        metadata.write(f"invalid_action: {snapshot['invalid_action']}\n")
        metadata.write("\nactions:\n")
        for i, action_description in enumerate(snapshot["actions"], start=1):
            metadata.write(f"{i}. {action_description}\n")


def train(
    total_updates=500,
    rollout_steps=256,
    save_snapshots=True,
    log_interval=25,
    env_variant=None,
):
    device = resolve_device()
    env_class = resolve_env_class(env_variant)

    env = env_class(
        MAX_INDIVIDUALS=4,
        MAX_ASSETS=4,
        MAX_LOANS=4,
        MAX_STEPS=8,
    )
    print(f"env_variant: {env_class.__name__}", flush=True)

    embed_dim = 128
    num_action_types = 4

    model = PolicyValueNet(
        embed_dim=embed_dim,
        num_action_types=num_action_types,
        max_individuals=env.max_individuals,
        max_assets=env.max_assets,
        num_amounts=len(env.amount_buckets),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    buffer = RolloutBuffer()

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
        episode_count = 0
        random_exploration_count = 0
        entropy_total = 0.0
        best_tax_advantage = 0.0
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
                    "tax_computation": info["tax_computation"],
                    "economic_sale_complete": info["economic_sale_complete"],
                    "non_transferor_positive_capital": (
                        info["non_transferor_positive_capital"]
                    ),
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
                episode_count += 1
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

        if update % log_interval == 0:
            if best_snapshot is not None and save_snapshots:
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
                    f"{(snapshot_dir / f'best_structure_{saved_snapshot_count - 1:04d}_update_{update:04d}.png').resolve()}",
                    flush=True,
                )

            print(
                "update_complete: "
                f"{update}; loss: {loss}; "
                f"cumulative_reward: {cumulative_reward}; "
                f"invalid_count: {invalid_count}; "
                f"positive_reward_count: {positive_reward_count}; "
                f"success_count: {success_count}; "
                f"episode_count: {episode_count}; "
                f"best_tax_advantage: {best_tax_advantage}; "
                f"avg_entropy: {entropy_total / rollout_steps}; "
                f"random_exploration_count: {random_exploration_count}; "
                f"saved_snapshots: {saved_snapshot_count}; "
                f"random_action_prob: {random_action_prob:.4f}",
                flush=True,
            )
            for action_type, rewards in reward_by_action.items():
                if rewards:
                    print(
                        ACTION_NAMES.get(action_type, action_type),
                        sum(rewards) / len(rewards),
                        len(rewards),
                        flush=True,
                    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument(
        "--env-variant",
        choices=("easy", "hard"),
        default=None,
        help="Use the scaffolded easy env or the unscaffolded hard env.",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Disable best-structure snapshot rendering.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        total_updates=args.total_updates,
        rollout_steps=args.rollout_steps,
        save_snapshots=not args.no_snapshots,
        log_interval=args.log_interval,
        env_variant=args.env_variant,
    )
