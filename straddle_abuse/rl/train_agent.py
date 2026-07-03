import copy
import os
from pathlib import Path

import torch

from straddle_abuse.rl.model import PolicyValueNet
from straddle_abuse.rl.ppo import masked_categorical, ppo_update
from straddle_abuse.rl.rollout import RolloutBuffer
from straddle_abuse.tax_env.env import (
    ENTER_STRADDLE,
    INVEST,
    REALIZE_GAIN,
    REALIZE_LOSS,
    TaxEnv,
)
from straddle_abuse.tax_env.state import StraddleLegKind


ACTION_NAMES = {
    ENTER_STRADDLE: "enter_straddle",
    REALIZE_GAIN: "realize_gain",
    REALIZE_LOSS: "realize_loss",
    INVEST: "invest",
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
    return torch.tensor(env.get_action_mask(), dtype=torch.bool, device=device)


def make_straddle_mask(env, action_type, device):
    if action_type == REALIZE_GAIN:
        mask = env.get_straddle_mask(StraddleLegKind.GAIN)
    elif action_type == REALIZE_LOSS:
        mask = env.get_straddle_mask(StraddleLegKind.LOSS)
    else:
        mask = [False for _ in range(env.max_straddles)]

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_fraction_mask(env, action_type, device):
    if action_type in {ENTER_STRADDLE, REALIZE_GAIN, REALIZE_LOSS, INVEST}:
        mask = env.get_fraction_mask()
    else:
        mask = [False for _ in env.fractions]

    return torch.tensor(mask, dtype=torch.bool, device=device)


def make_individual_mask(env, action_type, device):
    if action_type == INVEST:
        mask = env.get_individual_mask()
    else:
        mask = [False for _ in env.individual_ids]

    return torch.tensor(mask, dtype=torch.bool, device=device)


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
        "straddle": None,
        "fraction": None,
        "individual": None,
    }
    used_masks = {"action": action_mask}

    fraction_mask = make_fraction_mask(env, action_type_int, device)
    fraction, fraction_log_prob, fraction_entropy = sample_from_logits_or_uniform(
        out["fraction_logits"],
        fraction_mask,
        force_uniform,
    )

    log_prob = log_prob + fraction_log_prob
    entropy = entropy + fraction_entropy
    action["fraction"] = fraction
    used_masks["fraction"] = fraction_mask

    if action_type_int in {REALIZE_GAIN, REALIZE_LOSS}:
        straddle_mask = make_straddle_mask(env, action_type_int, device)
        straddle, straddle_log_prob, straddle_entropy = sample_from_logits_or_uniform(
            out["straddle_logits"],
            straddle_mask,
            force_uniform,
        )

        log_prob = log_prob + straddle_log_prob
        entropy = entropy + straddle_entropy
        action["straddle"] = straddle
        used_masks["straddle"] = straddle_mask

    if action_type_int == INVEST:
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

    return action, log_prob, entropy, out["value"], used_masks, force_uniform


def action_dict_to_env_action(action):
    return (
        int(action["action_type"].item()),
        int(action["straddle"].item()) if action["straddle"] is not None else 0,
        int(action["fraction"].item()) if action["fraction"] is not None else 0,
        int(action["individual"].item()) if action["individual"] is not None else 0,
    )


def reset_training_env(env, device):
    obs, _ = env.reset()
    return obs.to(device)


def has_valid_action(env, device) -> bool:
    return bool(make_action_mask(env, device).any().item())


def describe_env_action(env, env_action):
    action_type, straddle_idx, fraction_idx, individual_idx = env_action
    action_name = ACTION_NAMES.get(action_type, f"unknown_{action_type}")

    if action_type == ENTER_STRADDLE:
        amount = env.straddle_amounts[fraction_idx]
        return f"{action_name}(amount={amount})"

    if action_type in {REALIZE_GAIN, REALIZE_LOSS}:
        straddle_id = env.idx_to_straddle.get(straddle_idx, f"idx_{straddle_idx}")
        fraction = env.fractions[fraction_idx]
        return f"{action_name}(straddle_id={straddle_id}, fraction={fraction})"

    if action_type == INVEST:
        individual_id = env.idx_to_individual.get(individual_idx, f"idx_{individual_idx}")
        fraction = env.fractions[fraction_idx]
        return f"{action_name}(individual={individual_id}, fraction={fraction})"

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
        metadata.write(f"tax_reduction: {snapshot['tax_reduction']}\n")
        metadata.write(f"terminated: {snapshot['terminated']}\n")
        metadata.write(f"truncated: {snapshot['truncated']}\n")
        metadata.write(f"invalid_action: {snapshot['invalid_action']}\n")
        metadata.write("\nactions:\n")
        for i, action_description in enumerate(snapshot["actions"], start=1):
            metadata.write(f"{i}. {action_description}\n")


def train(total_updates=500, rollout_steps=256, save_snapshots=True, log_interval=25):
    device = resolve_device()

    env = TaxEnv(
        MAX_STRADDLES=5,
        MAX_STEPS=14,
        SUCCESS_TAX_ADVANTAGE=30.0,
    )

    embed_dim = 128
    num_action_types = 4

    model = PolicyValueNet(
        embed_dim=embed_dim,
        num_action_types=num_action_types,
        max_straddles=env.max_straddles,
        num_fractions=len(env.fractions),
        num_individuals=len(env.individual_ids),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    buffer = RolloutBuffer()

    snapshot_dir = Path("straddle_abuse/rl_snapshots")
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
                    "tax_reduction": float(info.get("tax_reduction", 0.0)),
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
            if (
                best_snapshot is not None
                and save_snapshots
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
                print(
                    "snapshot_saved: "
                    f"{(snapshot_dir / f'best_structure_update_{update:04d}.png').resolve()}",
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


if __name__ == "__main__":
    train()
