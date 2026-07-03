import copy
import os
from pathlib import Path

import torch

from dub_irish_dutch.tax_env.env import TaxEnv
from dub_irish_dutch.rl.ppo import masked_categorical, ppo_update
from dub_irish_dutch.rl.rollout import RolloutBuffer
from dub_irish_dutch.rl.model import PolicyValueNet


def resolve_device():
    requested = os.environ.get("SHELTER_CHECK_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("requested_device_unavailable: cuda; using cpu", flush=True)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def make_transfer_dst_mask(env, device):
    owner_idx = entity_to_idx(env, env.state.ip_owner)

    mask = []
    for i in range(len(env.idx_to_entity)):
        entity_id = env.idx_to_entity[i]
        entity = env.state.entities[entity_id]
        
        mask.append(
            i != owner_idx
            and entity.company_type == "Holding"
        )

    return torch.tensor(mask, dtype=torch.bool, device=device)


def full_mask(size, device):
    return torch.ones(size, dtype=torch.bool, device=device)


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

    action_mask = torch.tensor(env.get_action_mask(), dtype=torch.bool, device=device)

    action_type, action_log_prob, action_entropy = sample_from_logits_or_uniform(
        out["action_logits"],
        action_mask,
        force_uniform,
    )

    log_prob += action_log_prob
    entropy += action_entropy

    action_type_int = action_type.item()

    action = {
        "action_type": action_type,
        "src": None,
        "dst": None,
        "incorporation": None,
        "management": None,
        "company_type": None
    }

    used_masks = {
        "action": action_mask,
    }

    if action_type_int == 0:
        # add_child: src = parent, incorporation/management used
        parent_mask = make_entity_mask(env, device)

        parent, parent_log_prob, parent_entropy = sample_from_logits_or_uniform(
            out["src_logits"],
            parent_mask,
            force_uniform,
        )

        jurisdiction_mask = full_mask(len(env.idx_to_jurisdiction), device)
        incorporation, incorporation_log_prob, incorporation_entropy = (
            sample_from_logits_or_uniform(
                out["incorporation_logits"],
                jurisdiction_mask,
                force_uniform,
            )
        )

        management, management_log_prob, management_entropy = (
            sample_from_logits_or_uniform(
                out["management_logits"],
                jurisdiction_mask,
                force_uniform,
            )
        )

        company_type_mask = full_mask(len(env.idx_to_company_type), device)
        company_type, company_type_log_prob, company_type_entropy = (
            sample_from_logits_or_uniform(
                out["company_type_logits"],
                company_type_mask,
                force_uniform,
            )
        )

        log_prob += parent_log_prob
        log_prob += incorporation_log_prob
        log_prob += management_log_prob
        log_prob += company_type_log_prob

        entropy += parent_entropy
        entropy += incorporation_entropy
        entropy += management_entropy
        entropy += company_type_entropy

        action["src"] = parent
        action["incorporation"] = incorporation
        action["management"] = management
        action["company_type"] = company_type

        used_masks["src"] = parent_mask
        used_masks["incorporation"] = jurisdiction_mask
        used_masks["management"] = jurisdiction_mask
        used_masks["company_type"] = company_type_mask

    elif action_type_int == 1:
        # rent_ip: src = licensee, dst = licensor / IP holder

        licensee_mask = make_rent_licensee_mask(env, device)

        src, src_log_prob, src_entropy = sample_from_logits_or_uniform(
            out["src_logits"],
            licensee_mask,
            force_uniform,
        )

        licensor_mask = make_rent_licensor_mask(env, src.item(), device)

        dst, dst_log_prob, dst_entropy = sample_from_logits_or_uniform(
            out["dst_logits"],
            licensor_mask,
            force_uniform,
        )

        log_prob += src_log_prob
        log_prob += dst_log_prob

        entropy += src_entropy
        entropy += dst_entropy

        action["src"] = src
        action["dst"] = dst

        used_masks["src"] = licensee_mask
        used_masks["dst"] = licensor_mask

    elif action_type_int == 2:
        # transfer_ip: dst = new IP owner
        dst_mask = make_transfer_dst_mask(env, device)

        dst, dst_log_prob, dst_entropy = sample_from_logits_or_uniform(
            out["dst_logits"],
            dst_mask,
            force_uniform,
        )

        log_prob += dst_log_prob
        entropy += dst_entropy

        action["dst"] = dst

        used_masks["dst"] = dst_mask

    else:
        raise RuntimeError(f"Unknown action type: {action_type_int}")      

    return action, log_prob, entropy, out["value"], used_masks, force_uniform


def action_dict_to_env_action(action):
    return (
        int(action["action_type"].item()),
        int(action["src"].item()) if action["src"] is not None else 0,
        int(action["dst"].item()) if action["dst"] is not None else 0,
        int(action["incorporation"].item()) if action["incorporation"] is not None else 0,
        int(action["management"].item()) if action["management"] is not None else 0,
        int(action["company_type"].item()) if action["company_type"] is not None else 0
    )


def reset_training_env(env, device):
    obs, _ = env.reset()
    return obs.to(device)


def has_valid_action(env, device) -> bool:
    return bool(
        torch.tensor(
            env.get_action_mask(),
            dtype=torch.bool,
            device=device,
        ).any().item()
    )


def describe_env_action(env, env_action):
    action_type, src_idx, dst_idx, incorporation_idx, management_idx, company_type_idx = env_action

    if action_type == 0:
        parent_id = env.idx_to_entity.get(src_idx, f"idx_{src_idx}")
        incorporation = env.idx_to_jurisdiction[incorporation_idx]
        management = env.idx_to_jurisdiction[management_idx]
        company_type = env.idx_to_company_type[company_type_idx]
        return (
            f"add_child(parent={parent_id}, incorporation={incorporation}, "
            f"management={management}, company_type={company_type})"
        )

    if action_type == 1:
        licensee_id = env.idx_to_entity.get(src_idx, f"idx_{src_idx}")
        licensor_id = env.idx_to_entity.get(dst_idx, f"idx_{dst_idx}")
        return f"rent_ip(licensee={licensee_id}, licensor={licensor_id})"

    if action_type == 2:
        new_owner_id = env.idx_to_entity.get(dst_idx, f"idx_{dst_idx}")
        return f"transfer_ip(new_owner={new_owner_id})"

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

    _write_snapshot_metadata(
        metadata_path=image_stem.with_suffix(".txt"),
        snapshot=snapshot,
        update=update,
        snapshot_count=snapshot_count,
    )


def _write_snapshot_metadata(metadata_path, snapshot, update, snapshot_count):
    with metadata_path.open("w") as metadata:
        metadata.write(f"update: {update}\n")
        metadata.write(f"snapshot_index: {snapshot_count}\n")
        metadata.write(f"tax_advantage: {snapshot['tax_advantage']}\n")
        metadata.write(f"raw_tax_advantage: {snapshot['raw_tax_advantage']}\n")
        metadata.write(f"normalized_tax_advantage: {snapshot['normalized_tax_advantage']}\n")
        metadata.write(f"current_profit: {snapshot['current_profit']}\n")
        metadata.write(f"baseline_profit: {snapshot['baseline_profit']}\n")
        metadata.write(f"loophole_gate_complete: {snapshot['loophole_gate_complete']}\n")
        metadata.write(f"terminated: {snapshot['terminated']}\n")
        metadata.write(f"truncated: {snapshot['truncated']}\n")
        metadata.write(f"invalid_action: {snapshot['invalid_action']}\n")
        metadata.write("\nactions:\n")
        for i, action_description in enumerate(snapshot["actions"], start=1):
            metadata.write(f"{i}. {action_description}\n")


SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "rl_snapshots"


def train(total_updates=500, rollout_steps=256, save_snapshots=True, log_interval=25):
    device = resolve_device()

    env = TaxEnv(
        MAX_ENTITIES=4,
        MAX_STEPS=8,
    )

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

    snapshot_dir = SNAPSHOT_DIR
    saved_snapshot_count = 0

    for update in range(total_updates):
        obs = reset_training_env(env, device)

        buffer.clear()
        cumulative_reward = 0.0

        reward_by_action = {0: [], 1: [], 2: []}
        success_count = 0
        episode_count = 0
        invalid_count = 0
        positive_reward_count = 0
        random_exploration_count = 0
        best_tax_advantage = float("-inf")
        best_raw_tax_advantage = 0.0
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
            success_count += int(terminated)
            invalid_count += int(info.get("invalid_action", False))
            positive_reward_count += int(reward > 0.0)
            random_exploration_count += int(used_uniform)

            tax_advantage = float(info.get("tax_advantage", 0.0))
            raw_tax_advantage = float(info.get("raw_tax_advantage", 0.0))
            shorter_equal_advantage_trace = (
                best_snapshot is not None
                and tax_advantage == best_tax_advantage
                and raw_tax_advantage == best_raw_tax_advantage
                and len(current_episode_actions) < len(best_snapshot["actions"])
            )
            if (
                raw_tax_advantage > 0.0
                and (
                    best_snapshot is None
                    or tax_advantage > best_tax_advantage
                    or (
                        tax_advantage == best_tax_advantage
                        and raw_tax_advantage > best_raw_tax_advantage
                    )
                    or shorter_equal_advantage_trace
                )
            ):
                best_tax_advantage = tax_advantage
                best_raw_tax_advantage = raw_tax_advantage
                best_snapshot = {
                    "state": copy.deepcopy(env.state),
                    "tax_advantage": tax_advantage,
                    "raw_tax_advantage": raw_tax_advantage,
                    "normalized_tax_advantage": info.get(
                        "normalized_tax_advantage",
                        0.0,
                    ),
                    "current_profit": info.get("current_profit", 0.0),
                    "baseline_profit": info.get("baseline_profit", 0.0),
                    "loophole_gate_complete": info.get(
                        "loophole_gate_complete",
                        False,
                    ),
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

        loss = ppo_update(
            model=model,
            optimizer=optimizer,
            rollout=buffer,
        )

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
            print(
                "snapshot_saved: "
                f"{(snapshot_dir / f'best_structure_update_{update:04d}.png').resolve()}"
            )
            saved_snapshot_count += 1

        if update % log_interval == 0:
            print(
                f"update: {update}; loss: {loss}; "
                f"cumulative_reward: {cumulative_reward}; "
                f"success_count: {success_count}; "
                f"episode_count: {episode_count}; "
                f"invalid_count: {invalid_count}; "
                f"positive_reward_count: {positive_reward_count}; "
                f"random_exploration_count: {random_exploration_count}; "
                f"best_tax_advantage: {best_tax_advantage}; "
                f"random_action_prob: {random_action_prob:.4f}; "
                f"saved_snapshots: {saved_snapshot_count}"
            )
            for k, v in reward_by_action.items():
                if v:
                    print(k, sum(v) / len(v), len(v))


if __name__ == "__main__":
    train()
