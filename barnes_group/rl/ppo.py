import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def masked_categorical(logits, mask=None):
    if mask is not None:
        if not mask.any():
            raise ValueError(f"All actions masked out. mask={mask}, logits={logits}")
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    logits = torch.clamp(logits, -50, 50)

    if mask is not None:
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    return Categorical(logits=logits)


def evaluate_action(model, obs, action, masks=None):
    out = model(obs)

    log_prob = 0.0
    entropy = 0.0

    for action_key, logits_key in [
        ("action_type", "action_logits"),
        ("corp_a", "corp_a_logits"),
        ("corp_b", "corp_b_logits"),
        ("stock", "stock_logits"),
        ("amount", "amount_logits"),
    ]:
        if action.get(action_key) is None:
            continue

        mask_key = "action" if action_key == "action_type" else action_key
        mask = None if masks is None else masks.get(mask_key)
        dist = masked_categorical(out[logits_key], mask)
        action_value = action[action_key]
        log_prob = log_prob + dist.log_prob(action_value).squeeze()
        entropy = entropy + dist.entropy().squeeze()

    return log_prob.squeeze(), entropy.squeeze(), out["value"]


def ppo_update(
    model,
    optimizer,
    rollout,
    clip_eps=0.2,
    value_coef=0.1,
    entropy_coef=0.05,
    epochs=3,
):
    data = rollout.get_tensors()
    old_log_probs = data["log_probs"].squeeze(-1)
    advantages = data["advantages"]
    returns = data["returns"]

    advantages = (
        (advantages - advantages.mean())
        / (advantages.std(unbiased=False) + 1e-8)
    )
    advantages = torch.clamp(advantages, -10, 10)

    total_loss = 0.0

    for _ in range(epochs):
        new_log_probs = []
        entropies = []
        values = []

        for i, (obs, action) in enumerate(zip(data["observations"], data["actions"])):
            log_prob, entropy, value = evaluate_action(
                model=model,
                obs=obs,
                action=action,
                masks=data["masks"][i],
            )
            new_log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)

        new_log_probs = torch.stack(new_log_probs).squeeze(-1)
        entropies = torch.stack(entropies)
        values = torch.stack(values).squeeze(-1)

        ratio = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

        actor_loss = -torch.min(unclipped, clipped).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_bonus = entropies.mean()

        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / epochs
