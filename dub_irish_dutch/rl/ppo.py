import torch
import torch.nn.functional as F
from torch.distributions import Categorical

def masked_categorical(logits, mask=None):
    if logits.dim() == 2 and logits.size(0) == 1:
        logits = logits.squeeze(0)

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

    action_mask = None if masks is None else masks.get("action")
    src_mask = None if masks is None else masks.get("src")
    dst_mask = None if masks is None else masks.get("dst")
    incorporation_mask = None if masks is None else masks.get("incorporation")
    management_mask = None if masks is None else masks.get("management")
    company_type_mask = None if masks is None else masks.get("company_type")

    # can we still add a child
    dist_action = masked_categorical(out["action_logits"], action_mask)
    log_prob = log_prob + dist_action.log_prob(action["action_type"])
    entropy = entropy + dist_action.entropy()

    if "src" in action and action["src"] is not None:
        dist_src = masked_categorical(out["src_logits"], src_mask)
        log_prob = log_prob + dist_src.log_prob(action["src"])
        entropy = entropy + dist_src.entropy()
    if "dst" in action and action["dst"] is not None:
        dist_dst = masked_categorical(out["dst_logits"], dst_mask)
        log_prob = log_prob + dist_dst.log_prob(action["dst"])
        entropy = entropy + dist_dst.entropy()
    if "incorporation" in action and action["incorporation"] is not None:
        dist_incorp = masked_categorical(out["incorporation_logits"], incorporation_mask)
        log_prob = log_prob + dist_incorp.log_prob(action["incorporation"])
        entropy = entropy + dist_incorp.entropy()
    if "management" in action and action["management"] is not None:
        dist_mgmt = masked_categorical(out["management_logits"], management_mask)
        log_prob = log_prob + dist_mgmt.log_prob(action["management"])
        entropy = entropy + dist_mgmt.entropy()
    if "company_type" in action and action["company_type"] is not None:
        dist_cmp_type = masked_categorical(out["company_type_logits"], company_type_mask)
        log_prob = log_prob + dist_cmp_type.log_prob(action["company_type"])
        entropy = entropy + dist_cmp_type.entropy()

    return log_prob.squeeze(), entropy.squeeze(), out["value"]

def ppo_update(
        model,
        optimizer,
        rollout,
        clip_eps=0.2,
        value_coef=0.1,
        entropy_coef=0.01,
        epochs=3,
):
    data = rollout.get_tensors()

    old_log_probs = data["log_probs"]
    advantages = data["advantages"]
    returns = data["returns"]

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    advantages = torch.clamp(advantages, -10, 10)

    total_loss = 0.0

    for _ in range(epochs):
        new_log_probs = []
        values = []
        entropies = []

        for i, (obs, action) in enumerate(zip(data["observations"], data["actions"])):
            masks = None
            if "masks" in data:
                masks = data["masks"][i]

            log_prob, entropy, value = evaluate_action(
                model=model,
                obs=obs,
                action=action,
                masks=masks,
            )

            new_log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
        
        new_log_probs = torch.stack(new_log_probs)
        entropies = torch.stack(entropies)
        values = torch.stack(values).squeeze(-1)

        ratio = torch.exp(new_log_probs - old_log_probs)

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

        actor_loss  = -torch.min(unclipped, clipped).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_bonus = entropies.mean()

        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / epochs
