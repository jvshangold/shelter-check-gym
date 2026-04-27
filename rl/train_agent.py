# import warnings
# warnings.filterwarnings("ignore")
# from torch import multiprocessing

import torch
from torch.distributions import Categorical

from tax_env.env import TaxEnv
from rl.ppo import masked_categorical, ppo_update
from rl.rollout import RolloutBuffer
from rl.model import PolicyValueNet


def make_masks(env, device):
    return {
        "action": torch.tensor(env.get_action_mask(), dtype=torch.bool, device=device),
        "src": torch.tensor(env.get_entity_mask(), dtype=torch.bool, device=device),
        "dst": torch.tensor(env.get_entity_mask(), dtype=torch.bool, device=device),
        "license": torch.tensor(env.get_license_mask(), dtype=torch.bool, device=device),
        "incorporation": torch.ones(5, dtype=torch.bool, device=device),
        "management": torch.ones(5, dtype=torch.bool, device=device),
    }

def sample_action(model, obs, masks):
    out = model(obs)

    log_prob = 0.0
    entropy = 0.0

    out = model(obs)

    log_prob = 0.0
    entropy = 0.0

    # can we still add a child
    dist_action = masked_categorical(out["action_logits"], masks["action"])
    action_type = dist_action.sample()
    log_prob = log_prob + dist_action.log_prob(action_type)
    entropy = entropy + dist_action.entropy()

    dist_src = masked_categorical(out["src_logits"], masks["src"])
    src = dist_src.sample()
    log_prob = log_prob + dist_src.log_prob(src)
    entropy = entropy + dist_src.entropy()

    dist_dst = masked_categorical(out["dst_logits"], masks["dst"])
    dst = dist_dst.sample()
    log_prob = log_prob + dist_dst.log_prob(dst)
    entropy = entropy + dist_dst.entropy()

    dist_incorp = masked_categorical(out["incorporation_logits"], masks["incorporation"])
    incorporation = dist_incorp.sample()
    log_prob = log_prob + dist_incorp.log_prob(incorporation)
    entropy = entropy + dist_incorp.entropy()

    dist_mgmt = masked_categorical(out["management_logits"], masks["management"])
    management = dist_mgmt.sample()
    log_prob = log_prob + dist_mgmt.log_prob(management)
    entropy = entropy + dist_mgmt.entropy()

    action = {
        "action_type": action_type,
        "src": src,
        "dst": dst,
        "incorporation": incorporation,
        "management": management,
    }

    return action, log_prob, entropy, out["value"]

def action_dict_to_env_action(action):
    return (
        int(action["action_type"].item()),
        int(action["src"].item()),
        int(action["dst"].item()),
        int(action["incorporation"].item()),
        int(action["management"].item())
    )

def train():
    '''
    Main loop logic function
    '''
    device = torch.device(torch.cuda() if torch.cuda.is_available() else "cpu")

    # initialize the environment
    env = TaxEnv()
    
    embed_dim = 128
    num_action_types = 3
    num_jurisdictions = 5

    # initialize networks needed
    model = PolicyValueNet(
        embed_dim=embed_dim, 
        num_action_types=num_action_types, 
        num_jurisdictions=num_jurisdictions,
    ).to(device=device)

    buffer = RolloutBuffer()

    total_updates = 1000
    rollout_steps = 256

    for update in range(total_updates):
        obs = env.reset()
        obs = obs.to(device)

        buffer.clear()
        cumulative_reward = 0.0

        for _ in range(rollout_steps):
            masks = make_masks(env, device)

            with torch.no_grad():
                action, log_prob, entropy, value = sample_action(model, obs, masks)

            env_action = action_dict_to_env_action(action)

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            next_obs = next_obs.to(device)

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
                obs, info = env.reset()
                obs = obs.to(device)
            else: 
                obs = next_obs
    return

if __name__ == '__main__':
    train()
