import torch


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.masks = []

        self.advantages = None
        self.returns = None

    def add(self, obs, action, log_prob, reward, done, value, masks):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(value.detach())
        self.masks.append(masks)

    def clear(self):
        self.__init__()

    def compute_advantages(self, last_value, gamma=0.99, lam=0.95):
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        values = torch.stack(self.values).squeeze(-1)

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        last_value = last_value.detach().squeeze()

        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            gae = delta + gamma * lam * not_done * gae
            advantages[t] = gae

        self.advantages = advantages.detach()
        self.returns = (advantages + values).detach()

    def get_tensors(self):
        return {
            "observations": self.observations,
            "actions": self.actions,
            "log_probs": torch.stack(self.log_probs),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
            "advantages": self.advantages,
            "returns": self.returns,
            "masks": self.masks,
        }
