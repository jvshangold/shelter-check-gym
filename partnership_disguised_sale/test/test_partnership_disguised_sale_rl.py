import pytest


def test_policy_samples_valid_action_and_ppo_updates():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    import torch

    from partnership_disguised_sale.rl.model import PolicyValueNet
    from partnership_disguised_sale.rl.ppo import ppo_update
    from partnership_disguised_sale.rl.rollout import RolloutBuffer
    from partnership_disguised_sale.rl.train_agent import (
        action_dict_to_env_action,
        reset_training_env,
        sample_action,
    )
    from partnership_disguised_sale.tax_env.env import TaxEnv

    device = torch.device("cpu")
    env = TaxEnv(MAX_STEPS=4)
    model = PolicyValueNet(
        embed_dim=16,
        num_action_types=4,
        max_individuals=env.max_individuals,
        max_assets=env.max_assets,
        num_amounts=len(env.amount_buckets),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    buffer = RolloutBuffer()

    obs = reset_training_env(env, device)

    for _ in range(4):
        with torch.no_grad():
            action, log_prob, entropy, value, masks, _ = sample_action(
                model=model,
                env=env,
                obs=obs,
                device=device,
            )

        env_action = action_dict_to_env_action(action)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated

        assert not info["invalid_action"]

        buffer.add(
            obs=obs,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
            masks=masks,
        )
        obs = reset_training_env(env, device) if done else next_obs.to(device)

    with torch.no_grad():
        last_value = model(obs)["value"]

    buffer.compute_advantages(last_value)
    loss = ppo_update(model, optimizer, buffer, epochs=1)

    assert isinstance(loss, float)
