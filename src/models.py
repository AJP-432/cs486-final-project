import numpy as np
from stable_baselines3 import DDPG, PPO, DQN
from stable_baselines3.common.noise import NormalActionNoise

def create_model(model_name, env):
    if model_name == 'PPO':
        return _create_ppo_model(env)
    elif model_name == 'DDPG':
        return _create_ddpg_model(env)
    elif model_name == 'DQN':
        return _create_dqn_model(env)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def _create_ppo_model(env):
    model = PPO(
        "MultiInputPolicy",
        env,
        gamma=0.98,  # Discount factor
        batch_size=256,  # Batch size for training
        learning_rate=1e-3,  # Learning rate for the optimizer
        verbose=1,
    )
    return model

def _create_ddpg_model(env):
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    model = DDPG(
        "MultiInputPolicy",
        env,
        action_noise=action_noise,
        gamma=0.98,  # Discount factor
        batch_size=256,  # Batch size for training
        learning_rate=1e-3,  # Learning rate for the optimizer
        verbose=1,
    )
    return model

def _create_dqn_model(env):
    model = DQN(
        "MultiInputPolicy",
        env,
        gamma=0.98,  # Discount factor
        batch_size=256,  # Batch size for training
        learning_rate=1e-3,  # Learning rate for the optimizer
        verbose=1,
    )
    return model