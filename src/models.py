import numpy as np
from stable_baselines3 import DDPG, PPO, A2C
from stable_baselines3.common.noise import NormalActionNoise

def create_model(model_name, env):
    if model_name == 'PPO':
        return _create_ppo_model(env)
    elif model_name == 'DDPG':
        return _create_ddpg_model(env)
    elif model_name == 'A2C':
        return _create_a2c_model(env)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def load_model(model_name, model_path, env):
    if model_name == 'PPO':
        return PPO.load(model_path, env=env)
    elif model_name == 'DDPG':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        return DDPG.load(model_path, env=env, action_noise=action_noise)
    elif model_name == 'A2C':
        return A2C.load(model_path, env=env)
    else: 
        raise ValueError(f"Unknown model name: {model_name}")

def _create_ppo_model(env):
    model = PPO(
        "MultiInputPolicy",
        env,
        gamma=0.98,         # discount factor
        batch_size=256,     # batch size for training
        learning_rate=1e-3, # learning rate for the optimizer
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
        gamma=0.98,         # discount factor
        batch_size=256,     # batch size for training
        learning_rate=1e-3, # learning rate for the optimizer
        verbose=1,
    )
    return model

def _create_a2c_model(env):
    model = A2C(
        "MultiInputPolicy",
        env,
        gamma=0.98,         # discount factor
        n_steps=5,          # number of steps to run for each environment per update
        learning_rate=1e-3, # learning rate for the optimizer
        verbose=1,
    )
    return model