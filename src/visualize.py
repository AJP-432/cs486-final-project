import gymnasium as gym
import highway_env
from models import create_model
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import os

def visualize_model(model_name, total_timesteps):
    base_env = gym.make("parking-v0", render_mode="rgb_array")
    env = Monitor(base_env)
    env = DummyVecEnv([lambda: env])

    model = create_model(model_name, env)
    model_path = f"../checkpoints/{model_name}_model_{total_timesteps}"
    try:
        model.load(model_path)
        print(f"Loaded {model_name} model from {model_path}.")
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    # Save multiple rollout videos
    video_dir = "../videos"
    os.makedirs(video_dir, exist_ok=True)

    num_episodes = 4
    max_steps = 500  # Adjust as needed

    for ep in range(num_episodes):
        print(f"Recording episode {ep + 1}...")
        # Reset the environment for each episode
        env = DummyVecEnv([lambda: Monitor(gym.make("parking-v0", render_mode="rgb_array"))])
        env = VecVideoRecorder(
            env,
            video_folder=video_dir,
            record_video_trigger=lambda step: True,
            video_length=max_steps,
            name_prefix=f"{model_name}_ep{ep+1}"
        )

        obs = env.reset()
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            if dones:
                break
        env.close()

    print(f"Saved {num_episodes} video(s) to {video_dir}/")

if __name__ == "__main__":
    model_name = 'PPO'  # Change to 'DDPG' or 'DQN' as needed
    total_timesteps = 10_000  # Adjust as needed
    visualize_model(model_name, total_timesteps)