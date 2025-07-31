import gymnasium as gym
import highway_env
from models import load_model
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
from tqdm import trange

def visualize_model(model_name, total_timesteps):
    base_env = gym.make("parking-v0", render_mode="rgb_array")
    env = Monitor(base_env)
    env = DummyVecEnv([lambda: env])

    model_path = f"../checkpoints/{model_name}_model_{total_timesteps}"
    try:
        model = load_model(model_name, model_path, env)
        print(f"Loaded {model_name} model from {model_path}.")
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    video_dir = "../videos"
    video_folder = os.path.join(video_dir, f"{model_name}_{total_timesteps}_videos")
    os.makedirs(video_folder, exist_ok=True)
    
    num_videos_to_record = 4
    print(f"  Recording {num_videos_to_record} separate videos...")

    for i in range(num_videos_to_record):
        print(f"    - Recording video {i + 1}/{num_videos_to_record}...")
        
        record_env = RecordVideo(
            env=gym.make("parking-v0", render_mode="rgb_array"),
            video_folder=video_folder,
            name_prefix=f"rollout-{i}-{model_name}-{total_timesteps}",
        )

        vec_env = DummyVecEnv([lambda: record_env])
        model.set_env(vec_env)

        obs = vec_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, _rewards, done, _info = vec_env.step(action)
        
        vec_env.close()
        print(f"      Video {i + 1} saved.")

    print(f"  Finished recording. Videos are in: {video_folder}")
    print("-" * 40)

if __name__ == "__main__":
    models = ['A2C', 'PPO', 'DDPG']
    total_timesteps = [10_000, 50_000, 250_000]
    for model_name in models:
        print(f"Visualizing {model_name} model...")
        for ts in total_timesteps:
            print(f"Visualizing {model_name} model trained for {ts} timesteps...")
            visualize_model(model_name, ts)