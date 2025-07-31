import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from models import load_model
from stable_baselines3.common.vec_env import DummyVecEnv

def evaluate_single_model(model_name, total_timesteps, n_eval_episodes=100):
    print(f"  Evaluating {model_name} for {total_timesteps} timesteps...")
    
    eval_env = DummyVecEnv([lambda: gym.make("parking-v0")])
    model_path = f"../checkpoints/{model_name}_model_{total_timesteps}.zip"
    
    try:
        model = load_model(model_name, model_path, eval_env)
    except Exception as e:
        print(f"    Could not load model: {e}")
        return None

    episode_returns = []
    episode_lengths = []
    episode_infos = []
    
    for _ in tqdm(range(n_eval_episodes), desc="  Running episodes", leave=False):
        obs = eval_env.reset()
        done = False
        current_return = 0.0
        current_length = 0
        info = None
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            current_return += reward[0]
            current_length += 1

        # collect stats after each episode
        episode_returns.append(current_return)
        episode_lengths.append(current_length)
        episode_infos.append(info[0])

    eval_env.close()

    # calulcate stats
    successful_episodes = sum(1 for i in episode_infos if i.get('is_success', False))
    crashed_episodes = sum(1 for i in episode_infos if i.get('crashed', False))
    
    success_rate = (successful_episodes / n_eval_episodes) * 100
    crash_rate = (crashed_episodes / n_eval_episodes) * 100
    timeout_rate = 100 - success_rate - crash_rate
    
    avg_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    avg_ep_length = np.mean(episode_lengths)

    successful_returns = [r for r, i in zip(episode_returns, episode_infos) if i.get('is_success', False)]
    avg_success_return = np.mean(successful_returns) if successful_episodes > 0 else 0

    return {
        "Model": model_name,
        "Timesteps": total_timesteps,
        "Success Rate (%)": success_rate,
        "Crash Rate (%)": crash_rate,
        "Timeout Rate (%)": timeout_rate,
        "Avg Return": avg_return,
        "Std Dev Return": std_return,
        "Avg Success Return": avg_success_return,
        "Avg Ep Length": avg_ep_length,
    }

def create_graphs(df, graph_dir="../results/graphs"):
    os.makedirs(graph_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")

    plot_df = df.sort_values(by="Timesteps")
    plot_df['Timesteps'] = plot_df['Timesteps'].apply(lambda x: f"{x:,}")

    # plot 1: success rate vs. timesteps
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df, x="Timesteps", y="Success Rate (%)", hue="Model", marker="o")
    plt.title("Success Rate vs. Training Timesteps")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Success Rate (%)")
    plt.grid(True, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "success_rate_vs_timesteps.png"))
    plt.close()
    print("  - Saved 'success_rate_vs_timesteps.png'")

    # plot 2: average return vs. timesteps
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df, x="Timesteps", y="Avg Return", hue="Model", marker="o")
    plt.title("Average Return vs. Training Timesteps")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Return")
    plt.grid(True, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "avg_return_vs_timesteps.png"))
    plt.close()
    print("  - Saved 'avg_return_vs_timesteps.png'")

    # plot 3: episode outcomes for best models
    best_models_df = df[df['Timesteps'] == df['Timesteps'].max()]
    outcomes_df = best_models_df[['Model', 'Success Rate (%)', 'Crash Rate (%)', 'Timeout Rate (%)']].set_index('Model')
    
    outcomes_df.plot(kind='bar', stacked=True, figsize=(10, 7), colormap='viridis')
    plt.title(f"Episode Outcomes at {df['Timesteps'].max():,} Timesteps")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Model")
    plt.xticks(rotation=0)
    plt.legend(title='Outcome')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "episode_outcomes.png"))
    plt.close()
    print("  - Saved 'episode_outcomes.png'")

if __name__ == "__main__":
    models_to_eval = ['A2C', 'PPO', 'DDPG']
    timesteps_to_eval = [10_000, 50_000, 250_000]
    all_results = []

    print("Starting model evaluation...")
    for model_name in models_to_eval:
        for ts in timesteps_to_eval:
            results = evaluate_single_model(model_name, ts)
            if results:
                all_results.append(results)
    
    if not all_results:
        print("\nNo models were evaluated. Exiting.")
    else:
        results_df = pd.DataFrame(all_results)
        print("\n--- Evaluation Results ---")
        print(results_df.to_string())
        
        # save results to CSV
        os.makedirs("../results", exist_ok=True)
        print("\nSaving results to ../results/evaluation_results.csv")
        results_df.to_csv("../results/evaluation_results.csv", index=False)
        print("\nFull results saved to ../evaluation_results.csv")

        # generate graphs
        print("\n--- Generating Graphs ---")
        create_graphs(results_df)
        print("\nEvaluation complete. Graphs are saved in ../graphs/")
