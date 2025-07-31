import gymnasium as gym
import highway_env
from models import create_model

def train_model(model_name, total_timesteps):
    model = create_model(model_name, gym.make("parking-v0"))
    
    model.learn(total_timesteps=total_timesteps, log_interval=total_timesteps // 100)
    print(f"{model_name} model trained for {total_timesteps} timesteps.")
    model.save(f"../checkpoints/{model_name}_model_{total_timesteps}")
    print(f"Trained model saved to ../checkpoints/{model_name}_model_{total_timesteps}.")
    return model

def train():   
    models = [
        'PPO', 
        # 'DDPG', 
        # 'DQN'
    ]
    training_timestamps = [
        10_000, 
        # 50_000, 
        # 250_000
    ]
    
    for timestamp in training_timestamps:
        for model_name in models:
            print(f"Training {model_name} for {timestamp} timesteps...")
            trained_model = train_model(model_name, total_timesteps=timestamp)
            print(f"{model_name} trained successfully.")

if __name__ == "__main__":
    train()