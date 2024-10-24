import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import argparse
import os
import pickle

def train_expert(env_name, total_timesteps, n_envs=4):
    # Create parallel environments
    vec_env = make_vec_env(env_name, n_envs=n_envs)

    # Initialize the PPO agent
    model = PPO("MlpPolicy", vec_env, verbose=1,
                learning_rate=3e-4,
                n_steps=2048 // n_envs,  # Adjust steps per environment
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs=dict(net_arch=[64, 64]))

    # Train the agent
    model.learn(total_timesteps=total_timesteps)

    return model, vec_env

def collect_trajectories(model, env_name, num_trajectories):
    env = gym.make(env_name)
    all_states = []
    all_actions = []

    for _ in range(num_trajectories):
        obs, _ = env.reset()
        done = False
        trajectory_states = []
        trajectory_actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            trajectory_states.append(obs)
            trajectory_actions.append(action)
            
            obs = next_obs

        all_states.extend(trajectory_states)
        all_actions.extend(trajectory_actions)

    return {
        'states': np.array(all_states),
        'actions': np.array(all_actions)
    }

def save_trajectories(trajectories, env_name):
    folder_name = os.path.join("assets", f"trajectories_{env_name}")
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, "expert_trajectories.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Trajectories saved to {file_path}")

def visualize_continuous(model, vec_env):
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        if dones.any():
            obs = vec_env.reset()

def main():
    parser = argparse.ArgumentParser(description="Train expert and collect trajectories")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--trajectories", type=int, default=10, help="Number of trajectories to collect")
    parser.add_argument("--visualize", action="store_true", help="Visualize the trained model continuously")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    args = parser.parse_args()

    # Train the expert
    expert_model, vec_env = train_expert(args.env, args.timesteps, n_envs=args.n_envs)

    # Save the model
    expert_model.save(f"ppo_{args.env}") 

    # In the main function, replace the existing collect_trajectories and save_trajectories calls with:
    trajectories = collect_trajectories(expert_model, args.env, args.trajectories)
    save_trajectories(trajectories, args.env)

    # Print some statistics
    print(f"Collected {len(trajectories['states'])} state-action pairs")
    print(f"Average trajectory length: {len(trajectories['states']) / args.trajectories:.2f}")

    # Visualize continuously if requested
    if args.visualize:
        print("Visualizing trained model continuously. Press Ctrl+C to stop.")
        try:
            visualize_continuous(expert_model, vec_env)
        except KeyboardInterrupt:
            print("Visualization stopped.")

    vec_env.close()

if __name__ == "__main__":
    main()
