import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import argparse

def train_expert(env_name, total_timesteps):
    # Create the environment
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])

    # Initialize the PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=total_timesteps)

    return model

def collect_trajectories(model, env_name, num_trajectories):
    env = gym.make(env_name)
    trajectories = []

    for _ in range(num_trajectories):
        obs = env.reset()
        done = False
        trajectory = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)
            trajectory.append((obs, action, reward, next_obs, done))
            obs = next_obs

        trajectories.append(trajectory)

    return trajectories

def main():
    parser = argparse.ArgumentParser(description="Train expert and collect trajectories")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps for training")
    parser.add_argument("--trajectories", type=int, default=10, help="Number of trajectories to collect")
    args = parser.parse_args()

    # Train the expert
    expert_model = train_expert(args.env, args.timesteps)

    # Collect trajectories
    trajectories = collect_trajectories(expert_model, args.env, args.trajectories)

    # Print some statistics
    print(f"Collected {len(trajectories)} trajectories")
    print(f"Average trajectory length: {np.mean([len(traj) for traj in trajectories]):.2f}")

if __name__ == "__main__":
    main()

