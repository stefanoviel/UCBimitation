import argparse
import gym
import my_gym
from scipy import special
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt
import gym
import my_gym

def create_environment(args):
    env = gym.make(args.env_name, prop=args.noiseE, env_type=args.grid_type)
    env.seed(args.seed)
    return env


def parse_arguments():
    parser = argparse.ArgumentParser(description='UCB')
    # Add all your argument definitions here
    return parser.parse_args()






def main():
    args = parse_arguments()
    env = create_environment(args)
    setup_directories(args)

    expert_data = load_expert_data(args.expert_trajs)
    expert_fev = compute_features_expectation(expert_data['states'][:args.n_expert_trajs],
                                              expert_data['actions'][:args.n_expert_trajs],
                                              env)

    results = run_imitation_learning(args, env, expert_fev)
    save_results(args, results)

if __name__ == "__main__":
    main()