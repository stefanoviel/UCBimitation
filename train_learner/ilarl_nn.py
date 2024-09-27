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


def parse_arguments():
    parser = argparse.ArgumentParser(description='UCB')
    parser.add_argument('--env-name', default="DiscreteGaussianGridworld-v0", metavar='G',
                    help='name of the environment to run')
    parser.add_argument('--noiseE', type=float, default=0.0, metavar='G', help='probability of choosing a random action')
    parser.add_argument('--grid-type', type=int, default=None, metavar='N', help='1 easier, 0 harder, check environment for more details')
    parser.add_argument('--expert-trajs', metavar='G', help='path to expert data')
    parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                        help='maximal number of run of the algorithm')
    parser.add_argument('--num-of-NNs', type=int, default=10, metavar='N',
                        help='number of neural networks to use')
    
    # parser.add_argument('--beta', type=float, default=100.0, metavar='G',
    #                     help='log std for the policy (default: -0.0)')
    # parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
    #                     help='discount factor (default: 0.99)')
    # parser.add_argument('--seed', type=int, default=1, metavar='N',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--n-expert-trajs', type=int, default=2, metavar='N')

    return parser.parse_args()

def create_environment(args):
    env = gym.make(args.env_name, prop=args.noiseE, env_type=args.grid_type)
    env.seed(args.seed)
    return env

# TODO: then we assume that the expert is optimal? becuase if we get something better than the expert it will have a lower lowwer loss

def run_imitation_learning(K, L):
    # init pi c and zs (one z for each L) these are all neural networks
    # z is an estimation of PV^k where V are the avlues and P is the transition matrix
    
    for k in range(K):
        # sample one trajectory
        # estimate average reward for the expert trajectories using the current 
        # update c based on average reward of the expert C^{j+1} = \frac{1}{|\tau_pi_E|} \left[C^j - \eta \left( \sum_{s,a \sim \tau_pi_E} C_j(s,a) - \sum_{s,a \sim \tau_pi_j} C_j(s,a) \right)\right]

        for l in range(L):
            # sample one trajectory at most length N from the current policy 
            # update zi based on the discounted reward of the trajectory: \beta_Z^{k+1} = \theta_Z^k \ell - \eta \left( \nabla \sum_{i=1}^N z_\ell^k(s_i, q_i) - 2 \gamma \sum_{j=1+1}^N \gamma^{j-i-1} C(s_j^\ell, a_j^\ell) \right)
            
            pass

        # estimate z as the average of the zs
        # estimate b as std of the zs
        # Q = c + z - b

        # update pi based on Q \sum_{s \in \mathcal{D}} \left\langle \Pi_{\theta}(\cdot | s), \eta \left( Q^t(s, \cdot) + \log \frac{\Pi_{\theta}(\cdot | s)}{\Pi(\cdot | s)} \right) \right\rangle



# def main():
#     args = parse_arguments()
#     env = create_environment(args)
#     setup_directories(args)

#     expert_data = load_expert_data(args.expert_trajs)
#     expert_fev = compute_features_expectation(expert_data['states'][:args.n_expert_trajs],
#                                               expert_data['actions'][:args.n_expert_trajs],
#                                               env)

#     results = run_imitation_learning(args, env, expert_fev)
#     save_results(args, results)

# if __name__ == "__main__":
#     main()