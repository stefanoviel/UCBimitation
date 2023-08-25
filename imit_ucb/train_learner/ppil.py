import argparse
import gym
import gym_simple
import my_gym
from scipy import special
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent


parser = argparse.ArgumentParser(description='UCB')
parser.add_argument('--env-name', default="DiscreteGaussianGridworld-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-trajs', metavar='G',
                    help='path to expert data')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--eta', type=float, default=100.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--noiseE', type=float, default=0.0, metavar='G')
parser.add_argument('--grid-type', type=int, default=None, metavar='N')
parser.add_argument('--mass-mul', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot masses")
parser.add_argument('--len-mul', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot lengths")
parser.add_argument('--friction', default=False, action='store_true')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
if ( args.env_name == "gridworld-v0" or 
        args.env_name == "ContinuousGridworld-v0" or
            args.env_name == "GaussianGridworld-v0" or 
                args.env_name == "DiscreteGaussianGridworld-v0"):
    env = gym.make(args.env_name, prop = args.noiseE, env_type = args.grid_type)
    subfolder = "env"+str(args.env_name)+"type"+str(args.grid_type)+"noiseE"+str(args.noiseE)
    with open(assets_dir(subfolder+"/expert_trajs/"+args.expert_trajs), "rb") as f:
        data = pickle.load(f)

state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
assert(is_disc_action)
running_state = lambda x: x #ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

def compute_features_expectation(states,actions, env):
    features = []
    for traj_states, traj_actions in zip(states, actions):
        h = 0
        features_exp = 0
        for state,action in zip(traj_states, traj_actions):
            features_exp = features_exp + \
                args.gamma**h * np.concatenate([state, np.eye(env.action_space.n)[action]])
            h = h + 1
        features.append(features_exp)
    return np.mean(features, axis=0)

expert_fev = compute_features_expectation(data["states"], data["actions"],env)

def collect_trajectories(value_params_list, env):
    state = env.reset()
    action_features = np.eye(env.action_space.n)
    h = 0
    states = []
    next_states = []
    next_actions = []
    actions = []
    rewards = []
    done = False
    while h < 1e4 and not done:
        sum = 0
        for value_params in value_params_list:
            value = value_params.dot(np.vstack([state.reshape(-1,1).repeat(4, axis=1), action_features ]))
            sum = sum + value
        action_distribution = special.softmax(args.eta*sum/len(value_params_list))
        action = np.random.choice(env.action_space.n, p=action_distribution)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state 
        h = h + 1
    next_actions = actions[1:]
    sum = 0
    for value_params in value_params_list:
        value = value_params.dot(np.vstack([state.reshape(-1,1).repeat(4, axis=1), action_features ]))
        sum = sum + value
    action_distribution = special.softmax(args.eta*sum/len(value_params_list))
    action = np.random.choice(env.action_space.n, p=action_distribution)
    next_actions.append(action)
    print(done)
    if done:
        states.append(state)
        actions.append(np.random.choice(env.action_space.n))
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        next_states.append(next_state)
        next_actions.append(action)
    return states, actions, rewards, next_states, next_actions

def run_ppil(K = 100, tau=5):
    theta = np.zeros(state_dim + env.action_space.n)
    value_params_list = [theta]
    w = np.zeros(state_dim + env.action_space.n)
    action_features = np.eye(env.action_space.n)
    """create agent"""
    
    for k in range(K):
        states_dataset = []
        actions_dataset = []
        next_states_dataset = []
        next_actions_dataset = []
        for i in range(tau):
            states, actions, true_rewards, next_states, next_actions = collect_trajectories(value_params_list,  
                                                                env 
                                                                 )
            if i == 0:
                states_traj_data = [states]
                actions_traj_data = [actions]
            else:
                states_traj_data.append(states)
                actions_traj_data.append(actions)

            print("Episode " + str(k) + ": " + str(np.sum(true_rewards)))
            states_dataset = states_dataset + states
            actions_dataset = actions_dataset + actions
            next_states_dataset = next_states_dataset + next_states
            next_actions_dataset = next_actions_dataset + next_actions
        ### Approxiately solve logistic Bellman error minimization
        z = np.ones(len(states_dataset))/len(states_dataset)
        for _ in range(100):
            sample = np.random.choice(len(states_dataset), p = z)
            sample2 = np.random.choice(len(actions_traj_data))
            g_hat = - np.concatenate([states_dataset[sample], action_features[actions_dataset[sample]]]) + \
                                    args.gamma*np.concatenate([next_states_dataset[sample], 
                                action_features[next_actions_dataset[sample]]]) + \
                                     (1 - args.gamma)*np.concatenate([next_states_dataset[0], 
                                action_features[actions_traj_data[sample2][0]]])
            
            r = w.dot(np.concatenate([states_dataset[sample], action_features[actions_dataset[sample]]]))
            Q = theta.dot(np.concatenate(
                [states_dataset[sample], action_features[actions_dataset[sample]]]
                ))
            Q_next = theta.dot(
                np.vstack([next_states_dataset[sample].reshape(-1,1).repeat(4, axis=1), action_features ]
                ))
            sum = 0
            for past_theta in value_params_list:
                value = past_theta.dot(
                    np.vstack([next_states_dataset[sample].reshape(-1,1).repeat(4, axis=1), 
                    action_features ]))
                sum = sum + value
            pi = special.softmax(args.eta*sum/len(value_params_list))
            
            V_next = 1/args.eta*special.logsumexp(args.eta*Q_next + np.log(pi))
            delta = np.zeros(len(states_dataset))
            delta[sample] = r + V_next - Q
            z = special.softmax(delta)
            theta = theta - 0.01*g_hat
            w = w - 0.001*(compute_features_expectation(states_traj_data, actions_traj_data,env)
                            -
                            expert_fev)
            

        
        value_params_list.append(theta)
        
        plt.figure(k)
        plt.scatter(np.stack(states)[:,0], np.stack(states)[:,1], color="blue" )
        plt.scatter(np.stack(data["states"][0])[:,0], np.stack(data["states"][0])[:,1],color="red")
        plt.savefig("figs/"+ str(k) + "ppil.png")
        
        k = k + 1
run_ppil()