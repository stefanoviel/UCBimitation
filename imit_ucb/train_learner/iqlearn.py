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
parser.add_argument('--n-expert-trajs', type=int, default=2, metavar='N')
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
if not os.path.isdir(assets_dir(subfolder+f"/iqlearn/learned_models")):
    os.makedirs(assets_dir(subfolder+f"/iqlearn/learned_models"))
if not os.path.isdir(assets_dir(subfolder+f"/iqlearn/reward_history")):
    os.makedirs(assets_dir(subfolder+f"/iqlearn/reward_history"))
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
assert(is_disc_action)
running_state = lambda x: x #ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

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
    while h < 4e3 and not done: #before was 4e4
        sum = 0
        for value_params in value_params_list:
            value = value_params.dot(np.vstack([state.reshape(-1,1).repeat(4, axis=1), action_features ]))
            sum = sum + value
        action_distribution = special.softmax(-args.eta*sum/len(value_params_list))
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
    action_distribution = special.softmax(-args.eta*sum/len(value_params_list))
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

def run_iqlearn(K = 100, tau=1):
    theta = np.zeros(state_dim + env.action_space.n)
    value_params_list = [theta]
    action_features = np.eye(env.action_space.n)
    """create agent"""
    
    for k in range(K):
        states_dataset = []
        actions_dataset = []
        next_states_dataset = []
        next_actions_dataset = []
        rs = []
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

            rs.append(np.sum(np.array([args.gamma**h for h in range(len(true_rewards))])*true_rewards))
            print("Episode " + str(k) + ": " + str(rs[-1]))
            states_dataset = states_dataset + states
            actions_dataset = actions_dataset + actions
            next_states_dataset = next_states_dataset + next_states
            next_actions_dataset = next_actions_dataset + next_actions
        ### Approxiately solve logistic Bellman error minimization
        for _ in range(20): # before was 100
            gradient=0
            n = 0
            for traj_state, traj_actions in zip(data["states"][:args.n_expert_trajs],
                                                 data["actions"][:args.n_expert_trajs]):
                for state, action, next_state in zip(traj_state[:-1],
                            traj_actions[:-1],
                            traj_state[1:]):
                    n = n + 1
                    feature_s_a = np.concatenate([state, action_features[action]])
                    Q_s_a = feature_s_a.dot(theta)
                    features_next_state = np.vstack([ np.concatenate([next_state, 
                                                        action_features[a]]) for a in range(env.action_space.n)])
                    Q_next_state = features_next_state.dot(theta)
                    V_next_state = special.logsumexp(Q_next_state, axis=0)
                    
                    probs = special.softmax(Q_next_state, axis=0)
                    gradient += (feature_s_a - args.gamma*features_next_state.T.dot(probs))*(0.5*Q_s_a - 0.5*args.gamma*V_next_state)
            gradient = gradient/n
            
            gradient_2 = 0
            n = len(states_dataset)
            for b in zip(states_dataset,
                        actions_dataset,
                        next_states_dataset):
                state, action, next_state = b
                features_state = np.vstack([ np.concatenate([state, 
                                                    action_features[a]]) for a in range(env.action_space.n)])
                
                features_next_state = np.vstack([ np.concatenate([next_state, 
                                                    action_features[a]]) for a in range(env.action_space.n)])
                
                Q_next_state = features_next_state.dot(theta)
                Q_state = features_state.dot(theta)
                
                probs_next_state = special.softmax(Q_next_state, axis=0)
                probs_state = special.softmax(Q_state, axis=0)

                value_state = special.logsumexp(Q_state)
                value_next_state = special.logsumexp(Q_next_state)
                gradient_2 += (features_state.T.dot(probs_state) - args.gamma*features_next_state.T.dot(probs_next_state))*(value_state - args.gamma*value_next_state)
            gradient = gradient - gradient_2/n
        
            theta = theta + 0.005*gradient
        
        value_params_list = [theta]
        
        # plt.figure(k)
        # plt.scatter(np.stack(states)[:,0], np.stack(states)[:,1], color="blue" )
        # plt.scatter(np.stack(data["states"][0])[:,0], np.stack(data["states"][0])[:,1],color="red")
        # plt.savefig("figs/"+ str(k) + "iqlearn.png")

    with open(assets_dir(subfolder+f"/iqlearn/reward_history/{args.seed}.pkl"), "wb") as f:
        pickle.dump(np.array(rs), f)
    with open(assets_dir(subfolder+f"/iqlearn/learned_models/{args.seed}.pkl"), "wb") as f:
        pickle.dump(value_params_list, f)
        
run_iqlearn()