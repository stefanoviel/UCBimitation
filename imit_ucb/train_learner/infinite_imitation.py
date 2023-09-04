import argparse
import gym
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
parser.add_argument('--beta', type=float, default=100.0, metavar='G',
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
if not os.path.isdir(assets_dir(subfolder+f"/ilarl/learned_models")):
    os.makedirs(assets_dir(subfolder+f"/ilarl/learned_models"))
if not os.path.isdir(assets_dir(subfolder+f"/ilarl/reward_history")):
    os.makedirs(assets_dir(subfolder+f"/ilarl/reward_history"))
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
    for traj_states, traj_actions in zip(states[:args.n_expert_trajs], actions[:args.n_expert_trajs]):
        h = 0
        features_exp = 0
        for state,action in zip(traj_states, traj_actions):
            features_exp = features_exp + \
                args.gamma**h * np.concatenate([state, np.eye(env.action_space.n)[action]])
            h = h + 1
        features.append(features_exp)
    return np.mean(features, axis=0)

expert_fev = compute_features_expectation(data["states"][:args.n_expert_trajs],
                                             data["actions"][:args.n_expert_trajs],env)

def collect_trajectories(value_params_list, reward_weights, env, covariance_inv):
    state = env.reset()
    action_features = np.eye(env.action_space.n)
    h = 0
    states = []
    next_states = []
    actions = []
    rewards = []
    done = False
    while h < 1e4 and not done:
        bonus = compute_bonus(state,covariance_inv)
        sum = 0
        for value_params, w in zip(value_params_list,reward_weights):
            value = value_params.dot(np.vstack([state.reshape(-1,1).repeat(4, axis=1), action_features ]))
            r = w.dot(np.vstack([state.reshape(-1,1).repeat(4, axis=1), action_features ]))
            sum = sum + np.clip(r + args.gamma*value + args.beta*bonus,
                                -80/(1 - args.gamma),100/(1 - args.gamma))
        action_distribution = special.softmax(sum/len(value_params_list))
        action = np.random.choice(env.action_space.n, p=action_distribution)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state 
        h = h + 1
    print(done)
    if done:
        states.append(state)
        actions.append(np.random.choice(env.action_space.n))
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        next_states.append(next_state)
    return states, actions, rewards, next_states

def compute_covariance(states_dataset, actions_dataset):
    features = []
    covariance = 1/8e-2*np.eye(state_dim + env.action_space.n)
    for state,action in zip(states_dataset, actions_dataset):
        features.append(np.concatenate([state, np.eye(env.action_space.n)[action]]))
    for feature in features:
        covariance += np.outer(feature, feature)
    return covariance
def compute_bonus(state, covariance_inv):
    bonus = []
    for a in range(env.action_space.n):
        feature = np.concatenate([state, np.eye(env.action_space.n)[a]])
        bonus.append(np.sqrt(feature.dot(covariance_inv).dot(feature)))
    return np.array(bonus)

def run_imitation_learning(K, tau=5):
    value_params = np.zeros(state_dim + env.action_space.n)
    value_params_list = [value_params]
    w = np.zeros(state_dim + env.action_space.n)
    reward_weights = [w]
    action_features = np.eye(env.action_space.n)
    covariance_inv = 1/8e-2*np.eye(state_dim + env.action_space.n)
    """create agent"""
    rs=[]
    for k in range(K):
        states_dataset = []
        actions_dataset = []
        next_states_dataset = []
        for i in range(tau):
            states, actions, true_rewards, next_states = collect_trajectories(value_params_list, 
                                                                reward_weights, 
                                                                env, 
                                                                covariance_inv )
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
        reward_weights = []
        for i in range(tau):
            
            w = w - \
                0.001*(compute_features_expectation(states_traj_data,actions_traj_data,env) - expert_fev)
            reward_weights.append(w)
        covariance = compute_covariance(states_dataset, actions_dataset)
        covariance_inv = np.linalg.inv(covariance)
        targets_dataset = []
        value_params_list = []
        for i in range(tau):
            for state, next_state in zip(states_dataset, next_states_dataset):
                reward = reward_weights[i].dot(
                    np.vstack([next_state.reshape(-1,1).repeat(4, axis=1), action_features ])) 
                targets_dataset.append(special.logsumexp(np.clip(reward + args.gamma*value_params.dot(
                    np.vstack([next_state.reshape(-1,1).repeat(4, axis=1), action_features ])) 
                    + args.beta*compute_bonus(next_state, covariance_inv), -80/(1 - args.gamma), 
                    100/(1 - args.gamma))))
            
            target = 0
            #i = 0
            for state,action, value in zip(states_dataset, actions_dataset, targets_dataset):
                #i = i + 1
                #print(i)
                target = target + value*np.concatenate([state, np.eye(env.action_space.n)[action]])
            value_params = covariance_inv.dot(target)
            value_params_list.append(value_params)
        
        # plt.figure(k)
        # plt.scatter(np.stack(states)[:,0], np.stack(states)[:,1], color="blue" )
        # plt.scatter(np.stack(data["states"][0])[:,0], np.stack(data["states"][0])[:,1],color="red")
        # plt.savefig("figs/"+ str(k) + "imit.png")
    with open(assets_dir(subfolder+f"/ilarl/reward_history/{args.seed}_{args.n_expert_trajs}.p"), "wb") as f:
        pickle.dump(np.array(rs), f)
    with open(assets_dir(subfolder+f"/ilarl/learned_models/{args.seed}_{args.n_expert_trajs}.p"), "wb") as f:
        pickle.dump({"thetas": value_params_list,
                     "covariance": covariance_inv}, 
                    f)
        
run_imitation_learning(args.max_iter_num)
