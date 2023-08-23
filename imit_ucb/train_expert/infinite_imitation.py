import argparse
import gym
import gym_simple
import my_gym
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
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
if args.env_name == "gridworld-v0" or args.env_name == "ContinuousGridworld-v0" or args.env_name == "GaussianGridworld-v0" or args.env_name == "DiscreteGaussianGridworld-v0":
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

compute_features_expectation(data["states"], data["actions"],env)

def collect_trajectories(value_params, env, covariance_inv):
    state = env.reset()
    action_features = np.eye(env.action_space.n)
    h = 0
    states = []
    actions = []
    rewards = []
    done = False
    while h < 1e4 and not done:
        value = value_params.dot(np.vstack([state.reshape(-1,1).repeat(4, axis=1), action_features ]))
        reward = env.compute_reward()
        bonus = compute_bonus(state,covariance_inv)
        action = np.argmax(np.clip(reward + args.gamma*value + args.beta*bonus,
                                -80/(1 - args.gamma),100/(1 - args.gamma)))
        next_state, _, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state 
        h = h + 1
    print(done)
    if done:
        states.append(state)
        actions.append(np.random.choice(env.action_space.n))
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        states.append(next_state)
    return states, actions, rewards

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

def run_lsvi_ucb(K = 100):
    value_params = np.zeros(state_dim + env.action_space.n)
    action_features = np.eye(env.action_space.n)
    covariance_inv = 1/8e-2*np.eye(state_dim + env.action_space.n)
    """create agent"""
    states_dataset = []
    actions_dataset = []
    rewards_dataset = []
    
    for k in range(K):
        states, actions, rewards = collect_trajectories(value_params, env, covariance_inv )
        print("Episode " + str(k) + ": " + str(np.sum(rewards)))
        states_dataset = states_dataset + states
        actions_dataset = actions_dataset + actions
        rewards_dataset = rewards_dataset + rewards
        # Save expert data
        states_to_save = [states]
        actions_to_save = [actions]
        for _ in range(1):
            states_to_app, actions_to_app, _ = collect_trajectories(value_params, env, covariance_inv )
            states_to_save.append(states_to_app)
            actions_to_save.append(actions_to_app)
            # if not k % 45:
            #     plt.figure(k)
            #     plt.scatter(np.stack(states_to_app)[:,0], np.stack(states_to_app)[:,1] )
            #     plt.savefig("figs/"+ str(k) + ".png")
        with open(assets_dir(subfolder+"/expert_trajs")+"/trajs"+str(k)+".pkl", "wb") as f:
            pickle.dump({"states": states_to_save, "actions": actions_to_save}, f)
        covariance = compute_covariance(states_dataset, actions_dataset)
        covariance_inv = np.linalg.inv(covariance)
        targets_dataset = []
        for state, reward, next_state in zip(states_dataset[:-1], rewards_dataset, states_dataset[1:]):
            targets_dataset.append(np.max(np.clip(reward + args.gamma*value_params.dot(
                np.vstack([next_state.reshape(-1,1).repeat(4, axis=1), action_features ])) 
                + args.beta*compute_bonus(next_state, covariance_inv), -80/(1 - args.gamma), 100/(1 - args.gamma))))
        
        target = 0
        for state,action, value in zip(states_dataset[:-1], actions_dataset, targets_dataset):
            target = target + value*np.concatenate([state, np.eye(env.action_space.n)[action]])
        value_params = covariance_inv.dot(target)
        # if not k % 45:
        #      plt.figure(k)
        #      plt.scatter(np.stack(states)[:,0], np.stack(states)[:,1] )
        #      plt.savefig("figs/"+ str(k) + ".png")
        
        k = k + 1
run_lsvi_ucb()