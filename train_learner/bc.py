import argparse
import gym
import gym_simple
import my_gym
from scipy import special
from sklearn.ensemble import HistGradientBoostingClassifier
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
parser.add_argument('--n-expert-trajs', type=int, default=2, metavar='N')
parser.add_argument('--noiseE', type=float, default=0.0, metavar='G')
parser.add_argument('--grid-type', type=int, default=None, metavar='N')

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
if not os.path.isdir(assets_dir(subfolder+f"/bc/learned_models")):
    os.makedirs(assets_dir(subfolder+f"/bc/learned_models"))
if not os.path.isdir(assets_dir(subfolder+f"/bc/reward_history")):
    os.makedirs(assets_dir(subfolder+f"/bc/reward_history"))
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
assert(is_disc_action)
running_state = lambda x: x #ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

def collect_trajectories(policy, env):
    state = env.reset()
    action_features = np.eye(env.action_space.n)
    h = 0
    states = []
    next_states = []
    next_actions = []
    actions = []
    rewards = []
    done = False
    while h < 4e4 and not done: #before was 4e4
        action = policy.predict(state.reshape(1,-1))[0]
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state 
        h = h + 1
    next_actions = actions[1:]
    print(done)
    if done:
        states.append(state)
        actions.append(policy.predict(state.reshape(1,-1))[0])
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        next_states.append(next_state)
        next_actions.append(action)
    return states, actions, rewards, next_states, next_actions

def run_bc():
    if args.noiseE == 0.0:
        policy = HistGradientBoostingClassifier(min_samples_leaf=1).fit(
            np.array(data["states"])[:,:-1,:].reshape(-1, state_dim), 
            np.array(data["actions"]).reshape(-1, 1))
        

    else:
        ls = []
        lsa = []
        for traj,traja in zip(data["states"][:args.n_expert_trajs],
                                data["actions"][:args.n_expert_trajs]):
            ls.append(traj[:-1])
            lsa = lsa + traja
        policy = HistGradientBoostingClassifier(min_samples_leaf=1).fit(
            np.vstack(ls), 
            np.vstack(lsa))
    _, _, rewards, _, _ = collect_trajectories(policy, env)

    rs = np.sum(np.array([args.gamma**h for h in range(len(rewards))])*rewards)
    print(rs)

    with open(assets_dir(subfolder+f"/bc/reward_history/{args.seed}_{args.n_expert_trajs}.p"), "wb") as f:
            pickle.dump(np.array([rs]), f)
    with open(assets_dir(subfolder+f"/bc/learned_models/{args.seed}_{args.n_expert_trajs}.p"), "wb") as f:
            pickle.dump(policy, f)

        
run_bc()