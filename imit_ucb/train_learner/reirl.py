import argparse
import gym
import gym_simple
import my_gym
import os
import sys
import pickle
import time
import copy
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_critic import Value
from utils.reirl_tools import AdamOptimizer, Weights
from torch import nn
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

parser = argparse.ArgumentParser(description='PyTorch REIRL example')
parser.add_argument('--env-name', default="GaussianGridworld-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-trajs', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 1)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=50000, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--exp-type', type=str, default="mismatch", metavar='N',
                    help="experiment type: noise, friction or mismatch")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--noiseE', type=float, default=0.0, metavar='G')
parser.add_argument('--noiseL', type=float, default=0.0, metavar='G')
parser.add_argument('--grid-type', type=int, default=None, metavar='N')
parser.add_argument('--mass-mulL', type=float, default=1.0, metavar='G',
                    help="Mass Multiplier for learner environment")
parser.add_argument('--len-mulL', type=float, default=1.0, metavar='G',
                    help="Lenght Multiplier for learner environment")
parser.add_argument('--mass-mulE', type=float, default=1.0, metavar='G',
                    help="Mass multiplier for expert environment")
parser.add_argument('--len-mulE', type=float, default=1.0, metavar='G',
                    help="Lenght multiplier for expert environment")
parser.add_argument('--scheduler-lr', action='store_true', default=False,
                    help='Use discriminator lr scheduler')
parser.add_argument('--warm-up', action='store_true', default=False,
                    help='Discriminator Warm UP')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda',
                      index=args.gpu_index) if torch.cuda.is_available() else torch.device(
    'cpu')
print(device, "device")
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
max_grad = 40
global subfolder
"""environment"""
if ( args.env_name == "gridworld-v0" or 
        args.env_name == "ContinuousGridworld-v0" or
            args.env_name == "GaussianGridworld-v0" or 
                args.env_name == "DiscreteGaussianGridworld-v0"):
    env = gym.make(args.env_name, prop = args.noiseE, env_type = args.grid_type)
    subfolder = "env"+str(args.env_name)+"type"+str(args.grid_type)+"noiseE"+str(args.noiseE)
    with open(assets_dir(subfolder+"/expert_trajs/"+args.expert_trajs), "rb") as f:
        data = pickle.load(f)

if not os.path.isdir(assets_dir(subfolder+f"/reirl/learned_models")):
    os.makedirs(assets_dir(subfolder+f"/reirl/learned_models"))
if not os.path.isdir(assets_dir(subfolder+f"/reirl/reward_history")):
    os.makedirs(assets_dir(subfolder+f"/reirl/reward_history"))

state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0

action_dim = 1 if is_disc_action else env.action_space.shape[0]
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

reirl_weights = Weights(state_dim + env.action_space.n)

optim_epochs = 3  # 10
optim_batch_size = 64


"""create agent"""

policy_net = DiscretePolicy(state_dim, env.action_space.n)
optimizer_policy = torch.optim.Adam(policy_net.parameters(),
                                    lr=args.learning_rate)


def expert_reward(state, next, reward_type):
    weights = torch.from_numpy(reirl_weights.read())
    next = np.eye(env.action_space.n)[next]
    state = torch.from_numpy(np.concatenate([state,next]))
    return torch.matmul(weights, state).detach().numpy()

running_state = lambda x : x
ppo_agent = Agent(env, policy_net, device, custom_reward=expert_reward,
                  running_state=running_state, render=args.render,
                  num_threads=args.num_threads,
                  alpha=1)

if is_disc_action:
    actions = data["actions"][0]
    for l in data["actions"][1:]:
        actions = actions + l
    states = data["states"][0][:-1]
    for l in data["states"][1:]:
        states = states + l[:-1]
    
    one_hot_actions = to_categorical(actions, env.action_space.n)
    expert_traj = np.concatenate([np.vstack(states), one_hot_actions], axis=1)

def reirl(expert_feature_expectations,
          random_feature_expectations,
          opt,
          max_iter=500,
          learning_rate=0.005,
          verbose=False):
    # Compute features expectations

    expert_feature_expectations_mean = np.mean(expert_feature_expectations,
                                               axis=0)

    n_random_trajectories = int(len(random_feature_expectations))
    # importance_sampling = np.zeros(n_random_trajectories)
    # Weights initialization
    w = reirl_weights.read().flatten()
    # Gradient descent
    for i in range(max_iter):
        if verbose:
            print('Iteration %s/%s' % (i + 1, max_iter))

        to_exp = np.dot(random_feature_expectations, w)
        to_exp -= np.max(to_exp)
        importance_sampling = np.exp(to_exp)
        importance_sampling /= np.sum(importance_sampling, axis=0)
        weighted_sum = np.sum(np.multiply(
            np.array([importance_sampling, ] * random_feature_expectations.shape[1]).T,
                                          random_feature_expectations
                                          ), 
                                axis=0)
        grad = expert_feature_expectations_mean - weighted_sum
        w += opt.update(grad)
        w = w / np.linalg.norm(w, keepdims=True)
    return w

def update_params(batch, i_iter, opt):
    """update discriminator"""
    act = to_categorical(np.stack(batch.action).astype(int), env.action_space.n)
    learner_traj = np.hstack([np.stack(batch.state), act])
    reirl_weights.write(
        reirl(expert_traj, learner_traj, opt))
    value_net = Value(state_dim)
    optimizer_value = torch.optim.Adam(value_net.parameters(),
                                       lr=args.learning_rate)
    if i_iter > 0:
        j_max = 3 #if i_iter < 20 else 15
        for j in range(j_max): #3):
            batch, log = ppo_agent.collect_samples(3000)
            # print(
            #     '{}\tT_sample {}\texpert_R_avg {}\tR_avg {}'.format(
            #         i_iter*j_max + j, log['sample_time'], log['avg_c_reward'],
            #         log['avg_reward']))
            states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(
                device)
            actions = torch.from_numpy(np.stack(batch.action)).to(
                dtype).to(device)
            rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(
                device)
            masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
            with torch.no_grad():
                values = value_net(states)
                fixed_log_probs = policy_net.get_log_prob(states,
                                                          actions)
            """get advantage estimation from the trajectories"""
            advantages, returns = estimate_advantages(rewards, masks, values,
                                                      args.gamma, args.tau,
                                                      device
                                                    )
            """perform mini-batch PPO update"""
            optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
            for _ in range(optim_epochs):
                perm = np.arange(states.shape[0])
                np.random.shuffle(perm)
                perm = LongTensor(perm).to(device)
                states, actions, returns, advantages, fixed_log_probs = \
                    states[perm].clone(), actions[perm].clone(), \
                     returns[perm].clone(), \
                    advantages[perm].clone(), \
                    fixed_log_probs[perm].clone()

                for i in range(optim_iter_num):
                    ind = slice(i * optim_batch_size,
                                min((i + 1) * optim_batch_size,
                                    states.shape[0]))
                    states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                        states[ind], actions[ind], \
                        advantages[ind], returns[ind], fixed_log_probs[ind]
                    # Update the player
                    ppo_step(policy_net, value_net, optimizer_policy,
                             optimizer_value, 1, states_b, actions_b,
                             returns_b,
                             advantages_b, fixed_log_probs_b, args.clip_epsilon,
                             args.l2_reg, max_grad=max_grad)
                    
def main_loop():
    rewards = []
    best_reward = -10000

    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = ppo_agent.collect_samples(args.min_batch_size)
        opt = AdamOptimizer(state_dim+env.action_space.n, 5e-3)
        t0 = time.time()
        update_params(batch, i_iter, opt)
        t1 = time.time()
        if i_iter % args.log_interval == 0:
            print(
                '{}\tT_sample {}\tT_update {}\texpert_R_avg {}\tR_avg {}'.format(
                    i_iter, log['sample_time'], t1 - t0, log['avg_c_reward'],
                    log['avg_reward']))
            rewards.append(log['avg_reward'])
            pickle.dump(rewards, open(
                os.path.join(assets_dir(subfolder),
                             'reirl/reward_history/{}.p'.format(args.env_name
                                                                + str(
                                 args.seed))), 'wb'))
        if args.save_model_interval > 0 and (
                i_iter + 1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net)
            pickle.dump(policy_net,
                        open(os.path.join(assets_dir(subfolder),
                                          'reirl/learned_models/{}.p'.format(
                                              args.env_name + str(args.seed))), 'wb'))
            if log['avg_reward'] > best_reward:
                print(best_reward)
                pickle.dump(policy_net,
                            open(os.path.join(assets_dir(subfolder),
                                              'reirl/learned_models/{}_best.p'.format(
                                                args.env_name + str(
                                                args.seed))), 'wb'))
                best_reward = copy.deepcopy(log['avg_reward'])

            to_device(device, policy_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()