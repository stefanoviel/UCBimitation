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


parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
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
                    help='learning rate (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
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
    if not os.path.isdir(assets_dir(subfolder)):
        os.makedirs(assets_dir(subfolder))
        os.makedirs(assets_dir(subfolder+"/learned_models"))
elif args.env_name == "CartPole-v1" or args.env_name == "Acrobot-v1":
    env = gym.make(args.env_name)
    subfolder = "env"+ args.env_name + "mass" + str(args.mass_mul)+ "len" + str(args.len_mul)
    if not os.path.isdir(assets_dir(subfolder)):
        os.makedirs(assets_dir(subfolder))
        os.makedirs(assets_dir(subfolder+"/learned_models"))
    if args.env_name == "Acrobot-v1":
        env.env.LINK_LENGTH_1 *= args.len_mul
        env.env.LINK_LENGTH_2 *= args.len_mul
        env.env.LINK_MASS_1 *= args.mass_mul
        env.env.LINK_MASS_2 *= args.mass_mul
    elif args.env_name == "CartPole-v1":
        env.env.masspole *= args.mass_mul
        env.env.masscart *= args.mass_mul
        env.env.length *= args.len_mul
elif args.env_name == "HalfCheetah-v2" or args.env_name == "Walker2d-v2" or args.env_name == "Hopper-v2" or args.env_name == "Swimmer-v2" or args.env_name == "InvertedPendulum-v2" or args.env_name == "InvertedDoublePendulum-v2" or args.env_name == "Ant-v2":
    env = gym.make(args.env_name)
    if not args.friction:
        env.env.model.body_mass[:] *= args.mass_mul
        subfolder = "env"+ args.env_name + "mass" + str(args.mass_mul)
    else:
        env.env.model.geom_friction[:] *= args.mass_mul
        subfolder = "env" + args.env_name + "friction" + str(args.mass_mul)
    if not os.path.isdir(assets_dir(subfolder)):
        os.makedirs(assets_dir(subfolder))
        os.makedirs(assets_dir(subfolder+"/learned_models"))
else:
    env = gym.make(args.env_name)
    subfolder = None
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
running_state = lambda x: x #ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
    value_net = Value(state_dim)
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
policy_net.to(device)
value_net.to(device)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 10
optim_batch_size = 64

"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, render=args.render, num_threads=args.num_threads)


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    
    if not i_iter % 10:
        plt.figure(i_iter)
        plt.scatter(np.stack(batch.state)[:,0], np.stack(batch.state)[:,1] )
        plt.savefig("figs/"+ str(i_iter) + ".png")
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)
        fixed_log_probs = policy_net.get_log_prob(states, actions)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)


def main_loop():
    rewards = []
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))
            rewards.append([log['min_reward'], log['max_reward'], log['avg_reward']])
        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            pickle.dump((policy_net, value_net),
                        open(os.path.join(assets_dir(subfolder), 'learned_models/{}_ppo_{}.p'.format(args.env_name, i_iter)), 'wb'))
            to_device(device, policy_net, value_net)
            pickle.dump(np.array(rewards),open(os.path.join(assets_dir(subfolder),'learned_models/reward_{}.p'.format(args.env_name)), 'wb')) 
        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()
