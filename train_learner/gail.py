import argparse
import gym
import my_gym
import os
import sys
import pickle
import time
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch import nn
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent


parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
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
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=6144, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--exp-type', type=str, default="mismatch", metavar='N',
                    help = "experiment type: noise, friction or mismatch")
parser.add_argument('--alg', type=str, default="gail", metavar='N',
                    help = "gail or gaifo")
parser.add_argument('--reward-type', type=str, default="negative", metavar='N',
                    help = "functional form of the reward taking as "
                           "input the discriminator output. Options: positive, negative, airl")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--noiseE', type=float, default=None, metavar='G')
parser.add_argument('--grid-type', type=int, default=None, metavar='N')

parser.add_argument('--scheduler-lr', action='store_true', default=False,
                    help='Use discriminator lr scheduler')
parser.add_argument('--warm-up', action='store_true', default=False,
                    help='Discriminator Warm UP')
parser.add_argument('--n-expert-trajs', type=int, default=2, metavar='N')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
print(device, "device")
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
max_grad = 40
global subfolder
"""environment"""
reward_type = args.reward_type
"""environment"""
if ( args.env_name == "gridworld-v0" or 
        args.env_name == "ContinuousGridworld-v0" or
            args.env_name == "GaussianGridworld-v0" or 
                args.env_name == "DiscreteGaussianGridworld-v0"):
    env = gym.make(args.env_name, prop = args.noiseE, env_type = args.grid_type)
    subfolder = "env"+str(args.env_name)+"type"+str(args.grid_type)+"noiseE"+str(args.noiseE)
    with open(assets_dir(subfolder+"/expert_trajs/"+args.expert_trajs), "rb") as f:
        data = pickle.load(f)
if reward_type == "airl":
    if not os.path.isdir(assets_dir(subfolder+f"/airl/learned_models")):
        os.makedirs(assets_dir(subfolder+f"/airl/learned_models"))
    if not os.path.isdir(assets_dir(subfolder+f"/airl/reward_history")):
        os.makedirs(assets_dir(subfolder+f"/airl/reward_history"))
else:
    if not os.path.isdir(assets_dir(subfolder+f"/gail/learned_models")):
        os.makedirs(assets_dir(subfolder+f"/gail/learned_models"))
    if not os.path.isdir(assets_dir(subfolder+f"/gail/reward_history")):
        os.makedirs(assets_dir(subfolder+f"/gail/reward_history"))

state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0

action_dim = 1 if is_disc_action else env.action_space.shape[0]
running_state = lambda x : x #ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
if is_disc_action:
    policy_net = DiscretePolicy(state_dim, env.action_space.n)
else:
    policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
    if args.env_name == "Ant-v2":
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std, hidden_size=(256,256))

"""define discriminator"""
if args.alg == "gaifo" or args.alg=="airl":
    if args.env_name == "HalfCheetah-v2" or args.env_name == "Ant-v2":
        discrim_net = Discriminator(state_dim + state_dim, hidden_size=(400,300))

    else:
        discrim_net = Discriminator(state_dim + state_dim)
elif args.alg == "gail":
    discrim_net = Discriminator(state_dim + env.action_space.n) \
        if is_disc_action else Discriminator(state_dim + env.action_space.shape[0])


value_net = Value(state_dim)
discrim_criterion = nn.BCEWithLogitsLoss()
to_device(device, policy_net, value_net, discrim_net, discrim_criterion)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

if args.scheduler_lr:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_discrim,
                                                step_size=1000, gamma=0.5)

# optimization epoch number and batch size for PPO
optim_epochs = 20 #10
optim_batch_size = 64
state_only = True if (args.alg == "gaifo") else False

if is_disc_action and (args.alg == "gail" or args.alg == "airl"):
    actions = data["actions"][0]
    for l in data["actions"][1:args.n_expert_trajs]:
        actions = actions + l
    states = data["states"][0][:-1]
    for l in data["states"][1:args.n_expert_trajs]:
        states = states + l[:-1]
    
    one_hot_actions = to_categorical(actions, env.action_space.n)
    expert_traj = np.concatenate([np.vstack(states), one_hot_actions], axis=1)


def expert_reward(state, next, reward_type):
    if args.alg == "gaifo":
        input_discrim = tensor(np.hstack([state, next]), dtype=dtype)

    elif args.alg == "gail" or args.alg == "airl":
        if isinstance(next, int):
            next = torch.from_numpy(to_categorical(next, env.action_space.n)).to(dtype).to(device)
        input_discrim = tensor(np.hstack([state, next]), dtype=dtype)
    with torch.no_grad():
        if reward_type == "airl":
            return math.log(1 - torch.sigmoid(discrim_net(input_discrim))[0].item() + 1e-8)   \
                   - math.log(torch.sigmoid(discrim_net(input_discrim))[0].item() + 1e-8)
        if reward_type == "negative":
            return math.log(1 - torch.sigmoid(discrim_net(input_discrim))[0].item()  + 1e-8)
        if reward_type == "positive":
            return -math.log(torch.sigmoid(discrim_net(input_discrim))[0].item() + 1e-8)
"""create agent"""
agent = Agent(env, policy_net, device, custom_reward=expert_reward,
              running_state=running_state, render=args.render, num_threads=args.num_threads,
              state_only = state_only,
              reward_type= reward_type)


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    next_states = torch.from_numpy(np.stack(batch.next_state)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)
        fixed_log_probs = policy_net.get_log_prob(states, actions)
    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """update discriminator"""
    expert_discrim_input = torch.from_numpy(expert_traj).to(dtype).to(device)
    
    if i_iter < 10 and args.warm_up:
        discrim_epoch = 10
    else:
        discrim_epoch = 1
    for _ in range(discrim_epoch):
        if args.alg == "gail" or args.alg == "airl":
            if is_disc_action:
                actions_g = torch.from_numpy(to_categorical(actions.detach().numpy().astype(int),
                                                            env.action_space.n)).to(dtype).to(device)
                g_o = discrim_net(torch.cat([states, actions_g], 1))
            else:
                g_o = discrim_net(torch.cat([states, actions], 1))
        elif args.alg == "gaifo" :
            g_o = discrim_net(torch.cat([states, next_states], 1))
        e_o = discrim_net(expert_discrim_input)
        optimizer_discrim.zero_grad()
        discrim_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
            discrim_criterion(e_o, zeros((expert_traj.shape[0], 1), device=device))
        discrim_loss.backward()
        if args.scheduler_lr:
            scheduler.step()
        optimizer_discrim.step()

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs= \
            states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
            fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind],\

            # Update the player
            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg, max_grad = max_grad)
            
    
def main_loop():
    rewards = []
    episodes = []
    best_reward = -10000
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        discrim_net.to(torch.device('cpu'))
        batch, log = agent.collect_samples(args.min_batch_size)
        discrim_net.to(device)

        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\texpert_R_avg {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['avg_c_reward'], log['avg_reward']))
            rewards.append(log['avg_reward'])
            episodes.append(log['num_episodes'])
            to_save = {"rewards": rewards,
                        "episodes": episodes}
            if args.reward_type == "airl":
                pickle.dump(to_save, open(
                os.path.join(assets_dir(subfolder), 
                                'airl/reward_history/{}_{}.p'.format( str(args.seed), 
                                str(args.n_expert_trajs))), 'wb'))
            else:
                pickle.dump(to_save, open(
                os.path.join(assets_dir(subfolder), 
                            'gail/reward_history/{}_{}.p'.format(str(args.seed),
                            str(args.n_expert_trajs))), 'wb'))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
            if args.reward_type == "airl":
                pickle.dump((policy_net, value_net, discrim_net), open(os.path.join(assets_dir(subfolder),
                        'airl/learned_models/{}_{}.p'.format(str(args.seed),
                        str(args.n_expert_trajs))), 'wb'))
            else:
                pickle.dump((policy_net, value_net, discrim_net), open(os.path.join(assets_dir(subfolder),
                        'gail/learned_models/{}_{}.p'.format(str(args.seed),
                        str(args.n_expert_trajs))), 'wb'))
            
            if  log['avg_reward'] > best_reward:
                print(best_reward)
                if args.reward_type == "airl":
                    pickle.dump((policy_net, value_net, discrim_net),
                            open(os.path.join(assets_dir(subfolder),
                                              'airl/learned_models/{}_{}_best.p'.format(
                                                  str(args.seed),
                                                  str(args.n_expert_trajs))), 'wb'))
                else:
                    pickle.dump((policy_net, value_net, discrim_net),
                            open(os.path.join(assets_dir(subfolder),
                                              'gail/learned_models/{}_{}_best.p'.format(
                                                  str(args.seed),
                                                  str(args.n_expert_trajs))), 'wb'))

                best_reward = copy.deepcopy(log['avg_reward'])

            to_device(device, policy_net, value_net, discrim_net)


        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()