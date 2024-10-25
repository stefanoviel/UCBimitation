import argparse
import torch
from ilarl_nn.environment import create_environment
from ilarl_nn.training import run_imitation_learning
from ilarl_nn.utils import save_results

def parse_arguments():
    parser = argparse.ArgumentParser(description='UCB')
    parser.add_argument('--env-name', default="DiscreteGaussianGridworld-v0", metavar='G',
                    help='name of the environment to run')
    parser.add_argument('--noiseE', type=float, default=0.0, metavar='G', help='probability of choosing a random action')
    parser.add_argument('--grid-type', type=int, default=None, metavar='N', help='1 easier, 0 harder, check environment for more details')
    parser.add_argument('--expert-trajs', metavar='G', help='path to expert data')
    parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                        help='maximal number of run of the algorithm')
    parser.add_argument('--num-of-NNs', type=int, default=5, metavar='N',
                        help='number of neural networks to use')
    parser.add_argument('--seed', type=int, default=1, metavar='N')
    parser.add_argument('--eta', type=float, default=0.1, metavar='G')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G')
    parser.add_argument('--use-memory-replay', action='store_true',
                        help='use memory replay for policy updates')
    parser.add_argument('--buffer-size', type=int, default=2e5, metavar='N',
                        help='size of the replay buffer')
    parser.add_argument('--batch-size', type=int, default=2e4, metavar='N',
                        help='batch size for policy updates')
    parser.add_argument('--log-dir', type=str, default='runs',
                        help='directory for tensorboard logs')
    parser.add_argument('--z-std-multiplier', type=float, default=1.0, metavar='G',
                        help='multiplier for the standard deviation of the z-values')
    parser.add_argument('--recompute_rewards', action='store_true', help='Recompute rewards using the reward network instead of using stored rewards')
    parser.add_argument('--target_update_freq', type=int, default=100, help='Number of steps between target network updates')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    env = create_environment(args)
    
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    il_agent, all_true_rewards = run_imitation_learning(
        env, args.expert_trajs, args.max_iter_num, args.num_of_NNs, device, args, args.z_std_multiplier,
         args.seed,
        use_memory_replay=args.use_memory_replay,
        buffer_size=int(args.buffer_size),
        batch_size=int(args.batch_size),
        log_dir=args.log_dir,
        recompute_rewards=args.recompute_rewards
    )

    save_results(args, all_true_rewards)
