import os
import argparse

parser = argparse.ArgumentParser(description='Grid search hyperparameters')

parser.add_argument('--alg', metavar='G',
                    help='name of the environment to run')
parser.add_argument('--noiseE', type=float, default=0.0, metavar='G')
args = parser.parse_args()
n_trajs = [2] if args.noiseE == 0.0 else [1,2,3,5,10]
expert_f = "trajs16.pkl" if args.noiseE == 0.0 else "trajs6.pkl" if args.noiseE == 0.05 else "trajs30.pkl"
for seed in range(0,5):
    for n in n_trajs:
        string = f" python train_learner/{args.alg}.py \
            --env-name DiscreteGaussianGridworld-v0  --expert-trajs {expert_f} \
            --num-threads 1 --max-iter-num 30 \
            --save-model-interval 10 --grid-type 1 --noiseE 0.0 \
                --n-expert-trajs {n} --seed {seed}"
        if args.alg == "airl":
             string = f"{string} --reward-type airl"
        elif args.alg == "infinite_imitation":
            string = f"{string} --beta 8"
        command = f"alg{args.alg}noise{args.noiseE}n{n}seed{seed}"
        
    
        command=f"{command}.txt"
        with open(command, 'w') as file:
            file.write(f'{string}\n')
    
        os.system(f'sbatch --job-name={seed} submit.sh {command}')