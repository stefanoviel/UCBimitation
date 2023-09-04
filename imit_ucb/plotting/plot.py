import pickle
import argparse
import numpy as np
from utils import *
parser = argparse.ArgumentParser(description='Grid search hyperparameters')
parser.add_argument('--noiseE', type=float, default=0.0, metavar='G')
parser.add_argument('--n-expert-trajs', type=int, default=2, metavar='G')
args = parser.parse_args()
subfolder = "envDiscreteGaussianGridworld-v0type1noiseE"+str(args.noiseE)
algs = ["ppil","ilarl","iqlearn", "gail", "airl", "reirl"]
to_plot_x = []
to_plot_y = []
means = []
stds = []
xs = []
for alg in algs:
    for seed in range(0,5):
        to_plot_x = []
        to_plot_y = []
        with open(assets_dir(subfolder+f"/{alg}/reward_history/{seed}_{args.n_expert_trajs}.p"), "rb") as f:
            data = pickle.load(f)
        if alg in ["ppil","ilarl","iqlearn"]:
            tau = 5 if alg in ["ppil","ilarl"] else 1
            to_plot_y.append(np.array([np.mean(data[tau*i:tau*i+tau - 1])
                for i in range(np.int(len(data)/tau))]))
            to_plot_x.append(np.array([tau*i 
                for i in range(np.int(len(data)/tau))]))
        else:
            to_plot_x = data["episodes"]
            to_plot_y = data["rewards"]
    means.append(np.mean(to_plot_y,axis=0))
    stds.append(np.std(to_plot_y, axis=0))
    xs.append(np.mean(to_plot_x,axis=0))

    import pdb; pdb.set_trace()

        