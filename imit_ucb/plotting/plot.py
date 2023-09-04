import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import *

plt.style.use('seaborn')
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

colors = {"ppil": "green",
            "iqlearn":"goldenrod",
            "ilarl":"blue",
            "gail":"brown",
            "airl":"gray",
            "reirl":"darkcyan"}
for alg in algs:
    to_plot_x = []
    to_plot_y = []
    for seed in range(0,5):
        with open(assets_dir(subfolder+f"/{alg}/reward_history/{seed}_{args.n_expert_trajs}.p"), "rb") as f:
            data = pickle.load(f)
        if alg in ["ppil","ilarl","iqlearn"]:
            tau = 5 if alg in ["ppil","ilarl"] else 1
            to_plot_y.append(np.array([np.mean(data[tau*i:tau*i+tau])
                for i in range(np.int(len(data)/tau))]))
            to_plot_x.append(np.array([tau*i 
                for i in range(np.int(len(data)/tau))]))
        else:
            to_plot_x.append(np.cumsum(data["episodes"]))
            to_plot_y.append(data["rewards"])
    means.append(np.mean(to_plot_y,axis=0))
    stds.append(np.std(to_plot_y, axis=0))
    if alg == "reirl":
        xs.append(np.mean(to_plot_x,axis=0)/15)
    else:
        xs.append(np.mean(to_plot_x,axis=0))
#import pdb; pdb.set_trace()
plt.figure()
if args.noiseE == 0.0:
    max_expert = -546
    random_p = -754
elif args.noiseE == 0.05:
    max_expert = -563
    random_p = -754
elif args.noiseE == 0.1:
    max_expert = -529
    random_p = -724

for m,s,x,alg in zip(means, #[3:], 
                        stds, #[3:], 
                        xs, #[3:], 
                        algs, #[3:]
                        ):
    m_norm = (m - random_p)/(max_expert - random_p)
    s_norm = s/(max_expert - random_p)
    plt.plot(x,m_norm,"-o", color=colors[alg], label=alg)
    plt.fill_between(x,m_norm-s_norm,
                             m_norm+s_norm,
                             facecolor = colors[alg], 
                             alpha=0.1)
plt.legend(fontsize=20)
plt.xticks(fontsize=14)
plt.xlabel("MDP trajectories", fontsize=14)
plt.ylabel("Normalized Return", fontsize=14)
plt.savefig(f"fig.pdf")
plt.show()


        