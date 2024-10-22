# UCBimitation

Implementation of the experiments for the ICML 2024 paper: **Imitation Learning in Discounted Linear MDPs without exploration assumptions**

[Paper Link](https://arxiv.org/abs/2405.02181)

## Overview

This project implements a novel approach to imitation learning in Markov Decision Processes (MDPs). The method works without traditional exploration assumptions, making it more practical for real-world applications. Our approach considers infinite-horizon MDPs with discounted rewards, where the learner aims to imitate an expert policy using a dataset of expert trajectories.

The key innovation is handling imitation learning in environments where:
- The state and action spaces can be continuous
- The transition dynamics are unknown
- No exploration assumptions are required
- The cost function is unknown but assumed to be bounded

## Installation

Inside the `my_gym` folder, run:
```bash
pip install -e .
```

## Usage

To reproduce the experimental results, use the following command:

```bash
python -m ilarl_nn.main --env-name DiscreteGaussianGridworld-v0 \
      --expert-trajs assets/envDiscreteGaussianGridworld-v0type1noiseE0.0/expert_trajs/trajs16.pkl \
      --max-iter-num 100 --grid-type 1 --noiseE 0.0 --seed $seed \
      --num-of-NNs $nn --log-dir runs_memory_replay_zmul \
      --use-memory-replay --z-std-multiplier 10.0
```

Where:
- `$seed`: Random seed for reproducibility
- `$nn`: Number of neural networks to use in the ensemble

## Technical Background

The algorithm operates in an MDP framework defined by:
- State space (S)
- Action space (A)
- Transition dynamics (P)
- Cost function (c)
- Initial state distribution (ν₀)
- Discount factor (γ)

The goal is to learn a policy that matches expert behavior while being ε-suboptimal with respect to the true (unknown) cost function.

