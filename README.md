# UCBimitation

## instructions

inside folder my_gym run: 
`pip install -e .`


Authors' implementation for the experiment in the ICML 2024 paper: **Imitation Learning in Discounted Linear MDPs without exploration assumptions**

Link: https://arxiv.org/abs/2405.02181

To reproduce the results use the following commands (where num-of-NNs is the number of neural networks).

python -m ilarl_nn.main --env-name DiscreteGaussianGridworld-v0 \
      --expert-trajs assets/envDiscreteGaussianGridworld-v0type1noiseE0.0/expert_trajs/trajs16.pkl \
      --max-iter-num 100 --grid-type 1 --noiseE 0.0 --seed $seed --num-of-NNs $nn --log-dir runs_memory_replay_zmul --use-memory-replay --z-std-multiplier 10.0