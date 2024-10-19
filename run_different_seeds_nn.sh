#!/bin/bash

# Define the parameters
seeds=(22 42 64 128)          # List of different seeds
num_of_NNs=(2 5 10)          # List of different --num-of-NNs values

# Loop through each seed and NN value
for seed in "${seeds[@]}"; do
  for nn in "${num_of_NNs[@]}"; do
    # Construct the command
    command="python -m train_learner.ilarl_nn --env-name DiscreteGaussianGridworld-v0 \
      --expert-trajs assets/envDiscreteGaussianGridworld-v0type1noiseE0.0/expert_trajs/trajs16.pkl \
      --max-iter-num 100 --grid-type 1 --noiseE 0.0 --seed $seed --num-of-NNs $nn --log-dir run_memory_replay --use-memory-replay"
    
    # Print the command (optional)
    echo "Running: $command"
    
    # Execute the command
    eval $command
  done
done

