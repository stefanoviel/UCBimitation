#!/bin/bash

# Define the parameters
seeds=(22 42 64 128 256 512 1024 2048)          # List of different seeds
num_of_NNs=(2 5 10)          # List of different --num-of-NNs values

# Function to generate and run the command
run_command() {
    seed=$1
    nn=$2
    command="python -m train_learner.ilarl_nn --env-name DiscreteGaussianGridworld-v0 \
      --expert-trajs assets/envDiscreteGaussianGridworld-v0type1noiseE0.0/expert_trajs/trajs16.pkl \
      --max-iter-num 100 --grid-type 1 --noiseE 0.0 --seed $seed --num-of-NNs $nn --log-dir run_memory_replay --use-memory-replay"
    
    echo "Running: $command"
    eval $command
}

export -f run_command

# Use parallel to run the commands
parallel run_command ::: "${seeds[@]}" ::: "${num_of_NNs[@]}"
