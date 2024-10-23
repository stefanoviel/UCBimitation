#!/bin/bash

# Define the parameters
seeds=(3 22 42 64 99 123 456 789 999 1024)  # List of different seeds
num_of_NNs=(1 2 3 5 10)          # List of different --num-of-NNs values

# Function to generate and run the command
run_command() {
    seed=$1
    nn=$2
    command="python -m ilarl_nn.main --env-name DiscreteGaussianGridworld-v0 \
      --expert-trajs assets/envDiscreteGaussianGridworld-v0type1noiseE0.0/expert_trajs/trajs16.pkl \
      --max-iter-num 100 --grid-type 1 --noiseE 0.0 --seed $seed --num-of-NNs $nn --log-dir runs_memory_replay --use-memory-replay --z-std-multiplier 1.0"
    
    echo "Running: $command"
    eval $command
}

# Run commands in parallel
for seed in "${seeds[@]}"; do
    for nn in "${num_of_NNs[@]}"; do
        run_command $seed $nn &
    done
done

# Wait for all background processes to finish
wait

echo "All processes completed."
