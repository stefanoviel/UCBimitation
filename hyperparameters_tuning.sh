#!/bin/bash

# Define the base command
base_cmd="python -m ilarl_nn.main --env-name CartPole-v1 --expert-trajs assets/trajectories_CartPole-v1/expert_trajectories.pkl --max-iter-num 5000 --grid-type 1 --noiseE 0.0 --seed 5 --num-of-NNs 5 --log-dir runs/cartpole --use-memory-replay --z-std-multiplier 1.0 --recompute_rewards"

# Define parameter ranges
etas=(0.005 0.01 0.02)
target_update_freqs=(200 400 600)
buffer_sizes=(400 800 1200)
batch_sizes=(32 64 128)

# Create a temporary file to store the commands
command_file=$(mktemp)

# Generate commands with different parameter combinations
for eta in "${etas[@]}"; do
    for update_freq in "${target_update_freqs[@]}"; do
        for buffer in "${buffer_sizes[@]}"; do
            for batch in "${batch_sizes[@]}"; do
                # Ensure batch size is smaller than buffer size
                if [ "$batch" -lt "$buffer" ]; then
                    command="$base_cmd --eta $eta --target_update_freq $update_freq --buffer-size $buffer --batch-size $batch"
                    echo "Running: $command"
                    eval "$command" &
                fi
            done
        done
    done
done

# Wait for all background processes to finish
wait

# No need for temporary file