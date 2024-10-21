# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Read the CSV file
# df = pd.read_csv('runs_memory_replay/true_rewards.csv')

# # Group by num_of_NNs and run_id, then find the max true_reward for each group
# max_rewards = df.groupby(['num_of_NNs', 'run_id'])['true_reward'].max().reset_index()

# # Calculate the average and standard deviation of max true_rewards for each num_of_NNs
# avg_max_rewards = max_rewards.groupby('num_of_NNs')['true_reward'].agg(['mean', 'std']).reset_index()

# # Sort the results by num_of_NNs
# avg_max_rewards = avg_max_rewards.sort_values('num_of_NNs')

# # Create the plot
# plt.figure(figsize=(12, 7))
# plt.errorbar(avg_max_rewards['num_of_NNs'], avg_max_rewards['mean'], 
#              yerr=avg_max_rewards['std'], fmt='o-', capsize=5, capthick=2, ecolor='red', markersize=8)

# plt.xlabel('Number of Neural Networks')
# plt.ylabel('Average Max True Reward')
# plt.title('Average Max True Reward vs Number of Neural Networks (with Confidence Intervals) - memory replay')
# plt.grid(True)

# # Add value labels on the points
# for x, y, std in zip(avg_max_rewards['num_of_NNs'], avg_max_rewards['mean'], avg_max_rewards['std']):
#     plt.annotate(f'{y:.2f} ± {std:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

# # Show the plot
# plt.tight_layout()
# plt.show()
# plt.savefig('runs_memory_replay/avg_max_true_reward_vs_num_of_NNs.png')



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('runs_memory_replay/true_rewards.csv')

# Calculate the mean reward for each run
mean_rewards = df.groupby(['num_of_NNs', 'run_id'])['true_reward'].mean().reset_index()

# Calculate the average and standard deviation of mean rewards for each num_of_NNs
avg_mean_rewards = mean_rewards.groupby('num_of_NNs')['true_reward'].agg(['mean', 'std']).reset_index()

# Sort the results by num_of_NNs
avg_mean_rewards = avg_mean_rewards.sort_values('num_of_NNs')

# Create the plot
plt.figure(figsize=(12, 7))
plt.errorbar(avg_mean_rewards['num_of_NNs'], avg_mean_rewards['mean'], 
             yerr=avg_mean_rewards['std'], fmt='o-', capsize=5, capthick=2, ecolor='red', markersize=8)

plt.xlabel('Number of Neural Networks')
plt.ylabel('Average Mean True Reward')
plt.title('Average Mean True Reward vs Number of Neural Networks (with Confidence Intervals) - memory replay')
plt.grid(True)

# Add value labels on the points
for x, y, std in zip(avg_mean_rewards['num_of_NNs'], avg_mean_rewards['mean'], avg_mean_rewards['std']):
    plt.annotate(f'{y:.2f} ± {std:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

# Show the plot
plt.tight_layout()
plt.show()
plt.savefig('runs_memory_replay/avg_mean_true_reward_vs_num_of_NNs.png')





# import tensorflow as tf
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import os
# import re

# def extract_tensorboard_data(log_file, tag_name):
#     data = defaultdict(list)
    
#     for event in tf.compat.v1.train.summary_iterator(log_file):
#         for value in event.summary.value:
#             if value.tag == tag_name:
#                 data['step'].append(event.step)
#                 data['value'].append(value.simple_value)
    
#     return pd.DataFrame(data)

# def get_nn_number(folder_name):
#     match = re.search(r'nn(\d+)', folder_name)
#     return int(match.group(1)) if match else 0

# # Specify the path to your TensorBoard log file
# main_folder = 'runs_memory_replay'

# # Specify the tag name of the parameter you want to extract
# tag_name = 'Reward/Mean True Reward'


# all_data = []

# # Iterate through all subdirectories
# for subdir in os.listdir(main_folder):
#     subdir_path = os.path.join(main_folder, subdir)
#     if os.path.isdir(subdir_path):
#         # Find the events file in the subdirectory
#         for file in os.listdir(subdir_path):
#             if file.startswith('events.out.tfevents'):
#                 log_file = os.path.join(subdir_path, file)
#                 df = extract_tensorboard_data(log_file, tag_name)
#                 df['nn'] = get_nn_number(subdir)
#                 all_data.append(df)

# # Combine all dataframes
# combined_df = pd.concat(all_data, ignore_index=True)

# # Group by nn and step, then calculate the mean
# averaged_df = combined_df.groupby(['nn', 'step'])['value'].mean().reset_index()

# print(averaged_df)

# # Plot the averages for each nn
# plt.figure(figsize=(12, 8))
# for nn in averaged_df['nn'].unique():
#     nn_data = averaged_df[averaged_df['nn'] == nn]
#     plt.plot(nn_data['step'], nn_data['value'], label=f'NN {nn}')

# plt.xlabel('Step')
# plt.ylabel('Average True Reward in the episode')
# plt.title(f'Average {tag_name} over time for different NN counts')
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig(f'runs_memory_replay/Average true reward over time for different NN counts')