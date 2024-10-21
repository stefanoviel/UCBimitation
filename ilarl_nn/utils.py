import csv
import os
import fcntl
from datetime import datetime

def safe_write_csv(file_path, data, fieldnames):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', newline='') as csvfile:  # Changed to 'a' for append mode
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.path.getsize(file_path) == 0:  # Write header only if file is empty
            writer.writeheader()
        writer.writerows(data)

def prepare_csv_data(args, all_true_rewards):
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    data_to_write = []
    for i, reward in enumerate(all_true_rewards):
        row = {
            'iteration': i,
            'true_reward': reward,  # Changed 'reward' to 'true_reward'
            'run_id': run_id
        }
        row.update({arg: str(getattr(args, arg)) for arg in vars(args)})
        data_to_write.append(row)
    return data_to_write, list(data_to_write[0].keys())

def save_results(args, all_true_rewards):
    log_file_path = os.path.join(args.log_dir, "true_rewards.csv")
    
    data_to_write, fieldnames = prepare_csv_data(args, all_true_rewards)
    
    safe_write_csv(log_file_path, data_to_write, fieldnames)
    print(f"Results appended to {log_file_path}")
