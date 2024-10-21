import csv
import os
import fcntl
from datetime import datetime

def safe_write_csv(file_path, data, fieldnames):
    mode = 'a' if os.path.exists(file_path) else 'w'
    with open(file_path, mode) as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        writer.writerow(data)
        fcntl.flock(f, fcntl.LOCK_UN)

def prepare_csv_data(args, all_true_rewards):
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    data_to_write = []
    for i in range(args.max_iter_num):
        row = {
            'iteration': i,
            'true_reward': all_true_rewards[i],
            'run_id': run_id
        }
        row.update({arg: str(value) for arg, value in vars(args).items()})
        data_to_write.append(row)
    return data_to_write, list(data_to_write[0].keys())

def save_results(args, all_true_rewards):
    data_to_write, fieldnames = prepare_csv_data(args, all_true_rewards)
    log_file_path = os.path.join(args.log_dir, "true_rewards.csv") if args.log_dir else "runs/true_rewards.csv"
    safe_write_csv(log_file_path, data_to_write, fieldnames)
    print(f"True rewards and run parameters saved to {log_file_path}")

