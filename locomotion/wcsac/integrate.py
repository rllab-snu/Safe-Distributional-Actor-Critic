import numpy as np
import pickle
import glob
import os

run_list = glob.glob('results/*')
for run_idx in range(len(run_list)):
    run_name = run_list[run_idx]

    for item_name in ['score', "cost0", "cost1", "cost2"]:
        log_dir = f"{run_name}/{item_name}_log"
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        n_logs = len(glob.glob(f"{run_name}/{item_name}_*_log"))
        n_pickles = len(glob.glob(f"{run_name}/{item_name}_0_log/*.pkl"))

        for pickle_idx in range(n_pickles):
            data_list = []
            for log_idx in range(n_logs):
                with open(f"{run_name}/{item_name}_{log_idx}_log/record_{pickle_idx:02d}.pkl", "rb") as f:
                    data_list.append(pickle.load(f))
            new_data_list = []
            for data_idx in range(len(data_list[0])):
                for log_idx in range(n_logs):
                    new_data_list.append(data_list[log_idx][data_idx])
            with open(f"{run_name}/{item_name}_log/record_{pickle_idx:02d}.pkl", "wb") as f:
                pickle.dump(new_data_list, f)
