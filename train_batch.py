import sys
import subprocess
import os
from datetime import datetime

python_executable = sys.executable

tasks = [
    # cuda_device, task, exp_name, restore_name, ground, lateral, overhead, pf_name, term_collision_threshold
    (0, "G1Cat", "debug_", 'none', 0., 0., 0., "empty", 0.0),

]

# exp_name: debug mode if 'debug' in experiment name
# restore_name: '' means training from scratch
# ground, lateral, overhead: reward coefficients for ground, lateral, and overhead obstacles; you can set all to 1. for convenience.
# pf_name: name of the potential field, e.g., 'bar0', 'ceil1'
# term_collision_threshold: the threshold of collision distance to terminate the episode

processes = []

if __name__ == "__main__":
    output_dir = "./output_logs"
    os.makedirs(output_dir, exist_ok=True)
    process_cmd_map = {}
    for cuda_device, task, exp_name, restore_name, ground, lateral, overhead, pf_name, term_collision_threshold in tasks:
        cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} {python_executable} -m train_ppo --task {task} --restore_name {restore_name} --exp_name {exp_name}  --ground {ground} --lateral {lateral} --overhead {overhead} --term_collision_threshold {term_collision_threshold} --pf_name {pf_name}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        stdout_file = os.path.join(output_dir, f"{timestamp}_{cuda_device}_stdout.log")
        stderr_file = os.path.join(output_dir, f"{timestamp}_{cuda_device}_stderr.log")

        with open(stdout_file, "w") as out_file, open(stderr_file, "w") as err_file:
            print(f"Executing: {cmd}")
            out_file.write(f"{cmd}\n")
            err_file.write(f"{cmd}\n")
            process = subprocess.Popen(cmd, shell=True, stdout=out_file, stderr=err_file)
            processes.append(process)
            process_cmd_map[process] = cmd
    while processes:
        for process in processes:
            retcode = process.poll()
            if retcode is not None:
                if retcode != 0:
                    cmd = process_cmd_map[process]
                    print(f"\033[91mReturn code {retcode}.\nCommand: {cmd}\033[0m")
                processes.remove(process)

    print("All tasks completed.")