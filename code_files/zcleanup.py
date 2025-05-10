import os
import shutil
from config import Config

# utils.py
import torch.nn as nn


def delete_experiment_dirs():
    # Get experiment-specific subdirectory names from config
    experiment_name = Config.EXPERIMENT_NAME
    log_dir = os.path.join(Config.LOG_DIR, experiment_name)
    ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, experiment_name)
    test_res_dir = os.path.join("test_results/", experiment_name)

    deleted = False

    # Delete logs directory
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"[✓] Deleted log directory: {log_dir}")
        deleted = True
    else:
        print(f"[!] Log directory not found: {log_dir}")

    # Delete checkpoints directory
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        print(f"[✓] Deleted checkpoint directory: {ckpt_dir}")
        deleted = True
    else:
        print(f"[!] Checkpoint directory not found: {ckpt_dir}")

    # Delete test_results directory
    if os.path.exists(test_res_dir):
        shutil.rmtree(test_res_dir)
        print(f"[✓] Deleted test_results directory: {ckpt_dir}")
        deleted = True
    else:
        print(f"[!] Test_results directory not found: {ckpt_dir}")

    if not deleted:
        print("[i] Nothing was deleted. No matching experiment found.")

if __name__ == "__main__":
    delete_experiment_dirs()