import os
import pickle
import pprint

import sys

from ego4d.utils import logging
import numpy as np
import pytorch_lightning
import torch
from ego4d.tasks.H3M_HAR_Task import H3M_HAR_Task
from ego4d.utils.c2_model_loading import get_name_convert_func
from ego4d.utils.misc import gpu_mem_usage
from ego4d.utils.parser import load_default_config, parse_args
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from data.PATHS import DIR_PATH

import copy

logger = logging.get_logger(__name__)

import os
import pathlib
import shutil
import submitit


# Not sure why I can't import scripts.slurm?
# from scripts.slurm import copy_and_run_with_config
def init_and_run(run_fn, run_config):
    os.environ["RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["NODE_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    run_fn(run_config)


def copy_and_run_with_config(run_fn, run_config, directory, **cluster_config):
    working_directory = pathlib.Path(directory) / cluster_config["job_name"]
    copy_blacklist = [
        "data",
        "lightning_logs",
        "slurm",
        "logs",
        "pretrained_models",
        "checkpoints",
        "experimental",
        ".git",
        "output",
    ]
    shutil.copytree(".", working_directory, ignore=lambda x, y: copy_blacklist)
    os.chdir(working_directory)
    print(f"Running at {working_directory}")

    executor = submitit.SlurmExecutor(folder=working_directory)
    executor.update_parameters(**cluster_config)
    job = executor.submit(init_and_run, run_fn, run_config)
    print(f"job_id: {job}")

def load_model(task, ckp_path, label='intention'):
    checkpoint = torch.load(ckp_path)
    print("#"*50)
    if "state_dict" in checkpoint.keys():
        pre_train_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        print(task.model.intention_classifier.load_state_dict(pre_train_dict, strict=False))
        print(f"Checkpoint for {label}: {ckp_path} loaded")
    else:
        print(f"[URGENT] No checkpoint loaded for {label} classifier! ")
        print(checkpoint.keys())
    print("#"*50)
    return task

def main(cfg):
    seed_everything(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)

    # Choose task type based on config.
    TaskType = H3M_HAR_Task
    task = TaskType(cfg)

    # Load model from checkpoint if checkpoint file path is given.
    ckp_path_intention = cfg.CHECKPOINT_FILE_PATH[0]
    ckp_path_action = cfg.CHECKPOINT_FILE_PATH[1]

    # H3M consists on Intention + Action
    task = load_model(task, ckp_path_intention, label='intention')
    task = load_model(task, ckp_path_action, label='action')


    checkpoint_callback = ModelCheckpoint(
        monitor=task.checkpoint_metric, mode="min", save_last=True, save_top_k=1
    )
    
    if cfg.ENABLE_LOGGING:
        args = {"callbacks": [LearningRateMonitor(), checkpoint_callback]}
    else:
        args = {"logger": False, "callbacks": checkpoint_callback}

    trainer = Trainer(
        gpus=cfg.NUM_GPUS,
        num_nodes=cfg.NUM_SHARDS,
        accelerator=cfg.SOLVER.ACCELERATOR,
        max_epochs=cfg.SOLVER.MAX_EPOCH,
        num_sanity_val_steps=3,
        benchmark=True,
        log_gpu_memory="min_max",
        replace_sampler_ddp=False,
        fast_dev_run=cfg.FAST_DEV_RUN,
        default_root_dir=cfg.OUTPUT_DIR,
        plugins=DDPPlugin(find_unused_parameters=False),
        **args,
    )

    print("Test is enabled!")
    print("[INFO] Testing in the {} split!".format(cfg.TEST.SPLIT))
    return trainer.test(task)

if __name__ == "__main__":
    args = parse_args()

    # IN HERE WE WILL ADD THE BEST WEIGHTS FOR INTENTION AND FOR ACTION CLASSIFIER
    v_intention = 37
    v_action = 17

    cfg_file = DIR_PATH + "/data/config/config_H3M_HAR.yaml"
    cfg = load_default_config(args, cfg_file)
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ONLY_TESTING = True

    cfg.CHECKPOINT_FILE_PATH = [DIR_PATH + f"pretrained_models/intention_best_{v_intention}.ckpt", DIR_PATH +f"pretrained_models/action_best_{v_action}.ckpt" ]
    main(cfg)
