
import glob
import sys 

from ego4d.utils import logging

import torch
from ego4d.tasks.ICVAE_Task import ICVAE_Task

from ego4d.utils.parser import load_default_config, parse_args, adaptLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data.PATHS import DIR_PATH

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



def main(cfg):
    seed_everything(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)
    #logger.info("Run with config:")
    #logger.info(pprint.pformat(cfg))

    # Choose task type based on config.
    # TODO: change this to TASK_REGISTRY.get(cfg.cfg.DATA.TASK)(cfg)
    TaskType = ICVAE_Task
    task = TaskType(cfg)


    # TODO: LOAD THE MODEL
    # Load model from checkpoint if checkpoint file path is given.
    ckp_path = cfg.CHECKPOINT_FILE_PATH

    checkpoint = torch.load(ckp_path)
    print("#"*50)
    if "model_state" in checkpoint.keys():
        pre_train_dict = checkpoint["model_state"]
        print(task.model.load_state_dict(pre_train_dict, strict=False))
        print(f"Checkpoint {ckp_path} loaded")
    elif "state_dict" in checkpoint.keys():
        pre_train_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        print(task.model.load_state_dict(pre_train_dict, strict=False))
        print(f"Checkpoint {cfg.CHECKPOINT_FILE_PATH} loaded")
    else:
        print("[URGENT] No checkpoint loaded! ")
        print(checkpoint.keys())
    print("#"*50)

    checkpoint_callback = ModelCheckpoint(
        monitor=task.checkpoint_metric, mode="min", save_last=True, save_top_k=1
    )
    if cfg.ENABLE_LOGGING:
        args = {"callbacks": [LearningRateMonitor(), checkpoint_callback]}
    else:
        args = {"logger": False, "callbacks": checkpoint_callback}

    # TODO: verify the use of the dataset/dataloader
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
        #plugins=DDPPlugin(find_unused_parameters=False),
        **args,
    )

    print("Test is enabled!")
    print("[INFO] Testing in the {} split!".format(cfg.TEST.SPLIT))
    return trainer.test(task)

if __name__ == "__main__":
    args = parse_args()

    version=119

    hparams_file = DIR_PATH +'/lightning_logs/version_{}/hparams.yaml'.format(version)
    cfg_file = adaptLoader(hparams_file)
    cfg = load_default_config(args, cfg_file)
    cfg.TRAIN.ENABLE = False
    cfg.TEST.SPLIT = "test"
    cfg.TEST.ONLY_TESTING = True
    cfg.TEST.FROM_PREDICTION = True
    filenames = glob.glob( DIR_PATH +"/lightning_logs/version_{}/checkpoints/*.ckpt".format(version), recursive=True)
    best=True
    for weights_path in filenames:
        if (best and "last" not in weights_path) or (not best and "last" in weights_path):
            print("#"*25)
            print("[WEIGHTS]... {}".format(weights_path))
            print("#"*25)
            cfg.CHECKPOINT_FILE_PATH = weights_path
            main(cfg)
