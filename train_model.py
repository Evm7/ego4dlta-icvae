import sys
import torch
from ego4d.utils import logging

from ego4d.tasks.ICVAE_Task import  ICVAE_Task
from ego4d.tasks.H3M_Task import H3M_Task
from ego4d.tasks.HAR_Task import HAR_Task
from ego4d.tasks.H3M_HAR_Task import H3M_HAR_Task


from ego4d.utils.parser import load_default_config, parse_args
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ProgressBarBase
from pytorch_lightning.plugins import DDPPlugin
from data.PATHS import DIR_PATH

logger = logging.get_logger(__name__)

import os
import pathlib
import shutil
import submitit

from tqdm import tqdm


class LitProgressBar(ProgressBarBase):

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar


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

    type_model = cfg.MODEL.MODEL_NAME
    if type_model == 'ICVAE':
         TaskType = ICVAE_Task
    elif type_model == 'H3M':
        TaskType = H3M_Task
    else:
        print('The task could not be extracted from the CFG File. Please review and add the variable [MODEL_NAME]: {}'.format(type_model))
        sys.exit(0)
    print('Testing for the {} model: '.format(type_model))
    task = TaskType(cfg)

    ckp_path = cfg.CHECKPOINT_FILE_PATH
    if 'pretrained' in ckp_path:
        print('[INFO] Using Pretrained Weights from ', ckp_path)
        checkpoint = torch.load(ckp_path)
        if "state_dict" in checkpoint.keys():
            pre_train_dict = checkpoint["state_dict"]
            print("#"*25)
            print(task.load_state_dict(pre_train_dict, strict=False))
            print(f"Checkpoint {ckp_path} loaded")
            print("#"*25)

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
        plugins=DDPPlugin(find_unused_parameters=False),  ##default is placed to false
        **args,
    )
    if  cfg.TEST.ENABLE:
        print("Train and Test are enabled!")
        trainer.fit(task)
        return trainer.test()

    else:
        print("Train is enabled!")
        return trainer.fit(task)

if __name__ == "__main__":
    args = parse_args()
    totrain = 'H3M' # Choose between [ICVAE, H3M]
    finetuning = False


    cfg_file = DIR_PATH + "/data/config/config_{}.yaml".format(totrain)
    cfg = load_default_config(args, cfg_file)
    cfg.TRAIN.ENABLE = True

    # If interested in finetuning a model, please add here the path of weights to start.
    if finetuning:
        cfg.CHECKPOINT_FILE_PATH = DIR_PATH + "/pretrained_models/multitask_recognition.pt" # used for the action recognition

    main(cfg)
