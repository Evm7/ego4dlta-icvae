import sys
import json
import numpy as np

from ego4d.utils import logging
import torch
from ego4d.tasks.ICVAE_Task import ICVAE_Task
from ego4d.tasks.H3M_Task import H3M_Task



from ego4d.utils.parser import load_default_config, parse_args, adaptLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from data.PATHS import DIR_PATH

logger = logging.get_logger(__name__)

import os, glob
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
    type_model = cfg.MODEL.MODEL_NAME
    if type_model == 'ICVAE':
         TaskType = ICVAE_Task
    elif type_model == 'H3M' or type_model =="ActionClassifier":
        TaskType = H3M_Task
    else:
        print('The task could not be extrapolated from the CFG File. Please review and add the variable [MODEL_NAME]: {}'.format(type_model))
        sys.exit(0)
    print('Testing for the {} model: '.format(type_model))
    task = TaskType(cfg)


    # TODO: LOAD THE MODEL
    # Load model from checkpoint if checkpoint file path is given.
    ckp_path = cfg.CHECKPOINT_FILE_PATH

    checkpoint = torch.load(ckp_path)
    print("#"*100)
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
    print("#"*100)

    args = {"logger": False}
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
        plugins=DDPPlugin(find_unused_parameters=False),
        enable_checkpointing=False, # As we are testing, we do not want to create a new checkpoint dir. Instead, we will write the results obtained later on in the trained directory
        **args,
    )
    print("Test is enabled!")
    print("[INFO] Testing in the {} split!".format(cfg.TEST.SPLIT))
    return trainer.test(task)

def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f)

def read_json(filename):
  with open(filename) as jsonFile:
    data = json.load(jsonFile)
    jsonFile.close()
  return data

def create_prediction(split, filename):
    data = read_json('outputs/{}_predictor.json'.format(split))
    intention_preds = {clip_id: int(np.array(data["intention_preds"]).argmax()) for clip_id, data in data.items()}
    verbs_preds = {clip_id: np.array(data["verbs_preds"]).argmax(axis=1).tolist() for clip_id, data in data.items() if "verbs_preds" in data}
    nouns_preds = {clip_id: np.array(data["nouns_preds"]).argmax(axis=1).tolist() for clip_id, data in data.items()  if "nouns_preds" in data}

    with open(filename, 'w') as f:
        json.dump({"intention_preds": intention_preds,
                   "verbs_preds": verbs_preds,
                   "nouns_preds": nouns_preds}, f)

if __name__ == "__main__":
    args = parse_args()
    versions=list(range(19,20))
    for version in versions:
        hparams_file = DIR_PATH +'/lightning_logs/version_{}/hparams.yaml'.format(version)
        cfg_file = adaptLoader(hparams_file)
        cfg = load_default_config(args, cfg_file)


        cfg.TRAIN.ENABLE = False
        cfg.TEST.ONLY_TESTING = True
        best=True

        dir_name = DIR_PATH +"/lightning_logs/version_{}/".format(version)
        filenames = glob.glob(dir_name +"checkpoints/*.ckpt", recursive=True)
        for weights_path in filenames:
            if (best and "last" not in weights_path) or (not best and "last" in weights_path):
                print("#"*25)
                print("[WEIGHTS]... {}".format(weights_path))
                print("#"*25)
                cfg.CHECKPOINT_FILE_PATH = weights_path
                for s in ['val', 'test']:
                    cfg.TEST.SPLIT = s
                    results = main(cfg)
                    save_results(results, "{}{}_results.json".format(dir_name, cfg.TEST.SPLIT))
                    if cfg.MODEL.MODEL_NAME =="H3M":
                        print("[Prediction]... Creating predictor file for the model in ", s)
                        create_prediction(s, "{}{}_prediction.json".format(dir_name, cfg.TEST.SPLIT))

