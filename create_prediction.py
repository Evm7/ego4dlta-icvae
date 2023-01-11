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

def save_results(src, dst):
    shutil.copyfile(src, dst)

def read_json(filename):
  with open(filename) as jsonFile:
    data = json.load(jsonFile)
    jsonFile.close()
  return data

def create_prediction(split, filename):
    src = 'outputs/{}_lta.json'.format(split)
    dst = filename
    shutil.copyfile(src, dst)

def make_dir(version_h3m, version_intention, version_icvae):
    dir_name = DIR_PATH + '/results/h3m_{}_int_{}_icvae_{}'.format(version_h3m, version_intention, version_icvae)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    return dir_name +"/"

def check_compatibility(cfg_icvae, version_intent):
    hparams_file = DIR_PATH +'/lightning_logs/version_{}/hparams.yaml'.format(version_intent)
    cfg_file = adaptLoader(hparams_file)
    cfg_intent = load_default_config(args, cfg_file)

    # Then we check if there is correspondence with Configurations
    if cfg_icvae.FORECASTING.NUM_INPUT_CLIPS != cfg_intent.FORECASTING.NUM_INPUT_CLIPS:
        print("Models are not compatible in NUM_INPUT_CLIPS: H3M_intention - {} | ICVAE - {}".format(cfg_intent.FORECASTING.NUM_INPUT_CLIPS, cfg_icvae.FORECASTING.NUM_INPUT_CLIPS))
        return False
    elif cfg_icvae.CVAE.use_intention:
        if cfg_icvae.CVAE.num_intentions != cfg_intent.MODEL.NUM_CLASSES[1][0]:
            print("Models are not compatible in NUM_INTENTIONS: H3M_intention - {} | ICVAE - {}".format(cfg_intent.MODEL.NUM_CLASSES[1][0], cfg_icvae.CVAE.num_intentions))
            return False
    return True

def combine_predictions(version_h3m, version_intention, test_split):
    h3m_data = read_json(DIR_PATH + "/lightning_logs/version_{}/{}_prediction.json".format(version_h3m, test_split))
    int_data = read_json(DIR_PATH + "/lightning_logs/version_{}/{}_prediction.json".format(version_intention, test_split))

    resulting_data = {"intention_preds": int_data["intention_preds"],
                      "verbs_preds": h3m_data["verbs_preds"],
                      "nouns_preds":h3m_data["nouns_preds"]}

    comb_filepath = DIR_PATH +'outputs/{}_pred_comb.json'.format(test_split)
    json.dump(resulting_data, open(comb_filepath, 'w'))
    return comb_filepath

if __name__ == "__main__":
    ##
    # Two parameters need to be modified in order to create a correct prediction:
    # Version ICVAE and Version H3M
    ##
    version_h3m =20 # used for verbs and nouns
    version_intention =  1 # used for intention
    version_icvae = 122

    args = parse_args()
    hparams_file = DIR_PATH +'/lightning_logs/version_{}/hparams.yaml'.format(version_icvae)
    cfg_file = adaptLoader(hparams_file)
    cfg = load_default_config(args, cfg_file)

    #Checking the compatibility
    if not check_compatibility(cfg, version_intention):
        sys.exit(0)


    cfg.TRAIN.ENABLE = False
    cfg.TEST.ONLY_TESTING = True
    best=True
    cfg.TEST.FROM_PREDICTION = True
    cfg.TRAIN.BATCH_SIZE = 32
    cfg.TEST.BATCH_SIZE = 32


    dir_name_icvae = DIR_PATH +"/lightning_logs/version_{}/".format(version_icvae)
    dir_name = make_dir(version_h3m,version_intention, version_icvae)
    filenames = glob.glob(dir_name_icvae +"checkpoints/*.ckpt", recursive=True)
    for weights_path in filenames:
        if (best and "last" not in weights_path) or (not best and "last" in weights_path):
            print("#"*25)
            print("[WEIGHTS]... {}".format(weights_path))
            print("#"*25)
            cfg.CHECKPOINT_FILE_PATH = weights_path
            for s in ['val', 'test']:
                cfg.TEST.SPLIT = s
                comb_file_path = combine_predictions(version_h3m, version_intention, s)
                cfg.TEST.OUTPUTS_PATH = comb_file_path

                results = main(cfg)
                if s =="val":
                    save_results(DIR_PATH + "outputs/val_lta_results.json", "{}val_results.json".format(dir_name))
                print("[Prediction]... Creating predictor file for the model in ", s)
                create_prediction(s, "{}{}_lta.json".format(dir_name, cfg.TEST.SPLIT))

