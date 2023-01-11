import math, torch

from ..models import losses
from ..optimizers import lr_scheduler
from ..utils import distributed as du
from ..utils import logging
from ..datasets import loader
from ..models import build_model
from pytorch_lightning.core import LightningModule

logger = logging.get_logger(__name__)


class VideoTask(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # Backwards compatibility.
        if isinstance(cfg.MODEL.NUM_CLASSES, int):
            cfg.MODEL.NUM_CLASSES = [cfg.MODEL.NUM_CLASSES]

        if not hasattr(cfg.TEST, "NO_ACT"):
            logger.info("Default NO_ACT")
            cfg.TEST.NO_ACT = False

        if not hasattr(cfg.MODEL, "TRANSFORMER_FROM_PRETRAIN"):
            cfg.MODEL.TRANSFORMER_FROM_PRETRAIN = False

        if not hasattr(cfg.MODEL, "STRIDE_TYPE"):
            cfg.EPIC_KITCHEN.STRIDE_TYPE = "constant"

        self.cfg = cfg
        self.save_hyperparameters()
        self.model = build_model(cfg)
        self.losses_params = self.cfg.MODEL.lambdas
        if not self.cfg.CVAE.use_reconstruction:
            self.losses_params.pop("l2")


        if self.cfg.CVAE.weighted_loss and self.cfg.TRAIN.ENABLE:
            print("[INFO] Using Imbalanced weighted Focal Loss for crossentropy training")
            samples_per_class = torch.load(self.cfg.DATA.FOLDER + "/samples_per_cls.pt")
            self.loss_fun ={k: losses.get_loss_func(k) for k in  self.cfg.MODEL.losses}
            self.loss_fun["cross_entropy"] = {}
            for i, l in enumerate(["verbs", "nouns", "intention"]):
                self.loss_fun["cross_entropy"][l] = losses.FocalLoss(
                                                beta = 0.9, #0.9, 0.99, 0.999, 0.9999
                                                samples_per_cls= list(samples_per_class[l].values()),
                                                no_of_classes = self.cfg.MODEL.NUM_CLASSES[1],
                                                device = "cuda:0"
                                        )

        else:
            self.loss_fun ={k: losses.get_loss_func(k) for k in  self.cfg.MODEL.losses}

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step_end(self, training_step_outputs):
        if self.cfg.SOLVER.ACCELERATOR == "dp":
            training_step_outputs["loss"] = training_step_outputs["loss"].mean()
        return training_step_outputs

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def forward(self, inputs):
        return self.model(inputs)

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def setup(self, stage):
        # Setup is called immediately after the distributed processes have been
        # registered. We can now setup the distributed process groups for each machine
        # and create the distributed data loaders.
        # if not self.cfg.FBLEARNER:
        if self.cfg.SOLVER.ACCELERATOR != "dp":
            du.init_distributed_groups(self.cfg)


        if not self.cfg.TEST.ONLY_TESTING:
            self.train_loader = loader.construct_loader(self.cfg, "train")
            self.val_loader = loader.construct_loader(self.cfg, "val")
        else:
            print("[INFO] Only loading testing dataset")
            if self.cfg.TEST.SPLIT in "val":
                self.val_loader = loader.construct_loader(self.cfg, "val")
        self.test_loader = loader.construct_loader(self.cfg, "test")

    def configure_optimizers(self):
        steps_in_epoch = len(self.train_loader)
        return lr_scheduler.lr_factory(
            self.model, self.cfg, steps_in_epoch, self.cfg.SOLVER.LR_POLICY
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        if self.cfg.TEST.SPLIT in "val":
            return self.val_loader
        elif self.cfg.TEST.SPLIT in "train":
            return self.train_loader
        else:
            return self.test_loader


    def on_after_backward(self):
        if (
            self.cfg.LOG_GRADIENT_PERIOD >= 0
            and self.trainer.global_step % self.cfg.LOG_GRADIENT_PERIOD == 0
        ):
            for name, weight in self.model.named_parameters():
                if weight is not None:
                    self.logger.experiment.add_histogram(
                        name, weight, self.trainer.global_step
                    )
                    if weight.grad is not None:
                        self.logger.experiment.add_histogram(
                            f"{name}.grad", weight.grad, self.trainer.global_step
                        )
