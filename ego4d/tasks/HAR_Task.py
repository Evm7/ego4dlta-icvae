import torch
import pdb
import numpy as np
import itertools
from fvcore.nn.precise_bn import get_bn_modules
import json
import pprint

from ..evaluation import lta_metrics as metrics
from ..utils import distributed as du
from ..utils import logging
from ..utils import misc
from ..tasks.video_task import VideoTask
from ..models.losses import FocalLoss

logger = logging.get_logger(__name__)


class HAR_Task(VideoTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.checkpoint_metric = "val_top1_noun_err"
        self.action_loss =  self.cfg.MLPMixer.action_loss

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        labels, vision_features = batch["labels"], batch["vision_features"]

        preds = self.forward(vision_features)

        loss1 = self.loss_fun["cross_entropy"](preds[0], labels[:, 0])
        loss2 = self.loss_fun["cross_entropy"](preds[1], labels[:, 1])
        loss = loss1 + loss2
        top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
            preds[0], labels[:, 0], (1, 5)
        )
        top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
            preds[1], labels[:, 1], (1, 5)
        )

        step_result = {
            "loss": loss,
            "train_loss": loss.item(),
            "train_top1_verb_err": top1_err_verb.item(),
            "train_top5_verb_err": top5_err_verb.item(),
            "train_top1_noun_err": top1_err_noun.item(),
            "train_top5_noun_err": top5_err_noun.item(),
        }

        return step_result

    def training_epoch_end(self, outputs):
        if self.cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(self.model)) > 0:
            misc.calculate_and_update_precise_bn(
                self.train_loader, self.model, self.cfg.BN.NUM_BATCHES_PRECISE
            )
        _ = misc.aggregate_split_bn_stats(self.model)

        keys = [x for x in outputs[0].keys() if x != "loss"]
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def validation_step(self, batch, batch_idx):
        labels, vision_features = batch["labels"], batch["vision_features"]

        preds = self.forward(vision_features)

        top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
            preds[0], labels[:, 0], (1, 5)
        )
        top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
            preds[1], labels[:, 1], (1, 5)
        )
        return {
            "val_top1_verb_err": top1_err_verb.item(),
            "val_top5_verb_err": top5_err_verb.item(),
            "val_top1_noun_err": top1_err_noun.item(),
            "val_top5_noun_err": top5_err_noun.item(),
        }

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def test_step(self, batch, batch_idx):
        labels, vision_features, clip_id = batch["labels"], batch["vision_features"], batch["clip_id"]
        preds = self.forward(vision_features)

        return {
            "preds_verb": preds[0],
            "preds_noun": preds[1],
            "labels": labels,
            "clip_ids": clip_id,
        }

    def test_epoch_end(self, outputs):
        preds_verbs = torch.cat([x["preds_verb"] for x in outputs])
        preds_nouns = torch.cat([x["preds_noun"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        clip_ids = [x["clip_ids"] for x in outputs]
        clip_ids = [item for sublist in clip_ids for item in sublist]

        # Gather all labels from distributed processes.
        preds_verbs = torch.cat(du.all_gather([preds_verbs]), dim=0)
        preds_nouns = torch.cat(du.all_gather([preds_nouns]), dim=0)
        labels = torch.cat(du.all_gather([labels]), dim=0)
        clip_ids = list(itertools.chain(*du.all_gather_unaligned(clip_ids)))

        # Ensemble multiple predictions of the same view together. This relies on the
        # fact that the dataloader reads multiple clips of the same video at different
        # spatial crops.
        video_labels = {}
        video_verb_preds = {}
        video_noun_preds = {}
        assert preds_verbs.shape[0] == preds_nouns.shape[0]
        for i in range(preds_verbs.shape[0]):
            vid_id = clip_ids[i]
            video_labels[vid_id] = labels[i]
            if vid_id not in video_verb_preds:
                video_verb_preds[vid_id] = torch.zeros(
                    (self.cfg.MODEL.NUM_CLASSES[0]),
                    device=preds_verbs.device,
                    dtype=preds_verbs.dtype,
                )
                video_noun_preds[vid_id] = torch.zeros(
                    (self.cfg.MODEL.NUM_CLASSES[1]),
                    device=preds_nouns.device,
                    dtype=preds_nouns.dtype,
                )

            if self.cfg.DATA.ENSEMBLE_METHOD == "sum":
                video_verb_preds[vid_id] += preds_verbs[i]
                video_noun_preds[vid_id] += preds_nouns[i]
            elif self.cfg.DATA.ENSEMBLE_METHOD == "max":
                video_verb_preds[vid_id] = torch.max(
                    video_verb_preds[vid_id], preds_verbs[i]
                )
                video_noun_preds[vid_id] = torch.max(
                    video_noun_preds[vid_id], preds_nouns[i]
                )

        if du.get_local_rank() == 0:
            output_dict ={k:{"label": video_labels[k].cpu().tolist(), "verbs_preds" : video_verb_preds[k].cpu().tolist(),
                             "nouns_preds" : video_noun_preds[k].cpu().tolist() } for k in video_labels.keys()}
            json.dump(output_dict, open('outputs/{}_action.json'.format(self.cfg.TEST.SPLIT), 'w'))

        video_verb_preds = torch.stack(list(video_verb_preds.values()), dim=0)
        video_noun_preds = torch.stack(list(video_noun_preds.values()), dim=0)
        video_labels = torch.stack(list(video_labels.values()), dim=0)
        top1_verb_err, top5_verb_err = metrics.topk_errors(
            video_verb_preds, video_labels[:, 0], (1, 5)
        )
        top1_noun_err, top5_noun_err = metrics.topk_errors(
            video_noun_preds, video_labels[:, 1], (1, 5)
        )
        errors = {
            "top1_verb_err": top1_verb_err,
            "top5_verb_err": top5_verb_err,
            "top1_noun_err": top1_noun_err,
            "top5_noun_err": top5_noun_err,
        }
        for k, v in errors.items():
            self.log(k, v.item())