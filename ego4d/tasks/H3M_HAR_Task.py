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


class H3M_HAR_Task(VideoTask):
    checkpoint_metric = "val_top1_noun_err"

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        print("[TRAINING STEP]: This model should only be used for Testing")
        return {}

    def training_epoch_end(self, outputs):
        return None

    def validation_step(self, batch, batch_idx):
        print("[VALIDATION STEP]: This model should only be used for Testing")
        return {}

    def validation_epoch_end(self, outputs):
        return None

    def test_step(self, batch, batch_idx):
        intention_labels, vision_features, clip_id, action_labels = batch["intentions"], batch["vision_features"],\
                                                                    batch["clip_id"], batch["observed_labels"]
        intention_pred, actions_pred = self.forward(vision_features)
        return {
            "preds_intentions": intention_pred,
            "intention_labels": intention_labels,
            "action_labels": action_labels,
            "clip_ids": clip_id,
            "preds_verb": actions_pred[0],
            "preds_noun": actions_pred[1]
        }

    def test_epoch_end(self, outputs):
        preds_intentions = torch.cat([x["preds_intentions"] for x in outputs])
        intention_labels = torch.cat([x["intention_labels"] for x in outputs])

        preds_verbs = torch.cat([x["preds_verb"] for x in outputs])
        preds_nouns = torch.cat([x["preds_noun"] for x in outputs])
        action_labels = torch.cat([x["action_labels"] for x in outputs])

        clip_ids = [x["clip_ids"] for x in outputs]
        clip_ids = [item for sublist in clip_ids for item in sublist]

        # Gather all labels from distributed processes.
        preds_intentions = torch.cat(du.all_gather([preds_intentions]), dim=0)
        intention_labels = torch.cat(du.all_gather([intention_labels]), dim=0)
        preds_verbs = torch.cat(du.all_gather([preds_verbs]), dim=0)
        preds_nouns = torch.cat(du.all_gather([preds_nouns]), dim=0)
        action_labels = torch.cat(du.all_gather([action_labels]), dim=0)
        clip_ids = list(itertools.chain(*du.all_gather_unaligned(clip_ids)))

        # Ensemble multiple predictions of the same view together. This relies on the
        # fact that the dataloader reads multiple clips of the same video at different
        # spatial crops.
        video_intention_labels = {}
        video_intentions_preds = {}

        video_action_labels = {}
        video_verb_preds = {}
        video_noun_preds = {}
        for i in range(preds_intentions.shape[0]):
            vid_id = clip_ids[i]
            video_intention_labels[vid_id] = intention_labels[i]
            video_action_labels[vid_id] = action_labels[i]
            if vid_id not in video_intentions_preds:
                video_intentions_preds[vid_id] = torch.zeros(
                    (self.cfg.MLPMixer.num_intentions),
                    device=preds_intentions.device,
                    dtype=preds_intentions.dtype,
                )
                video_verb_preds[vid_id] = torch.zeros(
                    (self.cfg.FORECASTING.NUM_INPUT_CLIPS, self.cfg.MODEL.NUM_CLASSES[0][0]),
                    device=preds_verbs.device,
                    dtype=preds_verbs.dtype,
                )
                video_noun_preds[vid_id] = torch.zeros(
                    (self.cfg.FORECASTING.NUM_INPUT_CLIPS, self.cfg.MODEL.NUM_CLASSES[0][1]),
                    device=preds_nouns.device,
                    dtype=preds_nouns.dtype,
                )
            if self.cfg.DATA.ENSEMBLE_METHOD == "sum":
                video_intentions_preds[vid_id] += preds_intentions[i]
                video_verb_preds[vid_id] += preds_verbs[i]
                video_noun_preds[vid_id] += preds_nouns[i]
            elif self.cfg.DATA.ENSEMBLE_METHOD == "max":
                video_intentions_preds[vid_id] = torch.max(
                    video_intentions_preds[vid_id], preds_intentions[i]
                )
                video_verb_preds[vid_id] = torch.max(
                    video_verb_preds[vid_id], preds_verbs[i]
                )
                video_noun_preds[vid_id] = torch.max(
                    video_noun_preds[vid_id], preds_nouns[i]
                )




        if du.get_local_rank() == 0:
            output_dict ={k:{"intention_labels": video_intention_labels[k].cpu().tolist(),
                             "verbs_preds" : video_verb_preds[k].cpu().tolist(),
                             "nouns_preds" : video_noun_preds[k].cpu().tolist(),
                             "intention_preds":video_intentions_preds[k].cpu().tolist(),
                             "action_labels": video_action_labels[k].cpu().tolist(),
                          } for k in video_intention_labels.keys()}
            json.dump(output_dict, open('outputs/{}_predictor.json'.format(self.cfg.TEST.SPLIT), 'w'))


        video_intentions_preds = torch.stack(list(video_intentions_preds.values()), dim=0)
        video_intention_labels = torch.stack(list(video_intention_labels.values()), dim=0)
        top1_intention_err, top5_intention_err = metrics.topk_errors(
            video_intentions_preds, video_intention_labels, (1, 5)
        )
        errors = {
            "top1_intention_err": top1_intention_err,
            "top5_intention_err": top5_intention_err,
        }
        for k, v in errors.items():
            self.log(k, v.item())