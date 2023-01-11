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


class H3M_Task(VideoTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        print(self.model)
        self.only_recognition = cfg.TRAIN.DATASET == "Ego4dRecognition_Features"
        self.h3m_used = cfg.MLPMixer.h3m_used

        self.checkpoint_metric =  "val_top1_err_intention" if (not self.only_recognition and self.h3m_used) else "val_top1_noun_err"
        print(self.checkpoint_metric)

        self.noise_injection_mean = cfg.MLPMixer.noise_injection_mean
        self.noise_injection_std = cfg.MLPMixer.noise_injection_std
        self.noise_injection_factor = cfg.MLPMixer.noise_injection_factor
        self.action_loss =  self.cfg.MLPMixer.action_loss
        self.multitask_head_bool = cfg.MLPMixer.multitask_head



        self.loss_weights = {"intention": cfg.MLPMixer.weight_loss_intention, "action": cfg.MLPMixer.weight_loss_action}

        if cfg.MLPMixer.imbalanced:
            samples_per_cls = torch.load(cfg.DATA.FOLDER+ "/samples_per_cls.pt")
            self.loss_fun["cross_entropy"] = FocalLoss(beta = cfg.MLPMixer.beta,
                                                       samples_per_cls = samples_per_cls,
                                                       no_of_classes = cfg.MLPMixer.num_intentions,
                                                       reduction="mean")
        if self.action_loss:
            self.loss_fun["action_loss"] = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, inputs):
        return self.model(inputs)

    def addNoise(self, x):
        return  x + self.noise_injection_factor * (self.noise_injection_std * torch.randn(x.shape, device= x.device) + self.noise_injection_mean)

    def training_step(self, batch, batch_idx):
        if (not self.only_recognition and self.h3m_used):
            intentions_labels = batch["intentions"]
        vision_features, observed_labels = batch["vision_features"], batch["observed_labels"]
        torch.save(batch, "/home/evallsmascaro/PycharmProjects/GoalConditionedForecasting/outputs/batch_{}.pt".format(batch_idx) )

        vision_features = self.addNoise(vision_features)

        # This means that we are using Recognition Dataset
        if self.only_recognition:
            preds = self.forward(vision_features)

            loss1 = self.loss_fun["cross_entropy"](preds[0], observed_labels[:, 0])
            loss2 = self.loss_fun["cross_entropy"](preds[1], observed_labels[:, 1])
            loss = loss1 + loss2
            top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
                preds[0], observed_labels[:, 0], (1, 5)
            )
            top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
                preds[1], observed_labels[:, 1], (1, 5)
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
        else:
            if self.action_loss:
                verb_losses, top_1_err_verbs, top_5_err_verbs = 0, 0, 0
                noun_losses, top_1_err_nouns, top_5_err_nouns = 0, 0, 0

                if self.h3m_used:
                    preds, (verbs, nouns) =  self.forward(vision_features)
                else:
                    verbs, nouns = self.forward(vision_features)

                B, N, _ = verbs.shape
                for i in range(N):
                    verb_losses += self.loss_fun["action_loss"](verbs[:,i,:], observed_labels[:,i, 0])
                    noun_losses += self.loss_fun["action_loss"](nouns[:,i,:], observed_labels[:,i, 1])

                    top1_err_v, top5_err_v = metrics.distributed_topk_errors(
                        verbs[:,i,:], observed_labels[:,i,  0], (1, 5)
                    )
                    top_1_err_verbs += top1_err_v
                    top_5_err_verbs += top5_err_v

                    top1_err_n, top5_err_nn = metrics.distributed_topk_errors(
                        nouns[:,i,:], observed_labels[:,i, 1], (1, 5)
                    )

                    top_1_err_nouns += top1_err_n
                    top_5_err_nouns += top5_err_nn

                verb_losses = verb_losses / N
                noun_losses = noun_losses / N
                top_1_err_verbs = top_1_err_verbs / N
                top_5_err_verbs = top_5_err_verbs / N
                top_1_err_nouns = top_1_err_nouns / N
                top_5_err_nouns = top_5_err_nouns / N

                if self.h3m_used:
                    intention_loss = self.loss_fun["cross_entropy"](preds, intentions_labels)
                    top1_err_intention, top5_err_intention = metrics.distributed_topk_errors(
                        preds, intentions_labels, (1, 5)
                    )
                    loss = intention_loss * self.loss_weights['intention'] + (verb_losses + noun_losses)* self.loss_weights['action']
                else:
                    loss = verb_losses + noun_losses

                step_result = {
                    "loss": loss,
                    "train_loss": loss.item(),
                    "train_verb_loss" : verb_losses.item(),
                    "train_nouns_loss" : noun_losses.item(),
                    "train_top1_verb_err": top_1_err_verbs.item(),
                    "train_top5_verb_err": top_5_err_verbs.item(),
                    "train_top1_noun_err": top_1_err_nouns.item(),
                    "train_top5_noun_err": top_5_err_nouns.item(),
                }

                if self.h3m_used:
                    step_result.update({
                        "train_loss_intention": loss.item(),
                        "train_intention_loss": intention_loss.item(),
                        "top1_err_intention": top1_err_intention.item(),
                        "top5_err_intention": top5_err_intention.item(),
                    })

            else:
                preds = self.forward(vision_features)
                loss = self.loss_fun["cross_entropy"](preds, intentions_labels)
                top1_err_intention, top5_err_intention = metrics.distributed_topk_errors(
                    preds, intentions_labels, (1, 5)
                )

                step_result = {
                    "loss": loss,
                    "train_loss_intention": loss.item(),
                    "top1_err_intention": top1_err_intention.item(),
                    "top5_err_intention": top5_err_intention.item(),
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
        if (not self.only_recognition and self.h3m_used):
            intentions_labels = batch["intentions"]
        vision_features, observed_labels = batch["vision_features"], batch["observed_labels"]

        if self.only_recognition:
            preds = self.forward(vision_features)

            verb_losses = self.loss_fun["cross_entropy"](preds[0], observed_labels[:, 0])
            noun_losses = self.loss_fun["cross_entropy"](preds[1], observed_labels[:, 1])
            loss = verb_losses + noun_losses
            top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
                preds[0], observed_labels[:, 0], (1, 5)
            )
            top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
                preds[1], observed_labels[:, 1], (1, 5)
            )

            step_result = {
                "val_loss_intention": loss.item(),
                "val_verb_loss" : verb_losses.item(),
                "val_nouns_loss" : noun_losses.item(),
                "val_top1_verb_err": top1_err_verb.item(),
                "val_top5_verb_err": top5_err_verb.item(),
                "val_top1_noun_err": top1_err_noun.item(),
                "val_top5_noun_err": top5_err_noun.item(),
            }
            return step_result

        if self.action_loss:
            verb_losses, top_1_err_verbs, top_5_err_verbs = 0, 0, 0
            noun_losses, top_1_err_nouns, top_5_err_nouns = 0, 0, 0

            if self.h3m_used:
                preds, (verbs, nouns) = self.forward(vision_features)
            else:
                verbs, nouns = self.forward(vision_features)

            B, N, _ = verbs.shape
            for i in range(N):
                verb_losses += self.loss_fun["action_loss"](verbs[:, i, :], observed_labels[:, i, 0])
                noun_losses += self.loss_fun["action_loss"](nouns[:, i, :], observed_labels[:, i, 1])

                top1_err_v, top5_err_v = metrics.distributed_topk_errors(
                    verbs[:, i, :], observed_labels[:, i, 0], (1, 5)
                )
                top_1_err_verbs += top1_err_v
                top_5_err_verbs += top5_err_v

                top1_err_n, top5_err_nn = metrics.distributed_topk_errors(
                    nouns[:, i, :], observed_labels[:, i, 1], (1, 5)
                )

                top_1_err_nouns += top1_err_n
                top_5_err_nouns += top5_err_nn

            verb_losses = verb_losses / N
            noun_losses = noun_losses / N
            top_1_err_verbs = top_1_err_verbs / N
            top_5_err_verbs = top_5_err_verbs / N
            top_1_err_nouns = top_1_err_nouns / N
            top_5_err_nouns = top_5_err_nouns / N

            if self.h3m_used:
                intention_loss = self.loss_fun["cross_entropy"](preds, intentions_labels)
                top1_err_intention, top5_err_intention = metrics.distributed_topk_errors(
                    preds, intentions_labels, (1, 5)
                )

                loss = intention_loss * self.loss_weights['intention'] + (verb_losses + noun_losses) * self.loss_weights[
                    'action']
            else:
                loss = verb_losses + noun_losses


            step_result = {
                "val_loss_intention": loss.item(),
                "val_verb_loss" : verb_losses.item(),
                "val_nouns_loss" : noun_losses.item(),
                "val_top1_verb_err": top_1_err_verbs.item(),
                "val_top5_verb_err": top_5_err_verbs.item(),
                "val_top1_noun_err": top_1_err_nouns.item(),
                "val_top5_noun_err": top_5_err_nouns.item(),
            }

            if self.h3m_used:
                step_result.update(
                    {
                        "val_intention_loss": intention_loss.item(),
                        "val_top1_err_intention": top1_err_intention.item(),
                        "val_top5_err_intention": top5_err_intention.item(),
                    }
                )
        else:
            preds = self.forward(vision_features)

            loss = self.loss_fun["cross_entropy"](preds, intentions_labels)
            top1_err_intention, top5_err_intention = metrics.distributed_topk_errors(
                preds, intentions_labels, (1, 5)
            )
            step_result =  {
                "val_loss": loss.item(),
                "val_top1_err_intention": top1_err_intention.item(),
                "val_top5_err_intention": top5_err_intention.item(),
            }
        return step_result


    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def test_step(self, batch, batch_idx):
        if (not self.only_recognition and self.h3m_used):
            intentions_labels = batch["intentions"]
        action_labels, vision_features, clip_id =  batch["observed_labels"], batch["vision_features"], batch["clip_id"]

        if self.only_recognition:
            preds = self.forward(vision_features)
            return {
                "preds_verbs": preds[0],
                "preds_nouns": preds[1],
                "action_labels": action_labels,
                "clip_ids": clip_id,
            }
        if self.action_loss:
            if self.h3m_used:
                preds, (verbs, nouns) = self.forward(vision_features)
                return {
                    "preds_intentions": preds,
                    "preds_verbs": verbs,
                    "preds_nouns": nouns,
                    "intentions_labels": intentions_labels,
                    "action_labels": action_labels,
                    "clip_ids": clip_id,
                }
            else:
                verbs, nouns = self.forward(vision_features)
                return {
                    "preds_verbs": verbs,
                    "preds_nouns": nouns,
                    "action_labels": action_labels,
                    "clip_ids": clip_id,
                }

        else:
            preds = self.forward(vision_features)
            return {
                "preds_intentions": preds,
                "intentions_labels": intentions_labels,
                "clip_ids": clip_id,
            }


    def test_epoch_end(self, outputs):
        clip_ids = [x["clip_ids"] for x in outputs]
        clip_ids = [item for sublist in clip_ids for item in sublist]
        clip_ids = list(itertools.chain(*du.all_gather_unaligned(clip_ids)))

        if not self.only_recognition and self.h3m_used:
            preds_intentions = torch.cat([x["preds_intentions"] for x in outputs])
            intention_labels = torch.cat([x["intentions_labels"] for x in outputs])
            preds_intentions = torch.cat(du.all_gather([preds_intentions]), dim=0)
            intention_labels = torch.cat(du.all_gather([intention_labels]), dim=0)

        if self.action_loss or self.only_recognition:
            preds_verbs = torch.cat([x["preds_verbs"] for x in outputs])
            preds_nouns = torch.cat([x["preds_nouns"] for x in outputs])
            action_labels = torch.cat([x["action_labels"] for x in outputs])
            preds_verbs = torch.cat(du.all_gather([preds_verbs]), dim=0)
            preds_nouns = torch.cat(du.all_gather([preds_nouns]), dim=0)
            action_labels = torch.cat(du.all_gather([action_labels]), dim=0)


        # Ensemble multiple predictions of the same view together. This relies on the
        # fact that the dataloader reads multiple clips of the same video at different
        # spatial crops.
        if (not self.only_recognition and self.h3m_used):
            video_intention_labels = {}
            video_intentions_preds = {}

        video_action_labels = {}
        video_verb_preds = {}
        video_noun_preds = {}


        num = preds_intentions.shape[0] if ( not self.only_recognition and self.h3m_used) else preds_verbs.shape[0]
        for i in range(num):
            vid_id = clip_ids[i]
            if  not self.only_recognition and self.h3m_used:
                video_intention_labels[vid_id] = intention_labels[i]

            if self.action_loss or self.only_recognition:
                video_action_labels[vid_id] = action_labels[i]


            already_in = vid_id not in video_intentions_preds if (not self.only_recognition and self.h3m_used) else vid_id not in video_verb_preds
            if already_in:
                if (not self.only_recognition and self.h3m_used):
                    video_intentions_preds[vid_id] = torch.zeros(
                        (self.cfg.MLPMixer.num_intentions),
                        device=preds_intentions.device,
                        dtype=preds_intentions.dtype,
                    )

                if self.action_loss or self.only_recognition:
                    size_verb = (self.cfg.FORECASTING.NUM_INPUT_CLIPS, self.cfg.MODEL.NUM_CLASSES[0][0]) if (not self.only_recognition) else self.cfg.MODEL.NUM_CLASSES[0][0]
                    video_verb_preds[vid_id] = torch.zeros(
                        size_verb,
                        device=preds_verbs.device,
                        dtype=preds_verbs.dtype,
                    )
                    size_noun = (self.cfg.FORECASTING.NUM_INPUT_CLIPS, self.cfg.MODEL.NUM_CLASSES[0][1]) if (not self.only_recognition) else self.cfg.MODEL.NUM_CLASSES[0][1]
                    video_noun_preds[vid_id] = torch.zeros(
                        size_noun,
                        device=preds_nouns.device,
                        dtype=preds_nouns.dtype,
                    )

            if self.cfg.DATA.ENSEMBLE_METHOD == "sum":
                if (not self.only_recognition and self.h3m_used):
                    video_intentions_preds[vid_id] += preds_intentions[i]
                if self.action_loss or self.only_recognition:
                    video_verb_preds[vid_id] += preds_verbs[i]
                    video_noun_preds[vid_id] += preds_nouns[i]
            elif self.cfg.DATA.ENSEMBLE_METHOD == "max":
                if (not self.only_recognition and self.h3m_used):
                    video_intentions_preds[vid_id] = torch.max(
                        video_intentions_preds[vid_id], preds_intentions[i]
                    )
                if self.action_loss or self.only_recognition:
                    video_verb_preds[vid_id] = torch.max(
                        video_verb_preds[vid_id], preds_verbs[i]
                    )
                    video_noun_preds[vid_id] = torch.max(
                        video_noun_preds[vid_id], preds_nouns[i]
                    )


        if du.get_local_rank() == 0:
            if self.only_recognition or not self.h3m_used:
                output_dict ={k:{"verbs_preds" : video_verb_preds[k].cpu().tolist(),
                                 "nouns_preds" : video_noun_preds[k].cpu().tolist(),
                                 "action_labels": video_action_labels[k].cpu().tolist(),
                              } for k in video_action_labels.keys()}
            elif self.action_loss:
                output_dict ={k:{"intention_labels": video_intention_labels[k].cpu().tolist(),
                                 "verbs_preds" : video_verb_preds[k].cpu().tolist(),
                                 "nouns_preds" : video_noun_preds[k].cpu().tolist(),
                                 "intention_preds":video_intentions_preds[k].cpu().tolist(),
                                 "action_labels": video_action_labels[k].cpu().tolist(),
                              } for k in video_intention_labels.keys()}
            else:
                output_dict = {k: {"intention_labels": video_intention_labels[k].cpu().tolist(),
                                   "intention_preds": video_intentions_preds[k].cpu().tolist(),
                                   } for k in video_intention_labels.keys()}
            json.dump(output_dict, open('outputs/{}_predictor.json'.format(self.cfg.TEST.SPLIT), 'w'))



        if (not self.only_recognition and self.h3m_used):
            video_intentions_preds = torch.stack(list(video_intentions_preds.values()), dim=0)
            video_intention_labels = torch.stack(list(video_intention_labels.values()), dim=0)
            top1_intention_err, top5_intention_err = metrics.topk_errors(
                video_intentions_preds, video_intention_labels, (1, 5)
            )

        def resultsTop5(top5_preds):
            total, top5_ = 0, 0
            for ex in top5_preds.values():
                for i, l in enumerate(ex["label"]):
                    total += 1
                    if l in ex["preds"][i]:
                        top5_ += 1
            return top5_ / total * 100

        def topk(vector, k=5):
            return vector.argsort()[:, ::-1][:, :k]

        if self.only_recognition:
            top1_intention_err = torch.tensor(0.)
            top5_intention_err = torch.tensor(0.)

        if self.only_recognition and self.cfg.TEST.SPLIT != "test":
            video_verb_preds = torch.stack(list(video_verb_preds.values()), dim=0)
            video_noun_preds = torch.stack(list(video_noun_preds.values()), dim=0)
            video_labels = torch.stack(list(video_action_labels.values()), dim=0)

            top1_verb_err, top5_verb_err = metrics.topk_errors(
                video_verb_preds, video_labels[:, 0], (1, 5)
            )
            top1_noun_err, top5_noun_err = metrics.topk_errors(
                video_noun_preds, video_labels[:, 1], (1, 5)
            )
            errors = {
                "top1_noun_err": top1_noun_err,
                "top5_noun_err": top5_noun_err,
                "top1_verb_err": top1_verb_err,
                "top5_verb_err": top5_verb_err,
                "top1_intention_err": top1_intention_err,
                "top5_intention_err": top5_intention_err,
            }
            for k, v in errors.items():
                self.log(k, v.item())

        elif self.action_loss and self.cfg.TEST.SPLIT != 'test':
            verb_acc_TOP1 = np.array([[np.array(i["verbs_preds"]).argmax(axis=1), np.array(i["action_labels"])[:, 0]] for i in output_dict.values()])
            noun_acc_TOP1 = np.array([[np.array(i["nouns_preds"]).argmax(axis=1), np.array(i["action_labels"])[:, 1]] for i in output_dict.values()])
            verb_acc_TOP5 = {k: {"preds": topk(np.array(i["verbs_preds"]), k=5), "label": np.array(i["action_labels"])[:, 0]} for k, i in output_dict.items()}
            noun_acc_TOP5 = {k: {"preds": topk(np.array(i["nouns_preds"]), k=5), "label": np.array(i["action_labels"])[:, 1]} for k, i in output_dict.items()}

            top1_noun_err = torch.tensor(100 - (sum(noun_acc_TOP1[:, 0, :] == noun_acc_TOP1[:, 1, :]) / len(noun_acc_TOP1)).mean() * 100)
            top1_verb_err = torch.tensor(100 - (sum(verb_acc_TOP1[:, 0, :] == verb_acc_TOP1[:, 1, :]) / len(verb_acc_TOP1)).mean() * 100)
            top5_noun_err = torch.tensor(100 - resultsTop5(noun_acc_TOP5))
            top5_verb_err = torch.tensor(100 - resultsTop5(verb_acc_TOP5))



            errors = {
                "top1_noun_err": top1_noun_err,
                "top5_noun_err": top5_noun_err,
                "top1_verb_err": top1_verb_err,
                "top5_verb_err": top5_verb_err,
            }

            if self.h3m_used:
                errors.update({
                    "top1_intention_err": top1_intention_err,
                    "top5_intention_err": top5_intention_err,
                })
            for k, v in errors.items():
                self.log(k, v.item())
        else:
            if self.h3m_used:
                errors = {
                    "top1_intention_err": top1_intention_err,
                    "top5_intention_err": top5_intention_err,
                }
                for k, v in errors.items():
                    self.log(k, v.item())
