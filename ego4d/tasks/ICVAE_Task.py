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

logger = logging.get_logger(__name__)

class ICVAE_Task(VideoTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        #logger.info(pprint.pformat(cfg))
        self.checkpoint_metric = f"val_0_ED_{cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT-1}"

    def forward(self, inputs, tgts):
        return self.model(inputs, tgts=tgts)

    def training_step(self, batch, batch_idx):
        # batch: dict_keys(['forecast_labels', 'observed_labels', 'clip_id', 'forecast_times', 'intentions'])
        # observed_labels : [B, N, 2, 732]
        # forecast_labels: [B, Z, 2, 732]
        # intentions: [B]
        intentions, observed_labels, forecast_labels, forecast_embeds = batch["intentions"], batch["observed_labels"],\
                                                                        batch["forecast_labels"], batch["forecast_embeds"]

        # Preds is a list of tensors of shape (B, Z, C), where
        # - B is batch size,
        # - Z is number of future predictions,
        # - C is the class
        # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])
        batch_outputs = self.forward(batch, tgts=intentions)

        preds = batch_outputs["output"]
        assert len(preds) == len(self.cfg.MODEL.NUM_CLASSES), len(preds)

        losses = {k:0. for k in self.losses_params.keys()}

        # loss of the epoch
        step_result = {}
        for head_idx, (lab,pred_head) in enumerate(zip(["verbs", "nouns"],preds)): # verbs and nouns
            for seq_idx in range(pred_head.shape[1]): # different time (B, Z, C)
                # CLASSIFICATION LOSS
                k = "cross_entropy"
                if k in losses:
                    if self.cfg.CVAE.weighted_loss:
                        losses[k] += self.loss_fun[k][lab](
                            pred_head[:, seq_idx], forecast_labels[:, seq_idx, head_idx]
                        )
                    else:
                        losses[k] += self.loss_fun[k](
                            pred_head[:, seq_idx], forecast_labels[:, seq_idx, head_idx]
                        )

                # CLASSIFICATION METRICS
                top1_err, top5_err = metrics.distributed_topk_errors(
                    pred_head[:, seq_idx], forecast_labels[:, seq_idx, head_idx], (1, 5)
                )

                #VISUALIZATION
                step_result[f"train_{seq_idx}_{head_idx}_top1_err"] = top1_err.item()
                step_result[f"train_{seq_idx}_{head_idx}_top5_err"] = top5_err.item()

        # VISUALIZATION
        for head_idx in range(len(preds)):
            step_result[f"train_{head_idx}_top1_err"] = np.mean(
                [v for k, v in step_result.items() if f"{head_idx}_top1" in k]
            )
            step_result[f"train_{head_idx}_top5_err"] = np.mean(
                [v for k, v in step_result.items() if f"{head_idx}_top5" in k]
            )

        # KL divergence Loss
        k = "kl"
        if k in losses:
            mu, logvar = batch_outputs["mu"], batch_outputs["logvar"]
            losses["kl"] = self.loss_fun[k](mu, logvar)
            step_result["KL loss"] = losses[k].item()

        k = "l2"
        if k in losses:
            losses[k] = self.loss_fun[k](batch_outputs["forecast_embeds"], batch_outputs["decoded_output"].permute(1,0,2))
            step_result["L2 loss"] = losses[k].item()

        if "cross_entropy" in losses:
            step_result["classification loss"] = losses["cross_entropy"].item()  # + loss_kl
            #loss = loss / ((head_idx+1) * (seq_idx+1))


        mixed_loss = 0.
        for type_loss, lambda_loss in self.losses_params.items():
            mixed_loss+= losses[type_loss] * lambda_loss

        step_result["train_loss"] = mixed_loss.item() #+ loss_kl.item()
        step_result["loss"] = mixed_loss

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
        # batch: dict_keys(['forecast_labels', 'observed_labels', 'clip_id', 'forecast_times', 'intentions'])
        # observed_labels : [B, N, 2, 732]
        # forecast_labels: [B, Z, 2, 732]
        # intentions: [B]
        intentions, observed_labels, forecast_labels = batch["intentions"], batch["observed_labels"], batch["forecast_labels"]
        if 'vision_features' in batch:
            observed_labels = batch['vision_features']

        k = self.cfg.FORECASTING.NUM_SEQUENCES_TO_PREDICT

        # Preds is a list of tensors of shape (B, K, Z), where
        # - B is batch size,
        # - K is number of predictions,
        # - Z is number of future predictions,
        # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])
        preds = self.model.generate(intentions = intentions, observed_labels= observed_labels, k=k)
        # preds is:  [(B, K, Z)] , [(32, 5, 20)]

        step_result = {}
        for head_idx, pred in enumerate(preds):
            assert pred.shape[1] == k
            bi, ki, zi = (0, 1, 2)
            pred = pred.permute(bi, zi, ki)
            pred, forecast_labels = pred.cpu(), forecast_labels.cpu()

            label = forecast_labels[:, :, head_idx : head_idx + 1]
            auedit = metrics.distributed_AUED(pred, label)
            results = {
                f"val_{head_idx}_" + k: v for k, v in auedit.items()
            }
            step_result.update(results)

        return step_result

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def test_step(self, batch, batch_idx):
        # batch: dict_keys(['forecast_labels', 'observed_labels', 'clip_id', 'forecast_times', 'intentions'])
        # observed_labels : [B, N, 2, 732]
        # forecast_labels: [B, Z, 2, 732]
        # intentions: [B]

        k = self.cfg.FORECASTING.NUM_SEQUENCES_TO_PREDICT
        # Preds is a list of tensors of shape (B, K, Z), where
        # - B is batch size,
        # - K is number of predictions,
        # - Z is number of future predictions,
        # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])
        if 'vision_features' in batch:
            observed_labels = batch['vision_features']
        else:
            observed_labels = batch["observed_labels"]

        # only affects when performance is used
        observed_labels = observed_labels[:, -self.cfg.FORECASTING.NUM_INPUT_CLIPS:, :]
        preds = self.model.generate(intentions = batch["intentions"],
                                    observed_labels=observed_labels, k=k)  # [(B, K, Z)]

        if self.cfg.TEST.FROM_PREDICTION:
            return {
                'last_clip_ids': batch["clip_id"],
                'verb_preds': preds[0],
                'noun_preds': preds[1],
            }
        else:
            return {
                'last_clip_ids': batch["clip_id"],
                'verb_preds': preds[0],
                'noun_preds': preds[1],
                'forecast_labels': batch["forecast_labels"]
            }

    def test_epoch_end(self, outputs):

        test_outputs = {}

        keynotes = {"verb_preds": "verb", "noun_preds": "noun"}
        if not self.cfg.TEST.FROM_PREDICTION:
            keynotes["forecast_labels"]= "labels"

        for key in keynotes.keys():
            preds = torch.cat([x[key] for x in outputs], 0)
            preds = self.all_gather(preds).unbind()
            test_outputs[key] = torch.cat(preds, 0)

        last_clip_ids = [x['last_clip_ids'] for x in outputs]
        last_clip_ids = [item for sublist in last_clip_ids for item in sublist]
        last_clip_ids = list(itertools.chain(*du.all_gather_unaligned(last_clip_ids)))
        test_outputs['last_clip_ids'] = last_clip_ids

        if du.get_local_rank() == 0:
            pred_dict = {}
            for idx in range(len(test_outputs['last_clip_ids'])):
                pred_dict[test_outputs['last_clip_ids'][idx]] = {v:test_outputs[k][idx].cpu().tolist() for k,v in keynotes.items()}
            json.dump(pred_dict, open('outputs/{}_lta.json'.format(self.cfg.TEST.SPLIT), 'w'))

            if "forecast_labels" in keynotes:
                forecast_labels = torch.stack([torch.LongTensor(ex["labels"]) for ex in pred_dict.values()], dim=0)
                verb_ex = torch.stack([torch.LongTensor(ex["verb"]) for ex in pred_dict.values()], dim=0)
                noun_ex = torch.stack([torch.LongTensor(ex["noun"]) for ex in pred_dict.values()], dim=0)
                k = self.cfg.FORECASTING.NUM_SEQUENCES_TO_PREDICT
                preds = (verb_ex, noun_ex)

                step_result = {}
                for head_idx, pred in enumerate(preds):
                    assert pred.shape[1] == k
                    bi, ki, zi = (0, 1, 2)
                    pred = pred.permute(bi, zi, ki)
                    pred, forecast_labels = pred.cpu(), forecast_labels.cpu()

                    label = forecast_labels[:, :, head_idx : head_idx + 1]
                    auedit = metrics.distributed_AUED(pred, label)
                    results = {
                        f"val_{head_idx}_" + k: v[0] for k, v in auedit.items()
                    }
                    step_result.update(results)
                    print(results)

                json.dump(step_result, open('outputs/{}_lta_results.json'.format(self.cfg.TEST.SPLIT), 'w'))

