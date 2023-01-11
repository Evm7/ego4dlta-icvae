#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi input models."""

from torch.distributions.categorical import Categorical
import math
import copy

from functools import reduce
from ego4d.models.architectures.head_helper import MultiTaskHead, MultiTaskMViTHead
from ego4d.models.architectures.video_model_builder import SlowFast, _POOL1, MViT
from .build import MODEL_REGISTRY

from ego4d.models.architectures.cae import CAE
from ego4d.models.architectures.transformer import *
from ego4d.models.architectures.mlp_mixer import *
from ego4d.models.architectures.MultiHead import *


@MODEL_REGISTRY.register()
class ICVAE(CAE):
    def __init__(self, cfg):
        self.cfg = cfg
        self.params = self.cfg.CVAE
        self.params["lambdas"] = self.cfg.MODEL.lambdas
        self.params["cfg"] = cfg
        self.params["device"] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.shared_params = self.params.shared_embedding

        if self.params.featuretype in "language":
            self.params.latent_dim = self.params.semantic_dim
            self.params["pretrained_from"] = self.cfg.DATA.PATH_PREFIX + "taxonomy_clip_embeddings.pt"

        mlpmixer = None
        embedder = None

        if self.params.featuretype in 'vision':
            self.actions_labels = cfg.MLPMixer.action_loss
            self.params['feature_dimension'] = cfg.MLPMixer.feature_dimension
            mlpmixer =  MLPMixer(num_features=cfg.MLPMixer.num_features,
                                feature_dimension= cfg.MLPMixer.feature_dimension,
                                depth= cfg.MLPMixer.depth,
                                num_classes = cfg.MLPMixer.num_actions_classes,
                                expansion_factor=cfg.MLPMixer.expansion_factor,
                                expansion_factor_token = cfg.MLPMixer.expansion_factor_token,
                                dropout=cfg.MODEL.DROPOUT_RATE,
                                reduce_to_class=self.actions_labels,
                                action_loss=self.actions_labels,
                                position_encoder = cfg.MLPMixer.position_encoder,
                                test_noact=False
                              )
        if self.shared_params:
            embedder = EmbeddingActions(**self.params)
        encoder = Encoder_TRANSFORMER(**self.params)
        decoder = Decoder_TRANSFORMER(**self.params)
        multihead = MultiHeadDecoder_cvae(**self.params)
        super().__init__(encoder, decoder, multihead, embedder, mlpmixer, **self.params)



    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def forward(self, batch, tgts=None):
        if self.params.featuretype in 'vision':
            batch.update(self.resume_visual(batch))
        if self.shared_params:
            batch.update(self.embedder(batch))
        # encode
        batch.update(self.encoder(batch))
        batch["z"] = self.reparameterize(batch)
        # decode
        batch.update(self.decoder(batch))
        batch.update(self.multihead(batch))
        return batch

    def resume_visual(self, batch):
        x = batch['vision_features']
        B, N, S, D = x.shape
        if self.actions_labels:
            action_embed, nouns_classes, verbs_classes = [], [], []
            for i in range(N):
                (v, n), action_emb = self.mlpmixer(x[:,i, : ,:]) # y is [verb, noun]
                action_embed.append(action_emb)
                nouns_classes.append(n)
                verbs_classes.append(v)
            action_embed = torch.stack(action_embed, axis=1)
            nouns_classes = torch.stack(nouns_classes).permute(1,0,2)
            verbs_classes = torch.stack(verbs_classes).permute(1,0,2)
            batch['observed_labels'] = torch.cat((verbs_classes, nouns_classes), dim=2)
            batch['vision_features'] = action_embed
            return batch
        action_embed =  torch.stack([self.mlpmixer(x[:,i, : ,:]) for i in range(N)], axis=1)
        batch['vision_features'] = action_embed
        return batch

    def return_latent(self, batch, seed=None):
        distrib_param = self.encoder(batch)
        batch.update(distrib_param)
        return self.reparameterize(batch, seed=seed)

@MODEL_REGISTRY.register()
class H3M(nn.Module):
    '''
    H3M Network used to obtain the Intention and the Action through a MultiTask Approach.
    '''
    def __init__(self, cfg, from_preparator=False):
        super().__init__()
        self.params = cfg.CVAE
        self.params["cfg"] = cfg
        self.params["device"] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.mixer = cfg.MLPMixer.double_mixer
        self.classify_classes = cfg.MLPMixer.action_loss
        self.multitask_head_bool = cfg.MLPMixer.multitask_head
        self.only_recognition = cfg.TRAIN.DATASET == "Ego4dRecognition_Features"
        self.h3m_used = cfg.MLPMixer.h3m_used


        if self.multitask_head_bool:
            self.multitask_head = ActionHead(
                            dim_in=[ cfg.MultiHead.input_dimension],
                            num_classes= cfg.MODEL.NUM_CLASSES[0],
                            dropout_rate=cfg.MODEL.DROPOUT_RATE,
                            act_func="softmax",
                            test_noact=False)
        if not self.only_recognition and self.h3m_used:
            if self.mixer:
                self.action_mixer = MLPMixer(num_features=cfg.MLPMixer.num_features,
                                          feature_dimension=cfg.MLPMixer.feature_dimension,
                                          depth= cfg.MLPMixer.depth,
                                          num_classes = cfg.MLPMixer.num_actions_classes,
                                          expansion_factor=cfg.MLPMixer.expansion_factor,
                                          expansion_factor_token = cfg.MLPMixer.expansion_factor_token,
                                          dropout=cfg.MODEL.DROPOUT_RATE,
                                          reduce_to_class=self.classify_classes,
                                          action_loss=self.classify_classes,
                                            position_encoder = cfg.MLPMixer.position_encoder,
                                             test_noact=from_preparator
                                          )
            else:
                self.action_mixer = nn.AdaptiveAvgPool2d((1, cfg.MLPMixer.feature_dimension))


            self.intention_mixer = MLPMixer(num_features=cfg.FORECASTING.NUM_INPUT_CLIPS,
                                      feature_dimension=cfg.MLPMixer.feature_dimension,
                                      depth=cfg.MLPMixer.depth,
                                      num_classes = cfg.MLPMixer.num_intentions,
                                      expansion_factor=cfg.MLPMixer.expansion_factor,
                                      expansion_factor_token = cfg.MLPMixer.expansion_factor_token,
                                      dropout=cfg.MODEL.DROPOUT_RATE,
                                      reduce_to_class=True,
                                        position_encoder = cfg.MLPMixer.position_encoder
                                      )

    def forward(self, x):
        if self.only_recognition: # Using Ego4DRecognitionDataset
            # B, S, D = x.shape
            v, n = self.multitask_head(x[:, :, :])
            return [v, n]
        else:
            B, N, S, D = x.shape
            nouns_classes, verbs_classes = [], []

            if not self.h3m_used:
                for i in range(N):
                    v, n = self.multitask_head(x[:, i, :, :])
                    verbs_classes.append(v)
                    nouns_classes.append(n)
                verbs_classes = torch.stack(verbs_classes).permute(1, 0, 2)
                nouns_classes = torch.stack(nouns_classes).permute(1, 0, 2)
                return [verbs_classes, nouns_classes]
            else:
                if self.mixer and self.classify_classes:
                    action_embed = []
                    for i in range(N):
                        (v, n), action_emb = self.action_mixer(x[:,i, : ,:]) # y is [verb, noun]
                        action_embed.append(action_emb)
                        if self.multitask_head_bool: # Only used to classify the classes, but does not effect to the embedding obtained
                            v, n = self.multitask_head(x[:,i, : ,:])
                        nouns_classes.append(n)
                        verbs_classes.append(v)
                    action_embed = torch.stack(action_embed, axis=1)
                    nouns_classes = torch.stack(nouns_classes).permute(1,0,2)
                    verbs_classes = torch.stack(verbs_classes).permute(1,0,2)
                    return self.intention_mixer(action_embed), [verbs_classes, nouns_classes]

                elif self.mixer:
                    action_embed = torch.stack([self.action_mixer(x[:,i, : ,:]) for i in range(N)], axis=1)
                else:
                    action_embed = torch.stack([self.action_mixer(x[:,i, : ,:]).squeeze() for i in range(N)], axis=1)

            return self.intention_mixer(action_embed)


######
##
## EGO4D BASELINES FROM NOW ON. DO NOT REMOVE FOR COMPARISION
##
######


## EGO 4D MODELS
@MODEL_REGISTRY.register()
class MultiTaskSlowFast(SlowFast):
    def _construct_network(self, cfg, with_head=False):
        super()._construct_network(cfg, with_head=with_head)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        head = MultiTaskHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA // pool_size[0][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ],
                [
                    cfg.DATA.NUM_FRAMES // pool_size[1][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                ],
            ],  # None for AdaptiveAvgPool3d((1, 1, 1))
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )
        self.head_name = "head"
        self.add_module(self.head_name, head)

@MODEL_REGISTRY.register()
class RecognitionSlowFastRepeatLabels(MultiTaskSlowFast):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def forward(self, x, tgts=None):
        # keep only first input
        x = [xi[:, 0] for xi in x]
        x = super().forward(x)

        # duplicate predictions K times
        K = self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        x = [xi.unsqueeze(1).repeat(1, K, 1) for xi in x]
        return x

    def generate(self, x, k=1):
        x = self.forward(x)
        results = []
        for head_x in x:
            preds_dist = Categorical(logits=head_x)
            preds = [preds_dist.sample() for _ in range(k)]
            head_x = torch.stack(preds, dim=1)
            results.append(head_x)
        return results


@MODEL_REGISTRY.register()
class MultiTaskMViT(MViT):

    def __init__(self, cfg):

        super().__init__(cfg, with_head =False)

        self.head = MultiTaskMViTHead(
            [768],
            cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

#--------------------------------------------------------------------#

@MODEL_REGISTRY.register()
class ConcatAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        x = torch.stack(x, dim=1) # (B, num_input_clips, D)
        x = x.view(x.shape[0], -1) # (B, num_input_clips * D)
        return x

    @staticmethod
    def out_dim(cfg):
        return cfg.MODEL.MULTI_INPUT_FEATURES * cfg.FORECASTING.NUM_INPUT_CLIPS

@MODEL_REGISTRY.register()
class MeanAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        x = torch.stack(x, dim=1) # (B, num_input_clips, D)
        x = x.mean(1)
        return x

    @staticmethod
    def out_dim(cfg):
        return cfg.MODEL.MULTI_INPUT_FEATURES

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :, :]
        return self.dropout(x)

@MODEL_REGISTRY.register()
class TransformerAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_heads = cfg.MODEL.TRANSFORMER_ENCODER_HEADS
        num_layers = cfg.MODEL.TRANSFORMER_ENCODER_LAYERS
        dim_in = cfg.MODEL.MULTI_INPUT_FEATURES
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_in, num_heads),
            num_layers,
            norm=nn.LayerNorm(dim_in),
        )
        self.pos_encoder = PositionalEncoding(dim_in, dropout=0.2)

    def forward(self, x):
        x = torch.stack(x, dim=1) # (B, num_inputs, D)
        x = x.transpose(0, 1) # (num_inputs, B, D)
        x = self.pos_encoder(x)
        x = self.encoder(x) # (num_inputs, B, D)
        return x[-1] # (B, D) return last timestep's encoding

    @staticmethod
    def out_dim(cfg):
        return cfg.MODEL.MULTI_INPUT_FEATURES

#--------------------------------------------------------------------#

@MODEL_REGISTRY.register()
class MultiHeadDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        head_classes = [
            reduce((lambda x, y: x + y), cfg.MODEL.NUM_CLASSES)
        ] * self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        head_dim_in = MODEL_REGISTRY.get(cfg.FORECASTING.AGGREGATOR).out_dim(cfg)
        self.head = MultiTaskHead(
            dim_in=[head_dim_in],
            num_classes=head_classes,
            pool_size=[None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )

    def forward(self, x, tgts=None):
        x = x.view(x.shape[0], -1, 1, 1, 1)
        x = torch.stack(self.head([x]), dim=1) # (B, Z, #verbs + #nouns)
        x = torch.split(x, self.cfg.MODEL.NUM_CLASSES, dim=-1) # [(B, Z, #verbs), (B, Z, #nouns)]
        return x

#--------------------------------------------------------------------#

@MODEL_REGISTRY.register()
class ForecastingEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.build_clip_backbone()
        self.build_clip_aggregator()
        self.build_decoder()

    # to encode frames into a set of {cfg.FORECASTING.NUM_INPUT_CLIPS} clips
    def build_clip_backbone(self):
        cfg = self.cfg
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        backbone_config = copy.deepcopy(cfg)
        backbone_config.MODEL.NUM_CLASSES = [cfg.MODEL.MULTI_INPUT_FEATURES]
        backbone_config.MODEL.HEAD_ACT = None


        if cfg.MODEL.ARCH == "mvit":
            self.backbone = MViT(backbone_config, with_head=True)
        else:
            self.backbone = SlowFast(backbone_config, with_head=True)
        # replace with:
        # self.backbone = MODEL_REGISTRY.get(cfg.FORECASTING.BACKBONE)(backbone_config, with_head=True)

        if cfg.MODEL.FREEZE_BACKBONE:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Never freeze head.
            for param in self.backbone.head.parameters():
                param.requires_grad = True


    def build_clip_aggregator(self):
        self.clip_aggregator = MODEL_REGISTRY.get(self.cfg.FORECASTING.AGGREGATOR)(self.cfg)

    def build_decoder(self):
        self.decoder = MODEL_REGISTRY.get(self.cfg.FORECASTING.DECODER)(self.cfg)

    # input = [(B, num_inp, 3, T, H, W), (B, num_inp, 3, T', H, W)]
    def encode_clips(self, x):
        # x -> [torch.Size([2, 2, 3, 8, 224, 224]), torch.Size([2, 2, 3, 32, 224, 224])]
        assert isinstance(x, list) and len(x) >= 1

        num_inputs = x[0].shape[1]
        batch = x[0].shape[0]
        features = []
        for i in range(num_inputs):
            pathway_for_input = []
            for pathway in x:
                input_clip = pathway[:, i]
                pathway_for_input.append(input_clip)

            # pathway_for_input -> [torch.Size([2, 3, 8, 224, 224]), torch.Size([2, 3,32, 224, 224])]
            input_feature = self.backbone(pathway_for_input)
            features.append(input_feature)

        return features

    # input = list of clips: [(B, D)] x {cfg.FORECASTING.NUM_INPUT_CLIPS}
    # output = (B, D') tensor after aggregation
    def aggregate_clip_features(self, x):
        return self.clip_aggregator(x)

    # input = (B, D') tensor encoding of full video
    # output = [(B, Z, #verbs), (B, Z, #nouns)] probabilities for each z
    def decode_predictions(self, x, tgts):
        return self.decoder(x, tgts)

    def forward(self, x, tgts=None):
        features = self.encode_clips(x)
        x = self.aggregate_clip_features(features)
        x = self.decode_predictions(x, tgts)
        return x

    def generate(self, x, k=1):
        x = self.forward(x)
        results = []
        for head_x in x:
            if k>1:
                preds_dist = Categorical(logits=head_x)
                preds = [preds_dist.sample() for _ in range(k)]
            elif k==1:
                preds = [head_x.argmax(2)]
            head_x = torch.stack(preds, dim=1)
            results.append(head_x)

        return results
