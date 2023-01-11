import itertools
import os, json, random
from ..utils import distributed as du

import numpy as np

import torch
import torch.utils.data
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import (
    Compose,
    Lambda,
)

from .build import DATASET_REGISTRY
from . import ptv_dataset_helper_features
from ..utils import logging, video_transformer

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ego4dRecognition(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Ego4d ".format(mode)

        sampler = RandomSampler
        if cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler

        clip_sampler_type = "uniform" if mode == "test" else "random"
        clip_duration = (
            self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
        ) / self.cfg.DATA.TARGET_FPS
        clip_sampler = make_clip_sampler(clip_sampler_type, clip_duration)

        mode_ = 'test_unannotated' if mode=='test' else mode
        data_path = os.path.join(self.cfg.DATA.PATH_PREFIX, f'fho_lta_{mode_}.json')
        
        self.dataset = ptv_dataset_helper_features.clip_recognition_dataset(
            data_path=data_path,
            clip_sampler=clip_sampler,
            video_sampler=sampler,
            decode_audio=False,
            transform=self._make_transform(mode, cfg),
            video_path_prefix=self.cfg.DATA.PATH_PREFIX,
        )
        self._dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(self.dataset), 2)
        )

    @property
    def sampler(self):
        return self.dataset.video_sampler

    def _make_transform(self, mode: str, cfg):
        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            Lambda(lambda x: x/255.0),
                            Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
                        ]
                        + video_transformer.random_scale_crop_flip(mode, cfg)
                        + [video_transformer.uniform_temporal_subsample_repeated(cfg)]
                    ),
                ),
                Lambda(
                    lambda x: (
                        x["video"],
                        torch.tensor([x["verb_label"], x["noun_label"]]),
                        str(x["video_name"]) + "_" + str(x["video_index"]),
                        {},
                    )
                ),
            ]
        )

    def __getitem__(self, index):
        value = next(self._dataset_iter)
        return value

    def __len__(self):
        return self.dataset.num_videos


@DATASET_REGISTRY.register()
class Ego4dRecognition_Features(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Ego4d ".format(mode)
        self.split = mode
        sampler = RandomSampler
        if cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler

        clip_sampler_type = "uniform" if mode == "test" else "random"
        clip_duration = (
                                self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
                        ) / self.cfg.DATA.TARGET_FPS
        clip_sampler = make_clip_sampler(clip_sampler_type, clip_duration)

        mode_ = 'test_unannotated' if mode == 'test' else mode
        data_path = os.path.join(self.cfg.DATA.PATH_PREFIX, f'fho_lta_{mode_}.json')

        self.dataset = ptv_dataset_helper_features.clip_recognition_dataset(
            data_path=data_path,
            clip_sampler=clip_sampler,
            video_sampler=sampler,
            transform=self._make_transform(mode, cfg),
            video_path_prefix=self.cfg.DATA.PATH_PREFIX,
            feature_dir_path = self.cfg.DATA.FEAT_PREFIX,
            featuretype= self.cfg.DATA.FEATURE_TYPE
        )
        self._dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(self.dataset), 2)
        )

    @property
    def sampler(self):
        return self.dataset.video_sampler

    def _make_transform(self, mode: str, cfg):

        def extract_vision_features(x):
            y = x["features"]
            if self.cfg.MLPMixer.augmentation:
                if self.split in "train":
                    y = y[torch.randperm(14)[:self.cfg.MLPMixer.num_features], :]
                else:
                    y=y[-self.cfg.MLPMixer.num_features:, :]
            return torch.nn.functional.normalize(y, p=2.0, dim=1)

        if self.cfg.DATA.FEATURE_TYPE =="vision":
            return Compose(
            [
                Lambda(
                    lambda x: {
                        "vision_features": extract_vision_features(x),
                        "observed_labels": torch.tensor([x["verb_label"], x["noun_label"]]),
                        "clip_id": str(x["clip_uid"]) + "_" + str(x["action_idx"]),
                    }
                ),
            ]
        )
        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            Lambda(lambda x: x / 255.0),
                            Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
                        ]
                        + video_transformer.random_scale_crop_flip(mode, cfg)
                        + [video_transformer.uniform_temporal_subsample_repeated(cfg)]
                    ),
                ),
                Lambda(
                    lambda x: (
                        x["video"],
                        torch.tensor([x["verb_label"], x["noun_label"]]),
                        str(x["video_name"]) + "_" + str(x["video_index"]),
                        {},
                    )
                ),
            ]
        )

    def __getitem__(self, index):
        value = next(self._dataset_iter)
        return value

    def __len__(self):
        return self.dataset.num_videos

@DATASET_REGISTRY.register()
class Ego4dLongTermAnticipation_Features(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Ego4d ".format(mode)
        self.split = mode
        sampler = RandomSampler
        if cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler

        clip_sampler_type = "uniform" if mode == "test" else "random"
        clip_duration = (
            self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
        ) / self.cfg.DATA.TARGET_FPS
        clip_sampler = make_clip_sampler(clip_sampler_type, clip_duration)

        mode_ = 'test_unannotated' if mode=='test' else mode

        # [!!]
        if mode == 'test' and cfg.TEST.EVAL_VAL:
            mode_ = 'val'
        data_path = os.path.join(self.cfg.DATA.PATH_PREFIX, f'fho_lta_{mode_}.json')

        self.dataset = ptv_dataset_helper_features.clip_forecasting_dataset(
            data_path=data_path,
            clip_sampler=clip_sampler,
            num_input_actions=self.cfg.FORECASTING.NUM_INPUT_CLIPS,
            num_future_actions=self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT,
            video_sampler=sampler,
            transform=self._make_transform(mode, cfg),
            video_path_prefix=self.cfg.DATA.PATH_PREFIX,
            feature_dir_path = self.cfg.DATA.FEAT_PREFIX,
            featuretype= self.cfg.DATA.FEATURE_TYPE,
        )
        print("DATASET LENGTH: ",mode, self.dataset.num_videos)
        self._dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(self.dataset), 2)
        )

        def read_json(filename):
            with open(filename) as jsonFile:
                data = json.load(jsonFile)
                jsonFile.close()
            return data



        def intentions_mapper():
            ego4d = read_json(self.cfg.DATA.PATH_PREFIX + "ego4d.json")
            ids_scenarios = {x["video_uid"]: self.taxonomy["intention"].index(x["scenarios"][0]) for x in ego4d["videos"] if len(x["scenarios"])==1 and x["scenarios"][0] in self.taxonomy["intention"]}
            return ids_scenarios

        ## IMPORTANT: THIS IS TO USE THE DATA FROM PREDICTED, NOT GT
        if self.cfg.TEST.FROM_PREDICTION:
            previous_predictions = read_json(self.cfg.TEST.OUTPUTS_PATH)
            self.intention_pred = {clip_id: np.array(data) for clip_id, data in previous_predictions["intention_preds"].items()}
            self.verb_pred = {clip_id: np.array(data) for clip_id, data in previous_predictions["verbs_preds"].items()}
            self.noun_pred = {clip_id: np.array(data) for clip_id, data in previous_predictions["nouns_preds"].items()}
        else:
            self.taxonomy = read_json(self.cfg.DATA.PATH_PREFIX+"fho_lta_int_taxonomy.json") # fho_lta_int_taxonomy_complete if num_intentions > 52 else fho_lta_int_taxonomy
            if self.cfg.DATA.FEATURE_TYPE == "language":
                self.features = torch.load(self.cfg.DATA.PATH_PREFIX + "taxonomy_clip_embeddings.pt") #"fho_lta_text_embed.pt"
            self.intentions_mapper_dict = intentions_mapper()


    @property
    def sampler(self):
        return self.dataset.video_sampler

    def _make_transform(self, mode: str, cfg):
        class ReduceExpandInputClips:
            def __init__(self, transform):
                self.transform = transform

            def __call__(self, x):
                if x.dim() == 4:
                    x = x.unsqueeze(0)  # Handle num_clips=1

                n, c, t, h, w = x.shape
                x = x.transpose(0, 1)
                x = x.reshape(c, n * t, h, w)
                x = self.transform(x)

                if isinstance(x, list):
                    for i in range(len(x)):
                        c, _, h, w = x[i].shape
                        x[i] = x[i].reshape(c, n, -1, h, w)
                        x[i] = x[i].transpose(1, 0)
                else:
                    c, _, h, w = x.shape
                    x = x.reshape(c, n, t, h, w)
                    x = x.transpose(1, 0)

                return x

        def extract_forecast_labels(x):
            clips = x["forecast_clips"]
            nouns = torch.tensor([y["noun_label"] for y in clips])
            verbs = torch.tensor([y["verb_label"] for y in clips])
            labels = torch.stack([verbs, nouns], dim=-1)
            return labels

        def extract_forecast_embeds(x):
            clips = x["forecast_clips"]
            if self.cfg.DATA.FEATURE_TYPE == "onehot" or self.cfg.DATA.FEATURE_TYPE == "vision":
                nouns = torch.tensor([y["noun_label"] for y in clips])
                verbs = torch.tensor([y["verb_label"] for y in clips])
            else:
                nouns = torch.stack([self.features["nouns"][y["noun_label"]] if y["noun_label"]!=-1 else torch.zeros(self.cfg.CVAE.semantic_dim) for y in clips])
                verbs = torch.stack([self.features["verbs"][y["verb_label"]] if y["verb_label"]!=-1 else torch.zeros(self.cfg.CVAE.semantic_dim) for y in clips])

            labels = torch.stack([verbs, nouns], dim=-1)
            return labels

        def addAgumentation(observed_labels, prob=0.2, uniform=True):
            """
            Randomly create error labels to provide the model training for the refiner
            :param observed_labels:
            :return:
            """
            N, _ = observed_labels.shape
            if uniform:
                verbs = torch.randint(low=0, high=self.cfg.CVAE.num_verbs, size=(N, 1))
                nouns = torch.randint(low=0, high=self.cfg.CVAE.num_nouns, size=(N, 1))
                noise_labels = torch.cat([verbs, nouns], dim=1)
                mods = torch.LongTensor(np.random.choice(2, size= noise_labels.shape, p=[1-prob, prob]))
                observed_labels[mods==1] = noise_labels[mods==1]
                return observed_labels
            else:
                noise = torch.normal(0, prob, observed_labels.shape, device=observed_labels.device)
                noisy_labels = observed_labels + noise
                noisy_labels[:, 0] = torch.clamp(noisy_labels[ :, 0], min=0, max=self.cfg.CVAE.num_verbs - 1)
                noisy_labels[:, 1] = torch.clamp(noisy_labels[ :, 1], min=0, max=self.cfg.CVAE.num_nouns - 1)
                return torch.round(noisy_labels).type(torch.long)

        def extract_observed_labels(x):
            clips = x["input_clips"]
            if self.cfg.DATA.FEATURE_TYPE == "onehot" or self.cfg.DATA.FEATURE_TYPE == "vision":
                nouns = torch.tensor([y["noun_label"] for y in clips])
                verbs = torch.tensor([y["verb_label"] for y in clips])
            else:
                nouns = torch.stack([self.features["nouns"][y["noun_label"]] if y["noun_label"]!=-1 else torch.zeros(self.cfg.CVAE.semantic_dim) for y in clips])
                verbs = torch.stack([self.features["verbs"][y["verb_label"]] if y["verb_label"]!=-1 else torch.zeros(self.cfg.CVAE.semantic_dim) for y in clips])
            labels = torch.stack([verbs, nouns], dim=-1) # B, N, 2
            if self.cfg.CVAE.add_noisy_labels and self.split in "train":
                labels = addAgumentation(labels, prob=0.3, uniform=False)
            return labels

        # last visible annotated clip: (clip_uid + action_idx)
        def extract_clip_id(x):
            last_clip = x['input_clips'][-1]
            return f'{last_clip["clip_uid"]}_{last_clip["action_idx"]}'

        def extract_forecast_times(x):
            clips = x["forecast_clips"]
            start_end = [(y["clip_start_sec"], y["clip_end_sec"]) for y in clips]
            return {"label_clip_times": start_end}

        def extract_intentions(x):
            if self.cfg.CVAE.use_intention:
                last_clip = x['input_clips'][-1]
                video_uid = last_clip["video_uid"]
                return self.intentions_mapper_dict[video_uid]
            else:
                return 0

        def extract_vision_features(x):
            y =  torch.stack(x["features"], axis=0)
            if self.cfg.MLPMixer.augmentation:
                if self.split in "train":
                    y = y[:,torch.randperm(self.cfg.MLPMixer.num_features)[:self.cfg.MLPMixer.num_features], :]
                else:
                    y=y[:,-self.cfg.MLPMixer.num_features:, :]
            return torch.nn.functional.normalize(y[:,-self.cfg.MLPMixer.num_features:, :], p=2.0, dim=2)

        def extract_predicted_results(x):
            clip_id = extract_clip_id(x)
            if clip_id not in self.verb_pred:
                print("Error when obtaining results from {}".format(clip_id))
                newid = random.choice(list(self.verb_pred.keys()))
                return {
                    "clip_id": clip_id,
                    "observed_labels": torch.stack([torch.tensor(-1), torch.tensor(-1)], dim=-1),
                    "intentions": self.intention_pred[newid],
                    "forecast_labels": extract_forecast_labels(x)

                }
            else:
                verbs = torch.tensor(self.verb_pred[clip_id])
                nouns = torch.tensor(self.noun_pred[clip_id])
                if clip_id in self.intention_pred:
                    intent = self.intention_pred[clip_id]
                else:
                    print("Error when obtaining intention results from {}".format(clip_id))
                    intent = np.array(0)
                return {
                    "clip_id" : clip_id,
                    "observed_labels": torch.stack([verbs, nouns], dim=-1),
                    "intentions": intent,
                    "forecast_labels": extract_forecast_labels(x)

                }


        if self.cfg.DATA.FEATURE_TYPE in  "vision" :
            if self.cfg.TEST.FROM_PREDICTION:
                return Compose(
                    [
                        Lambda(
                            lambda x: (
                                     extract_predicted_results(x)
                            )
                        ),
                    ]
                )
            if self.cfg.MODEL.ARCH == "mlpmixer":
                return Compose(
                    [
                        Lambda(
                            lambda x: (
                                {
                                    "vision_features": extract_vision_features(x),
                                    "observed_labels": extract_observed_labels(x),
                                    "clip_id": extract_clip_id(x),
                                    "intentions": extract_intentions(x),
                                }
                            )
                        ),
                    ]
                )
            return Compose(
                [
                    Lambda(
                        lambda x: (
                            {
                                "vision_features": extract_vision_features(x),
                                "forecast_labels": extract_forecast_labels(x),
                                "forecast_embeds": extract_forecast_embeds(x),
                                "observed_labels": extract_observed_labels(x),
                                "clip_id": extract_clip_id(x),
                                "forecast_times": extract_forecast_times(x),
                                "intentions": extract_intentions(x),
                            }
                        )
                    ),
                ]
            )
        elif self.cfg.DATA.FEATURE_TYPE == "onehot" :
            if self.cfg.TEST.FROM_PREDICTION:
                return Compose(
                    [
                        Lambda(
                            lambda x: (
                                extract_predicted_results(x)
                            )
                        ),
                    ]
                )
            return Compose(
                [
                    Lambda(
                        lambda x: (
                            {
                                "forecast_labels": extract_forecast_labels(x),
                                "forecast_embeds": extract_forecast_embeds(x),
                                "observed_labels": extract_observed_labels(x),
                                "clip_id": extract_clip_id(x),
                                "forecast_times": extract_forecast_times(x),
                                "intentions": extract_intentions(x),
                            }
                        )
                    ),
                ]
            )
        else:

            return Compose(
                [
                    Lambda(
                        lambda x: (
                            {
                                "forecast_labels" : extract_forecast_labels(x),
                                "forecast_embeds": extract_forecast_embeds(x),
                                "observed_labels" : extract_observed_labels(x),
                                "clip_id" : extract_clip_id(x),
                                "forecast_times" : extract_forecast_times(x),
                                "intentions" : extract_intentions(x),
                            }
                        )
                    ),
                ]
            )

    def __getitem__(self, index):
        value = next(self._dataset_iter)
        return value

    def __len__(self):
        return self.dataset.num_videos
