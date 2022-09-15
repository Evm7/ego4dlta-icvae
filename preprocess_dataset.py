import torch, json, os, glob, argparse
import numpy as np
import tqdm


def parse_args():
    """
    Parse the following arguments for preprocessing features from Slowfast 8x8 R101
    Args:
        dir_annotations (string): path where the annotations can be found. Default: /data/annotations/
        download_path (string): path where the ego4d features where downloaded into. Default: Ego4d/v1/slowfast8x8_r101_k400/
        features_path (string): path to where the preprocessed features will be stored for model training/testing. Default: Ego4D/features_pad/
        W (int): windows size of the feature extractor. Info provided by Ego4d when extracting the downloaded features. Default: 32
        S (int): stride of the feature extractor. Info provided by Ego4d when extracting the downloaded features. Default=16
        N (int): number of features to pad to. Default: 15
        """
    parser = argparse.ArgumentParser(
        description="Provide details of the preprocessment of the Ego4d downloaded features."
    )
    parser.add_argument(
        "download_path",
        help="Path where the ego4d features where downloaded into",
        default="Ego4d/v1/slowfast8x8_r101_k400/",
        type=str,
    )
    parser.add_argument(
        "features_path",
        help="Path to where the preprocessed features will be stored for model training/testing",
        default="Ego4D/features_pad/",
        type=str,
    )
    parser.add_argument(
        "W",
        help="Windows size of the feature extractor",
        default=32,
        type=int,
    )
    parser.add_argument(
        "S",
        help="Stride of the feature extractor",
        default=16,
        type=int,
    )
    parser.add_argument(
        "N",
        help="Number of features to pad to",
        default=15,
        type=int,
    )
    parser.add_argument(
        "--remove",
        help="To remove the downloaded matched features or not after padding",
        default=True,
        type=bool,
    )
    return parser.parse_args()


def read_json(filename):
    with open(filename) as jsonFile:
        data = json.load(jsonFile)
        jsonFile.close()
    return data


def read_annotations(dir_annotations):
    """
    Parses the annotations files and maps the videoIDs (name of the downloaded files) with the clipIDs,
     together with the start and end frame assigned for them
    Args:
        dir_annotations: directory where the annotations are found

    Returns: mapped dictionary with all the summary of VidID an ClipID

    """
    summary_data = {}
    split = ["train", "test_unannotated", "val"]  # test, val or train
    for s in split:
        data_info = read_json(dir_annotations + "fho_lta_{}.json".format(s))

        for c in data_info["clips"]:
            video_uid = c["video_uid"]
            clip_uid = "{}_{}".format(c["clip_uid"], c["action_idx"])
            if video_uid not in summary_data:
                summary_data[video_uid] = []

            info_clip = {"clip_uid": clip_uid,
                         "start_frame": c["action_clip_start_frame"],
                         "end_frame": c["action_clip_end_frame"]}
            summary_data[video_uid].append(info_clip)
    return summary_data


def frame_to_feat(num_frame, W=32, S=16):
    """
    Matches the frame start/end from a video to the assigned feature belonging to it
    Args:
        num_frame: frame index
        W: window size assigned by the preprocessor model. Strategy followed when extracting the feature
        S: stride assigned by the preprocessor model. Strategy followed when extracting the feature

    Returns: index of feature to filter from/to

    """
    return int(num_frame / S)


def preprocess_features(args):
    """
    Read annotations to match the VidId with ClipID and obtain the start and end frame from the action clips.
    Extract the given feature and padds it based on the arguments passed.
    Args:
        args: arguments parsed as inputs. More info in function @parsing_args

    Returns: features padd in the features_path and deleted from download_path to avoid memory issues, if args remove
     is True

    """
    summary_data = read_annotations(args["dir_annotations"])
    downloads_files = np.array(glob.glob(args["download_path"] + "*.pt", recursive=True))

    for f in tqdm.tqdm(downloads_files):
        clip_uid = f.split("/")[-1][:-3]
        if clip_uid in summary_data:
            results = summary_data[clip_uid]
            if os.path.exists(f):
                data = torch.load(f)
                for i, res in enumerate(results):
                    start_feat = frame_to_feat(res["start_frame"])
                    end_feat = frame_to_feat(res["end_frame"])
                    feat_clip = data[start_feat:end_feat].clone()
                    store_path = args["features_path"] + res["clip_uid"] + ".pt"
                    if (end_feat - start_feat) != args["N"]:
                        pad = torch.zeros(args["N"] - feat_clip.shape[0], feat_clip.shape[1])
                        feat_clip = torch.vstack([pad, feat_clip])
                    torch.save(feat_clip, store_path)
                if args["remove"]:
                    os.remove(f)

if __name__ == '__main__':
    args = parse_args()
    preprocess_features(args)
