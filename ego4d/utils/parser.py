#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import sys
import yaml
import typing

from ..config.defaults import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        cfg (str): path to the config file.
        opts (argument): provide additional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument("--job_name", default="", type=str)
    parser.add_argument("--on_cluster", action="store_true")
    parser.add_argument("--working_directory", default="", type=str)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards", help="Number of shards using by the job", default=1, type=int
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See ego4d/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    '''
    if len(sys.argv) == 1:
        parser.print_help()
    '''
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards"):
        cfg.NUM_SHARDS = args.num_shards
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir
    if hasattr(args, "fast_dev_run"):
        cfg.FAST_DEV_RUN = args.fast_dev_run

    return cfg

def load_default_config(args, cfg_file):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()

    # Load config from cfg.
    if cfg_file is not None:
        cfg.merge_from_file(cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards"):
        cfg.NUM_SHARDS = args.num_shards
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir
    if hasattr(args, "fast_dev_run"):
        cfg.FAST_DEV_RUN = args.fast_dev_run
    return cfg

# This part of the script is added by @Evm7 in order to automatize the loading of hparams.yaml when testing
class Loader(yaml.SafeLoader):
    pass

class Dumper(yaml.SafeLoader):
    pass

class Tagged(typing.NamedTuple):
    tag: str
    value: object

def construct_undefined(self, node):
    if isinstance(node, yaml.nodes.ScalarNode):
        value = self.construct_scalar(node)
    elif isinstance(node, yaml.nodes.SequenceNode):
        value = self.construct_sequence(node)
    elif isinstance(node, yaml.nodes.MappingNode):
        value = self.construct_mapping(node)
    else:
        assert False, f"unexpected node: {node!r}"
    return Tagged(node.tag, value)

def is_dict(a):
    return type(a) == dict

def is_tag(a):
    return type(a) == Tagged

def unload_tag(tag):
    return tag.value['dictitems']

def load_file_yaml(filename, Loader=None):
    with open(filename) as file:
        if Loader == None:
            documents = yaml.safe_load(file)
        else:
            documents = yaml.load(file, Loader=Loader)
    return documents

def unload_dict(dict_):
    new_dict = {}
    for key, value in dict_.items():
        if is_tag(value):
            dict2 = unload_tag(value)
            value = unload_dict(dict2)
        new_dict[key] = value
    return new_dict

def save_dict_as_yaml(filename, data):
    with open(filename, 'w') as file:
        yaml.dump(data, file)
        return

def adaptLoader(filename, experiments_file = '/home/evallsmascaro/PycharmProjects/GoalConditionedForecasting/experiments/test_config.yaml'):
    '''

    :param filename: hparams.yaml filename saved with the configuration file when training a model
    :return: CfgNode file path saved need for building the model
    '''
    Loader.add_constructor(None, construct_undefined)
    documents = load_file_yaml(filename, Loader=Loader)
    configuration = unload_dict(documents)
    _ = save_dict_as_yaml(experiments_file, configuration['cfg'])
    return experiments_file


