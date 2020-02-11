# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import logging
from collections import OrderedDict
import torch
import pickle

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import LOFARTrainer, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, LOFAREvaluator
from detectron2.structures import BoxMode

from tridentnet import add_tridentnet_config


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_tridentnet_config(cfg)
    cfg.merge_from_file("/data/mostertrij/tridentnet/detectron2/projects/TridentNet/configs/my_tridentnet_fast_R_101_C4_3x.yaml")
    DATASET_NAME= "LGZ_v5_more_rotations"
    cfg.DATASETS.TRAIN = (f"{DATASET_NAME}_train",)
    cfg.DATASETS.VAL = (f"{DATASET_NAME}_val",)
    cfg.DATASETS.TEST = (f"{DATASET_NAME}_test",)
    #cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_lofar_dicts(annotation_filepath):
    """
    Read LOFAR ground truth labels and add the appropriate detectron2 boxmode
    """
    with open(annotation_filepath, "rb") as f:
        dataset_dicts = pickle.load(f)
    for i in range(len(dataset_dicts)):
        for ob in dataset_dicts[i]['annotations']:
            ob['bbox_mode'] = BoxMode.XYXY_ABS
    return dataset_dicts


def register_lofar_datasets(cfg):
    """
    Register LOFAR dataset
    """
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    DATASET_NAME= "LGZ_v5_more_rotations"
    base_path = f"/data/mostertrij/data/frcnn_images/{DATASET_NAME}/LGZ_COCOstyle/annotations/"

    from detectron2.data import DatasetCatalog, MetadataCatalog
    for d in ["train", "val", "test"]:
        DatasetCatalog.register(f"{DATASET_NAME}_" + d, 
                                lambda d=d: get_lofar_dicts(os.path.join(base_path,f"VIA_json_{d}.pkl")))
        MetadataCatalog.get(f"{DATASET_NAME}_" + d).set(thing_classes=["radio_source"])
    lofar_metadata = MetadataCatalog.get(f"{DATASET_NAME}_train")

    print(cfg.DATASETS.TRAIN, cfg.DATASETS.VAL, cfg.DATASETS.TEST)


def main(args):
    cfg = setup(args)
    register_lofar_datasets(cfg)
    
    if args.eval_only:
        model = LOFARTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = LOFARTrainer.test(cfg, model)
        return res

    trainer = LOFARTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
