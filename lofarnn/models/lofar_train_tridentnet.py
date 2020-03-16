# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import pickle

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.engine import LOFARTrainer, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode


def add_tridentnet_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.TRIDENT = CN()

    # Number of branches for TridentNet.
    _C.MODEL.TRIDENT.NUM_BRANCH = 3
    # Specify the dilations for each branch.
    _C.MODEL.TRIDENT.BRANCH_DILATIONS = [1, 2, 3]
    # Specify the stage for applying trident blocks. Default stage is Res4 according to the
    # TridentNet paper.
    _C.MODEL.TRIDENT.TRIDENT_STAGE = "res4"
    # Specify the test branch index TridentNet Fast inference:
    #   - use -1 to aggregate results of all branches during inference.
    #   - otherwise, only using specified branch for fast inference. Recommended setting is
    #     to use the middle branch.
    _C.MODEL.TRIDENT.TEST_BRANCH_IDX = 1


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
    cfg.merge_from_file("/home/jacob/Development/LOFAR-ML/lofarnn/models/source_tridentnet_fast_R_101_C4_3x.yaml")
    DATASET_NAME= "fixed"
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
    DATASET_NAME= "fixed"
    base_path = f"/home/jacob/Development/LOFAR-ML/data/processed/fixed/COCO/annotations/"

    from detectron2.data import DatasetCatalog, MetadataCatalog
    for d in ["train", "val", "test"]:
        DatasetCatalog.register(f"{DATASET_NAME}_" + d, 
                                lambda d=d: get_lofar_dicts(os.path.join(base_path,f"json_{d}.pkl")))
        MetadataCatalog.get(f"{DATASET_NAME}_" + d).set(thing_classes=["Optical_Source"])
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
