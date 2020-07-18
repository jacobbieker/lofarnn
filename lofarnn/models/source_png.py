#!/usr/bin/env python

# # Import and load Detectron2 and libraries


# Some basic setup
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os

# import some common detectron2 utilities

from detectron2.evaluation import COCOEvaluator

try:
    environment = os.environ["LOFARNN_ARCH"]
except:
    os.environ["LOFARNN_ARCH"] = "XPS"
    environment = os.environ["LOFARNN_ARCH"]
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    DefaultPredictor,
)
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from lofarnn.models.evaluators.SourceEvaluator import SourceEvaluator
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from lofarnn.models.dataloaders.SourceMapper import SourceMapper
from sys import argv


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)  # , mapper=SourceMapper(cfg))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(
            cfg, dataset_name
        )  # , mapper=SourceMapper(cfg, False))


import pickle

# # Load and inspect our data
def get_lofar_dicts(annotation_filepath, fraction=1.0):
    with open(annotation_filepath, "rb") as f:
        dataset_dicts = pickle.load(f)
    if fraction < 0.99999:
        # Only take subset of the dataset
        num_entries = len(dataset_dicts)
        num_kept = int(fraction * num_entries)
        step_size = int(num_entries / num_kept)
        new_dicts = []
        for i in range(len(dataset_dicts), step=step_size):
            new_dicts.append(dataset_dicts[i])
        dataset_dicts = new_dicts
    return dataset_dicts


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg

cfg = get_cfg()
print("Load configuration file")
assert len(argv) > 1, "Insert path of configuration file when executing this script"
cfg.merge_from_file(argv[1])
FRACTION = float(argv[4])
EXPERIMENT_NAME = (
    argv[2]
    + f"_size{cfg.INPUT.MIN_SIZE_TRAIN[0]}_prop{cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}_depth{cfg.MODEL.RESNETS.DEPTH}_batchSize{cfg.SOLVER.IMS_PER_BATCH}_anchorSize{cfg.MODEL.ANCHOR_GENERATOR.SIZES}_frac{FRACTION}"
)
DATASET_PATH = argv[3]
if environment == "XPS":
    cfg.OUTPUT_DIR = os.path.join("/mnt/10tb/", "reports", EXPERIMENT_NAME)
else:
    cfg.OUTPUT_DIR = os.path.join("/home/s2153246/data/", "reports", EXPERIMENT_NAME)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Output path: {cfg.OUTPUT_DIR}")
print(f"Attempt to load training data from: {DATASET_PATH}")
multi = False
all_channel = False
precompute = True
semseg = False
norm = True
# Register train set with fraction
for d in ["train"]:
    DatasetCatalog.register(
        f"{argv[2]}_" + d,
        lambda d=d: get_lofar_dicts(
            os.path.join(
                DATASET_PATH,
                f"json_{d}_prop{precompute}_all{all_channel}_multi{multi}_seg{semseg}_norm{norm}.pkl",
            ),
            fraction=FRACTION,
        ),
    )
    MetadataCatalog.get(f"{argv[2]}_" + d).set(thing_classes=["Optical source"])
# Keep val and test set the same so that its always testing on the same stuff
for d in ["val", "test"]:
    DatasetCatalog.register(
        f"{argv[2]}_" + d,
        lambda d=d: get_lofar_dicts(
            os.path.join(
                DATASET_PATH,
                f"json_{d}_prop{precompute}_all{all_channel}_multi{multi}_seg{semseg}_norm{norm}.pkl",
            ),
            fraction=1,
        ),
    )
    MetadataCatalog.get(f"{argv[2]}_" + d).set(thing_classes=["Optical source"])
lofar_metadata = MetadataCatalog.get("train")

cfg.DATASETS.TRAIN = (f"{argv[2]}_train",)
cfg.DATASETS.VAL = (f"{argv[2]}_test",)
cfg.DATASETS.TEST = (
    f"{argv[2]}_val",
)  # Swapped because TEST is used for eval, and val is not, but can be used later
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
    f.write(cfg.dump())
trainer = Trainer(cfg)

trainer.resume_or_load(resume=True)

trainer.train()

print("Done training")
