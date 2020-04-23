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

os.environ["LOFARNN_ARCH"] = "XPS"
environment = os.environ["LOFARNN_ARCH"]
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation import COCOEvaluator,inference_on_dataset
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
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)#, mapper=SourceMapper(cfg))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)#, mapper=SourceMapper(cfg, False))

import pickle

# # Load and inspect our data
def get_lofar_dicts(annotation_filepath):
    with open(annotation_filepath, "rb") as f:
        dataset_dicts = pickle.load(f)
    return dataset_dicts


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg

cfg = get_cfg()
print("Load configuration file")
assert len(argv) > 1, "Insert path of configuration file when executing this script"
cfg.merge_from_file(argv[1])
EXPERIMENT_NAME= argv[2] + f'_size{cfg.INPUT.MIN_SIZE_TRAIN[0]}_prop{cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}_depth{cfg.MODEL.RESNETS.DEPTH}_batchSize{cfg.SOLVER.IMS_PER_BATCH}_anchorSize{cfg.MODEL.ANCHOR_GENERATOR.SIZES}'
DATASET_PATH= argv[3]
cfg.OUTPUT_DIR = os.path.join("/mnt/HDD/", "reports", EXPERIMENT_NAME)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Output path: {cfg.OUTPUT_DIR}")
print(f"Attempt to load training data from: {DATASET_PATH}")
multi = False
all_channel = False
precompute = True
for d in ["train", "val", "test"]:
    DatasetCatalog.register(f"{EXPERIMENT_NAME}_" + d,
                            lambda d=d: get_lofar_dicts(os.path.join(DATASET_PATH, f"json_{d}_prop{precompute}_all{all_channel}_multi{multi}.pkl")))
    MetadataCatalog.get(f"{EXPERIMENT_NAME}_" + d).set(thing_classes=["Optical source"])
lofar_metadata = MetadataCatalog.get("train")
'''
#cfg.MODEL.WEIGHTS = os.path.join("/home/jacob/Development/lofarnn/reports/frcnn_long_size200_prop4096_depth101_batchSize2_anchorSize[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 26, 32, 48, 64, 80, 96, 112, 128, 256, 512]]", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
import numpy as np
import random
import cv2
import imgaug
from detectron2.utils.visualizer import Visualizer
from detectron2.structures.instances import Instances
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt

dataset_dicts = get_lofar_dicts(os.path.join(DATASET_PATH, f"json_test.pkl"))
for i, d in enumerate(random.sample(dataset_dicts, 20)):
    im = cv2.imread(d["file_name"])
    source = BoundingBox(d["annotations"][0]["bbox"][0],d["annotations"][0]["bbox"][1],d["annotations"][0]["bbox"][2],d["annotations"][0]["bbox"][3])
    bbs = BoundingBoxesOnImage([source], shape=im.shape)
    image_bbs = bbs.draw_on_image(im, size=1, alpha=1, color=(255,105,180))
    outputs = predictor(im)
    print(outputs['instances'])
    pred_boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    if len(pred_boxes) > 0:
        pred_bbs = BoundingBoxesOnImage([BoundingBox(pred_boxes[0][0],pred_boxes[0][1],
                                                    pred_boxes[0][2],pred_boxes[0][3])], shape=im.shape)
        image_all = pred_bbs.draw_on_image(image_bbs, size=1, color=(255,255,255), alpha=1)
    else:
        image_all = image_bbs
    plt.imshow(image_all)
    plt.savefig(f"/home/jacob/Development/test_{i}.png")
    plt.clf()
    plt.cla()
exit()
'''
# # Train mode

cfg.DATASETS.TRAIN = (f"{EXPERIMENT_NAME}_train",)
cfg.DATASETS.VAL = (f"{EXPERIMENT_NAME}_val",)
cfg.DATASETS.TEST = (f"{EXPERIMENT_NAME}_test",)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
    f.write(cfg.dump())
trainer = Trainer(cfg)

trainer.resume_or_load(resume=False)

trainer.train()

print('Done training')

#cfg.MODEL.WEIGHTS = os.path.join("frcnn_long_size200_prop4096_depth101_batchSize2_anchorSize[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 26, 32, 48, 64, 80, 96, 112, 128, 256, 512]]", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
import numpy as np
import random
import cv2
import imgaug
from detectron2.utils.visualizer import Visualizer
dataset_dicts = get_lofar_dicts(os.path.join(DATASET_PATH, f"json_test.pkl"))
for i, d in enumerate(random.sample(dataset_dicts, 10)):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(outputs)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(f'test_iou_{i}.png',v.get_image()[:, :, ::-1])

print("Evaluate performance for validation set")

# returns a torch DataLoader, that loads the given detection dataset,
# with test-time transformation and batching.
val_loader = build_detection_test_loader(cfg, f"{EXPERIMENT_NAME}_val")

imsize = cfg.INPUT.MAX_SIZE_TRAIN
evaluator = SourceEvaluator(f"{EXPERIMENT_NAME}_val", cfg, False)

# Val_loader produces inputs that can enter the model for inference,
# the results of which can be evaluated by the evaluator
# The return value is that which is returned by evaluator.evaluate()
predictions = inference_on_dataset(trainer.model, val_loader, evaluator)
print(predictions)

"""
#Test set evaluation
# returns a torch DataLoader, that loads the given detection dataset, 
# with test-time transformation and batching.
test_loader = build_detection_test_loader(cfg, f"{DATASET_NAME}_test")
#evaluator = COCOEvaluator("lofar_data1_val", cfg, False, output_dir="./output/")
my_dataset = get_lofar_dicts(os.path.join(base_path,f"VIA_json_test.pkl"))
imsize = 200
evaluator = LOFAREvaluator(f"{DATASET_NAME}_test", cfg, False, imsize, gt_data=None, overwrite=True)
            
# Val_loader produces inputs that can enter the model for inference, 
# the results of which can be evaluated by the evaluator
# The return value is that which is returned by evaluator.evaluate()
predictions = inference_on_dataset(trainer.model, test_loader, evaluator, overwrite=True)
# Create evaluator
"""