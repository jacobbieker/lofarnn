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
from detectron2.evaluation import COCOEvaluator
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
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)


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
EXPERIMENT_NAME= argv[2]
DATASET_PATH= argv[3]
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Output path: {cfg.OUTPUT_DIR}")
print(f"Attempt to load training data from: {DATASET_PATH}")

for d in ["train", "val", "test"]:
    DatasetCatalog.register(f"{EXPERIMENT_NAME}_" + d,
                            lambda d=d: get_lofar_dicts(os.path.join(DATASET_PATH, f"json_{d}.pkl")))
    MetadataCatalog.get(f"{EXPERIMENT_NAME}_" + d).set(thing_classes=["Optical source"])
lofar_metadata = MetadataCatalog.get("train")

import pickle

# # Train mode

# To implement the LOFAR relevant metrics I changed
# DefaultTrainer into LOFARTrainer
# where the latter calls LOFAREvaluator within build_hooks instead of the default evaluator
# this works for the after the fact test eval
# for train eval those things are somewhere within a model 
# specifically a model that takes data and retuns a dict of losses

cfg.DATASETS.TRAIN = (f"{EXPERIMENT_NAME}_train",)
cfg.DATASETS.VAL = (f"{EXPERIMENT_NAME}_val",)
cfg.DATASETS.TEST = (f"{EXPERIMENT_NAME}_test",)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = Trainer(cfg)

trainer.resume_or_load(resume=False)

trainer.train()

# ### trainer.storage.history('loss_cls').latest()

print('Done training')

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
import numpy as np
import random
import cv2
from detectron2.utils.visualizer import Visualizer
dataset_dicts = get_lofar_dicts(os.path.join(DATASET_PATH, f"json_test.pkl"))
for d in random.sample(dataset_dicts, 50):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=lofar_metadata,
                   scale=4.0,
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    i = np.random.randint(0,50)
    cv2.imwrite(f'test_{i}.png',v.get_image()[:, :, ::-1])

"""
# Look at training curves in tensorboard:
get_ipython().run_line_magic('load_ext', 'tensorboard')
#%tensorboard --logdir output --host "0.0.0.0" --port 6006
get_ipython().run_line_magic('tensorboard', '--logdir output  --port 6006')
# In local command line input 
#ssh -X -N -f -L localhost:8890:localhost:6006 tritanium
# Then open localhost:8890 to see tensorboard
"""

# # Inference mode


"""
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
#cfg.DATASETS.TEST = (f"{DATASET_NAME}_", )
predictor = DefaultPredictor(cfg)




from detectron2.utils.visualizer import ColorMode
random.seed(5455)
aap = get_lofar_dicts(os.path.join(base_path,f"VIA_json_test.pkl"))
for d in random.sample(aap, 60):
    #print(d["file_name"])
    if not d["file_name"].endswith('_rotated0deg.png'):
        continue
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(outputs["instances"])
    v = Visualizer(im[:, :, ::-1],
                   metadata=lofar_metadata, 
                   scale=1, 
                  instance_mode=ColorMode.IMAGE #_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(10,10))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()





#Val set evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LOFAREvaluator
from detectron2.data import build_detection_test_loader

# returns a torch DataLoader, that loads the given detection dataset, 
# with test-time transformation and batching.
val_loader = build_detection_test_loader(cfg, f"{DATASET_NAME}_val")

#evaluator = COCOEvaluator("lofar_data1_val", cfg, False, output_dir="./output/")
my_dataset = get_lofar_dicts(os.path.join(base_path,f"VIA_json_val.pkl"))

imsize = 200
evaluator = LOFAREvaluator(f"{DATASET_NAME}_val", cfg, False,imsize, gt_data=None, overwrite=True)
            
# Val_loader produces inputs that can enter the model for inference, 
# the results of which can be evaluated by the evaluator
# The return value is that which is returned by evaluator.evaluate()
predictions = inference_on_dataset(trainer.model, val_loader, evaluator, overwrite=True)

# Create evaluator
#1.28, 70.44, 8.83, 5.84
#1.16, 69.71, 9.64, 6.93




#Test set evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LOFAREvaluator
from detectron2.data import build_detection_test_loader

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




def baseline(single, multi):
    total = single + multi
    correct = single/total
    print(f"Baseline assumption cat is {correct:.1%} correct")
    return correct

def our_score(single, multi,score_dict):
    fail_single = score_dict['assoc_single_fail_fraction']*single + score_dict['unassoc_single_fail_fraction']*single
    fail_multi = score_dict['assoc_multi_fail_fraction']*multi + score_dict['unassoc_multi_fail_fraction']*multi
    total = single + multi
    correct = (total-(fail_single+fail_multi))/total
    print(f"Our cat is {correct:.1%} correct")
    return correct
def improv(baseline, our_score):
    print(f"{(our_score-baseline)/baseline:.2%} improvement")
    
test_score_dict = {'assoc_single_fail_fraction': 0.0012224938875305957, 'assoc_multi_fail_fraction': 0.3433242506811989, 
                   'unassoc_single_fail_fraction': 0.1136919315403423, 'unassoc_multi_fail_fraction': 0.10899182561307907}
single, multi = 818,367
baseline = baseline(single, multi)
our_score = our_score(single, multi,test_score_dict)
improv(baseline, our_score)




test_score_dict = {'assoc_single_fail_fraction': 0.0012224938875305957, 'assoc_multi_fail_fraction': 0.3433242506811989, 'unassoc_single_fail_fraction': 0.1136919315403423, 'unassoc_multi_fail_fraction': 0.10899182561307907}
"""
