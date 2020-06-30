from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os
from lofarnn.models.dataloaders.utils import get_lofar_dicts
# import some common detectron2 utilities

try:
    environment = os.environ["LOFARNN_ARCH"]
except:
    os.environ["LOFARNN_ARCH"] = "XPS"
    environment = os.environ["LOFARNN_ARCH"]
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, DefaultPredictor
from lofarnn.models.evaluators.SourceEvaluator import SourceEvaluator
from lofarnn.models.evaluators.LossEvalHook import LossEvalHook
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from lofarnn.models.dataloaders.SourceMapper import SourceMapper
from lofarnn.models.dataloaders.utils import make_physical_dict
from sys import argv
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg