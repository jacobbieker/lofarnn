import argparse
import logging
import os
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.logger import setup_logger

from detectron2.engine import hooks
from detectron2.engine.train_loop import SimpleTrainer

from lofarnn.models.dataloaders.SourceMapper import SourceMapper
from lofarnn.models.evaluators.SourceEvaluator import SourceEvaluator
from lofarnn.models.evaluators.LossEvalHook import LossEvalHook
from detectron2.engine import DefaultTrainer


class SourceTrainer(DefaultTrainer):
    """
    Modified DefaultTrainer to add in what's needed for LOFAR Source Detection

    Adds support for evaluating on multiple cuts in the data, given by physical_dict

    """
    _physical_dicts = None

    def __init__(self, cfg, physical_dict):
        super(SourceTrainer, self).__init__(cfg)
        SourceTrainer._physical_dicts = physical_dict


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None, physical_dict=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if physical_dict is None:
            physical_dict = cls._physical_dicts
        return SourceEvaluator(dataset_name, cfg, True, output_folder, physical_cut_dict=physical_dict)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.cfg.TEST.EXTRA_EVAL,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                SourceMapper(self.cfg, True)
            ),
        ))
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, mapper=SourceMapper(cfg))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, mapper=SourceMapper(cfg, False))