import os
import numpy as np
#from lofarnn.data.datasets import create_variable_source_dataset
from lofarnn.utils.coco import create_coco_dataset
from lofarnn.data.datasets import create_variable_source_dataset

os.environ["LOFARNN_ARCH"] = "XPS"

environment = os.environ["LOFARNN_ARCH"]

cutout_directory = "/mnt/LargeSSD/variable_lgz/"
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)

cutout_directory = "/mnt/LargeSSD/fixed_lgz/"
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)

cutout_directory = "/mnt/LargeSSD/fixed_lgz_fluxlimit/"
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)

cutout_directory = "/mnt/LargeSSD/variable_lgz_fluxlimit/"
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)

cutout_directory = "/mnt/LargeSSD/variable_lgz_nooptical/"
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)

cutout_directory = "/mnt/LargeSSD/fixed_lgz_nooptical/"
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)

cutout_directory = "/mnt/LargeSSD/fixed_lgz_fluxlimit_nooptical/"
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)

cutout_directory = "/mnt/LargeSSD/variable_lgz_fluxlimit_nooptical/"
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)