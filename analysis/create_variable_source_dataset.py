import os
import numpy as np
#from lofarnn.data.datasets import create_variable_source_dataset
from lofarnn.utils.coco import create_coco_dataset
from lofarnn.data.datasets import create_variable_source_dataset

os.environ["LOFARNN_ARCH"] = "XPS"

environment = os.environ["LOFARNN_ARCH"]

if environment == "ALICE":
    dr_two = "/home/s2153246/data/data/LoTSS_DR2/lofar-surveys.org/downloads/DR2/mosaics/"
    vac = '/home/s2153246/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    cutout_directory = "/home/s2153246/data/processed/variable_fixed_all_channels/"
    pan_wise_location = "/home/s2153246/data/catalogues/pan_allwise.fits"
    multi_process = True
else:
    pan_wise_location = "/run/media/jacob/SSD_Backup/hetdex_ps1_allwise_photoz_v0.6.fits"
    dr_two = "/run/media/jacob/SSD_Backup/mosaics/"
    vac = '/run/media/jacob/SSD_Backup/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    cutout_directory = "/home/jacob/variable_lgz_nooptical/"
    multi_process = True
create_variable_source_dataset(cutout_directory=cutout_directory,
                               pan_wise_location=pan_wise_location,
                               value_added_catalog_location=vac,
                               dr_two_location=dr_two,
                               use_multiprocessing=multi_process,
                               all_channels=True,
                               filter_lgz=True,
                               no_source=True,
                               filter_optical=True,
                               num_threads=12,
                               strict_filter=False,
                               verbose=False,
                               gaussian=False)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)
import os
import numpy as np
#from lofarnn.data.datasets import create_variable_source_dataset
from lofarnn.utils.coco import create_coco_dataset
from lofarnn.data.datasets import create_variable_source_dataset

os.environ["LOFARNN_ARCH"] = "XPS"

environment = os.environ["LOFARNN_ARCH"]

if environment == "ALICE":
    dr_two = "/home/s2153246/data/data/LoTSS_DR2/lofar-surveys.org/downloads/DR2/mosaics/"
    vac = '/home/s2153246/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    cutout_directory = "/home/s2153246/data/processed/variable_fixed_all_channels/"
    pan_wise_location = "/home/s2153246/data/catalogues/pan_allwise.fits"
    multi_process = True
else:
    pan_wise_location = "/run/media/jacob/SSD_Backup/hetdex_ps1_allwise_photoz_v0.6.fits"
    dr_two = "/run/media/jacob/SSD_Backup/mosaics/"
    vac = '/run/media/jacob/SSD_Backup/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    cutout_directory = "/home/jacob/variable_lgz_fluxlimit_nooptical/"
    multi_process = True
create_variable_source_dataset(cutout_directory=cutout_directory,
                               pan_wise_location=pan_wise_location,
                               value_added_catalog_location=vac,
                               dr_two_location=dr_two,
                               use_multiprocessing=multi_process,
                               all_channels=True,
                               filter_lgz=True,
                               no_source=True,
                               filter_optical=True,
                               strict_filter=False,
                               verbose=False,
                               gaussian=False)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=False, precomputed_proposals=False, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=True, resize=400)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False, rotation=None, convert=True, all_channels=True, precomputed_proposals=False, resize=400)
