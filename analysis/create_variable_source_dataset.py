import os

from lofarnn.data.datasets import create_variable_source_dataset
from lofarnn.utils.coco import create_coco_dataset

#os.environ["LOFARNN_ARCH"] = "XPS"

environment = os.environ["LOFARNN_ARCH"]

if environment == "ALICE":
    dr_two = "/home/s2153246/data/data/LoTSS_DR2/lofar-surveys.org/downloads/DR2/mosaics/"
    vac = '/home/s2153246/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    cutout_directory = "/home/s2153246/data/processed/variable_fixed/"
    pan_wise_location = "/home/s2153246/data/catalogues/pan_allwise.fits"
    multi_process = True
else:
    pan_wise_location = "/home/jacob/hetdex_ps1_allwise_photoz_v0.6.fits"
    dr_two = "/run/media/jacob/SSD_Backup/mosaics/"
    vac = '/home/jacob/Development/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    cutout_directory = "/home/jacob/Development/LOFAR-ML/data/processed/variable_fixed2/"
    multi_process = False

create_variable_source_dataset(cutout_directory=cutout_directory,
                               pan_wise_location=pan_wise_location,
                               value_added_catalog_location=vac,
                               dr_two_location=dr_two,
                               use_multiprocessing=multi_process)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=True, resize=200)