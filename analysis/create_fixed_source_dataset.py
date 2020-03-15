import os

from lofarnn.data.datasets import create_fixed_source_dataset
from lofarnn.utils.coco import create_coco_dataset

environment = os.environ["LOFARNN_ARCH"]

if environment == "ALICE":
    dr_two = "/home/s2153246/data/data/LoTSS_DR2/lofar-surveys.org/downloads/DR2/mosaics/"
    vac = '/home/s2153246/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    cutout_directory = "/home/s2153246/data/processed/fixed/"
    pan_wise_location = "/home/s2153246/data/catalogues/pan_allwise.fits"
else:
    pan_wise_location = "/home/jacob/hetdex_ps1_allwise_photoz_v0.6.fits"
    dr_two = "/run/media/jacob/34b36a2c-5b42-41cd-a1fa-7a09e5414860/lofar-surveys.org/downloads/DR2/mosaics/"
    vac = '/home/jacob/Development/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    cutout_directory = "/home/jacob/Development/LOFAR-ML/data/processed/fixed/"

create_fixed_source_dataset(cutout_directory=cutout_directory,
                            pan_wise_location=pan_wise_location,
                            value_added_catalog_location=vac,
                            dr_two_location=dr_two,
                            fixed_size=300,
                            use_multiprocessing=False)
create_coco_dataset(root_directory=cutout_directory, multiple_bboxes=False)