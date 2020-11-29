import os

from lofarnn.data.datasets import create_source_dataset
from lofarnn.utils.coco import create_coco_dataset

os.environ["LOFARNN_ARCH"] = "XPS"

environment = os.environ["LOFARNN_ARCH"]

if environment == "ALICE":
    dr_two = (
        "/home/s2153246/data/data/LoTSS_DR2/lofar-surveys.org/downloads/DR2/mosaics/"
    )
    vac = "/home/s2153246/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits"
    cutout_directory = "/home/s2153246/data/processed/fixed_all_channels/"
    pan_wise_location = "/home/s2153246/data/catalogues/pan_allwise.fits"
    multi_process = True
else:
    pan_wise_location = (
        "/run/media/jacob/SSD_Backup/hetdex_ps1_allwise_photoz_v0.6.fits"
    )
    dr_two = "/run/media/jacob/SSD_Backup/mosaics/"
    vac = "/run/media/jacob/SSD_Backup/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits"
    cutout_directory = "/home/jacob/gaussian_all/"
    multi_process = True

create_source_dataset(
    cutout_directory=cutout_directory,
    pan_wise_location=pan_wise_location,
    value_added_catalog_location=vac,
    dr_two_location=dr_two,
    use_multiprocessing=multi_process,
    all_channels=True,
    verbose=False,
    fixed_size=300 / 3600.0,
    filter_lgz=False,
    gaussian=True,
)
# (0,15,30,45,60,75,90,105,120,135,150,165,180)
create_coco_dataset(
    root_directory=cutout_directory,
    multiple_bboxes=True,
    rotation=None,
    convert=True,
    all_channels=False,
    resize=200,
)
