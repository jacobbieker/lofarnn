import os

import numpy as np

from lofarnn.data.datasets import create_source_dataset
from lofarnn.utils.cnn import create_cnn_dataset

try:
    environment = os.environ["LOFARNN_ARCH"]
except:
    os.environ["LOFARNN_ARCH"] = "XPS"
    environment = os.environ["LOFARNN_ARCH"]

if environment == "ALICE":
    dr_two = (
        "/home/s2153246/data/data/LoTSS_DR2/lofar-surveys.org/downloads/DR2/mosaics/"
    )
    vac = "/home/s2153246/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits"
    comp_cat = "/home/s2153246/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_v1.2.comp.fits"
    cutout_directory = "/home/s2153246/data/processed/fixed_lgz_cnn_final/"
    pan_wise_location = "/home/s2153246/data/combined_panstarr_allwise.fits"
    multi_process = True
else:
    pan_wise_location = "/home/jacob/combined_panstarr_allwise_flux.fits"
    dr_two = "/run/media/jacob/768E313E8E30F7E7/mosaics/"
    comp_cat = "/home/jacob/Downloads/LOFAR_HBA_T1_DR1_merge_ID_v1.2.comp.fits"
    vac = (
        "/home/jacob/Downloads/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2b_restframe.fits"
    )
    cutout_directory = "/home/jacob/LoTSS_LGZ/"
    multi_process = False

rotation = 180
size = (300.0 / 3600.0) * np.sqrt(2)
print(size)

create_source_dataset(
    cutout_directory=cutout_directory,
    pan_wise_location=pan_wise_location,
    value_added_catalog_location=vac,
    dr_two_location=dr_two,
    component_catalog_location=comp_cat,
    use_multiprocessing=multi_process,
    all_channels=True,
    filter_lgz=True,
    fixed_size=size,
    no_source=False,
    filter_optical=True,
    strict_filter=False,
    verbose=False,
    radio_only=True
)

#exit()
create_cnn_dataset(
    root_directory=cutout_directory,
    counterpart_catalog=pan_wise_location,
    rotation=rotation,
    convert=False,
    vac_catalog=vac,
    normalize=True,
    multi_rotate_only=vac,
    resize=None,
)
exit()
create_cnn_dataset(
    root_directory=cutout_directory,
    counterpart_catalog=pan_wise_location,
    rotation=rotation,
    convert=False,
    all_channels=True,
    vac_catalog=vac,
    normalize=False,
    multi_rotate_only=vac,
    resize=None,
)
