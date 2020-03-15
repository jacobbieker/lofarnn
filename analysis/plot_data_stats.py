import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import lofarnn.visualization.cutouts as vis
from lofarnn.data.datasets import get_lotss_objects
import numpy as np
from astropy.io import fits
import os
from pathlib import Path

os.environ["LOFARNN_ARCH"] = "XPS"
environment = os.environ["LOFARNN_ARCH"]

if environment == "ALICE":
    vac = '/home/s2153246/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    cutout_directory = "/home/s2153246/data/processed/fixed/"
    pan_wise_location = "/home/s2153246/data/catalogues/pan_allwise.fits"
    prefix = "/home/s2153246/data/"
else:
    vac = '/home/jacob/Development/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    cutout_directory = "/home/jacob/Development/LOFAR-ML/data/processed/fixed/"
    pan_wise_location = "/home/jacob/hetdex_ps1_allwise_photoz_v0.6.fits"
    prefix = "/home/jacob/Development/LOFAR-ML/reports/"


l_objects = get_lotss_objects(vac, True)
l_objects = l_objects[~np.isnan(l_objects['LGZ_Size'])]
l_objects = l_objects[~np.isnan(l_objects["ID_ra"])]
mosaic_names = set(l_objects["Mosaic_ID"])

# Open the Panstarrs and WISE catalogue
pan_wise_catalogue = fits.open(pan_wise_location, memmap=True)
pan_wise_catalogue = pan_wise_catalogue[1].data
i_mag = pan_wise_catalogue["iFApMag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel('iFApMag')
plt.savefig(os.path.join(prefix,"iFApMagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["w1Mag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel('w1Mag')
plt.savefig(os.path.join(prefix,"w1MagDist.png"))
plt.close()
del i_mag
plt.hist(l_objects["LGZ_Size"], density=True, bins=40)
plt.xlabel("LGZ_Size")
plt.savefig(os.path.join(prefix,"LGZSizeDist.png"))
plt.close()

image_paths = Path(cutout_directory).rglob("*.npy")
fig, ax = vis.plot_cutout_and_bboxes(next(image_paths), "Debug")
plt.savefig(os.path.join(prefix, "Debug_test.png"))
plt.close()
vis.plot_statistics(image_paths, save_path=prefix)
