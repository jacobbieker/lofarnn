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
    pan_wise_location = "/run/media/jacob/SSD_Backup/hetdex_ps1_allwise_photoz_v0.6.fits"
    dr_two = "/run/media/jacob/SSD_Backup/mosaics/"
    vac = '/run/media/jacob/SSD_Backup/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    cutout_directory = "/run/media/jacob/SSD_Backup/variable_fixed_all_channels/"
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
i_mag = pan_wise_catalogue["gFApMag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel('gFApMag')
plt.savefig(os.path.join(prefix,"gFApMagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["rFApMag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel('rFApMag')
plt.savefig(os.path.join(prefix,"rFApMagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["zFApMag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel('zFApMag')
plt.savefig(os.path.join(prefix,"zFApMagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["yFApMag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel('yFApMag')
plt.savefig(os.path.join(prefix,"yFApMagDist.png"))
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
i_mag = pan_wise_catalogue["w2Mag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel('w2Mag')
plt.savefig(os.path.join(prefix,"w2MagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["w3Mag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel('w3Mag')
plt.savefig(os.path.join(prefix,"w3MagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["w4Mag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel('w4Mag')
plt.savefig(os.path.join(prefix,"w4MagDist.png"))
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
