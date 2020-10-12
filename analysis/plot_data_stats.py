import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import lofarnn.visualization.cutouts as vis
from astropy.table import Table


def get_lotss_objects(fname, verbose=False):
    """
    Load the LoTSS objects from a file
    """

    with fits.open(fname) as hdul:
        table = hdul[1].data

    if verbose:
        print(table.columns)

    # convert from astropy.io.fits.fitsrec.FITS_rec to astropy.table.table.Table
    return Table(table)


import numpy as np
from astropy.io import fits
import os
from pathlib import Path

os.environ["LOFARNN_ARCH"] = "XPS"
environment = os.environ["LOFARNN_ARCH"]

if environment == "ALICE":
    vac = "/home/s2153246/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits"
    cutout_directory = "/home/s2153246/data/processed/fixed/"
    pan_wise_location = "/home/s2153246/data/catalogues/pan_allwise.fits"
    prefix = "/home/s2153246/data/"
else:
    pan_wise_location = "/home/jacob/hetdex_ps1_allwise_photoz_v0.6.fits"
    dr_two = "/run/media/jacob/SSD_Backup/mosaics/"
    vac = "/home/jacob/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits"
    cutout_directory = "/run/media/jacob/SSD_Backup/variable_fixed_all_channels/"
    prefix = "/home/jacob/Development/test_lofarnn/"


l_objects = get_lotss_objects(vac, True)
l_objects = l_objects[~np.isnan(l_objects["LGZ_Size"])]
l_objects = l_objects[~np.isnan(l_objects["ID_ra"])]
print(np.max(l_objects["LGZ_Size"].data / l_objects["LGZ_Width"].data))
print(np.min(l_objects["LGZ_Size"].data / l_objects["LGZ_Width"].data))
exit()
mosaic_names = set(l_objects["Mosaic_ID"])
plt.hist(l_objects["LGZ_Assoc"], bins=20)
plt.xlabel("Number of Components")
plt.yscale("log")
plt.ylabel("Number of Radio Sources")
plt.title("Number of Components per Radio Source")
plt.savefig(os.path.join(prefix, "LGZAssocDist.png"))
plt.close()
plt.hist(l_objects["LGZ_Size"], bins=20)
plt.xlabel("Size of radio object (Arcseconds)")
plt.yscale("log")
plt.axvline(x=300, c="r")
plt.ylabel("Number of Radio Sources")
plt.title("Size of Radio Sources")
plt.savefig(os.path.join(prefix, "LGZSizeDist.png"))
plt.close()
# plt.hist(l_objects["LGZ_Size"].data/l_objects["LGZ_Width"].data, bins=20)
# plt.xlabel("Axis Ratio")
# plt.yscale('log')
# plt.xlim(0,500)
# plt.ylabel("Number of Radio Sources")
# plt.title("Axis Ratio of Radio Sources")
# plt.savefig(os.path.join(prefix, "LGZAxisRatioDist.png"))
# plt.close()
plt.hist(l_objects["z_best"], bins=20)
plt.xlabel("Redshift (z)")
plt.yscale("log")
plt.ylabel("Number of Radio Sources")
plt.savefig(os.path.join(prefix, "LGZzDist.png"))
plt.close()
exit()
# Open the Panstarrs and WISE catalogue
pan_wise_catalogue = fits.open(pan_wise_location, memmap=True)
pan_wise_catalogue = pan_wise_catalogue[1].data

layers = [
    "iFApMag",
    "gFApMag",
    "rFApMag",
    "zFApMag",
    "yFApMag",
    "w1Mag",
    "w2Mag",
    "w3Mag",
    "w4Mag",
]

names = ["i", "g", "r", "z", "y", "w1", "w2", "w3", "w4"]

colors = ["b", "g", "r", "orange", "y", "pink", "m", "c", "gray"]
# Plot them all on top of each other
# for i, layer in enumerate(layers):
#    i_mag = l_objects[layer]
#    i_mag = i_mag[~np.isinf(i_mag)]
#    i_mag = i_mag[i_mag > -98]
#    plt.hist(i_mag, bins=20, fill=False, edgecolor=colors[i], label=names[i])
# plt.legend(loc='best')
# plt.title("Optical Counterpart Magnitude Distribution")
# plt.xlim(8,30)
# plt.yscale("log")
# plt.xlabel("Magnitude")
# plt.ylabel("Number of  optical sources")
# plt.savefig(os.path.join(prefix, "allMagsCounterparts.png"), dpi=300)
# plt.close()
# exit()
# del i_mag
i_mag = pan_wise_catalogue["gFApMag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel("gFApMag")
plt.savefig(os.path.join(prefix, "gFApMagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["rFApMag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel("rFApMag")
plt.savefig(os.path.join(prefix, "rFApMagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["zFApMag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel("zFApMag")
plt.savefig(os.path.join(prefix, "zFApMagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["yFApMag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel("yFApMag")
plt.savefig(os.path.join(prefix, "yFApMagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["w1Mag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel("w1Mag")
plt.savefig(os.path.join(prefix, "w1MagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["w2Mag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel("w2Mag")
plt.savefig(os.path.join(prefix, "w2MagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["w3Mag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel("w3Mag")
plt.savefig(os.path.join(prefix, "w3MagDist.png"))
plt.close()
del i_mag
i_mag = pan_wise_catalogue["w4Mag"]
i_mag = i_mag[~np.isinf(i_mag)]
i_mag = i_mag[i_mag > -98]
plt.hist(i_mag, density=True, bins=40)
plt.xlabel("w4Mag")
plt.savefig(os.path.join(prefix, "w4MagDist.png"))
plt.close()
del i_mag
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex="col", sharey="row")
ax1.hist(l_objects["LGZ_Size"], density=True, bins=20)
ax1.set_xlabel("Size")
plt.savefig(os.path.join(prefix, "LGZSizeDist.png"))
plt.close()
plt.hist(l_objects["z_best"], density=True, bins=20)
plt.xlabel("Redshift (z)")
plt.savefig(os.path.join(prefix, "LGZzDist.png"))
plt.close()
plt.hist(l_objects["LGZ_Assoc"], bins=20)
plt.xlabel("Number of Components")
plt.yscale("log")
plt.savefig(os.path.join(prefix, "LGZAssocDist.png"))
plt.close()
plt.hist(l_objects["LGZ_Width"], density=True, bins=20)
plt.xlabel("LGZ_Width")
plt.savefig(os.path.join(prefix, "LGZWidthDist.png"))
plt.close()
