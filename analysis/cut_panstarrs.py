import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from lofarnn.models.dataloaders.utils import get_lotss_objects

<<<<<<< Updated upstream
dat = get_lotss_objects("/home/jacob/combined_panstarr_allwise_flux.fits")
print(dat.columns)
# Get Mag_R values
r_mag = np.nan_to_num(dat["rFApMag"])
frac_value = len(r_mag[r_mag > 0]) / len(r_mag)
print(f"Frac: {frac_value}")
print(f"Percentiles: {np.nanpercentile(r_mag, [1,5,25,50,75,95,99])}")
exit()
for col in ["iFApFlux",
            "w1Flux",
            "gFApFlux",
            "rFApFlux",
            "zFApFlux",
            "yFApFlux",
            "w2Flux",
            "w3Flux",
            "w4Flux", ]:
=======
dat = get_lotss_objects("/home/s2153246/data/dr2_combined.fits")

for col in ["MAG_R",
            "MAG_W1",
            "MAG_W2"]:
>>>>>>> Stashed changes
    plt.hist(np.nan_to_num(dat[col]), bins=np.arange(np.nanpercentile(dat[col], 2), np.nanpercentile(dat[col], 98), 0.5))
    plt.title(col)
    plt.show()
    print(f"{col}: Max: {np.nanmax(dat[col])}, Min: {np.nanmin(dat[col])}, Median: {np.nanmedian(dat[col])}, "
          f"1st Percentile: {np.nanpercentile(dat[col], 1)}, 5th Percentile: {np.nanpercentile(dat[col], 5)}, 95th Percentile: {np.nanpercentile(dat[col], 95)}, 99th Percentile: {np.nanpercentile(dat[col], 99)}")
exit()
dat = get_lotss_objects("/home/jacob/hetdex_ps1_allwise_photoz_v0.6.fits")
print(dat.columns)
exit()
good_cols = [
    "AllWISE",
    "objID",
    "id",
    "objName",
    "z_best",
    "ra",
    "dec",
    "iFApFlux",
    "w1Flux",
    "gFApFlux",
    "rFApFlux",
    "zFApFlux",
    "yFApFlux",
    "w2Flux",
    "w3Flux",
    "w4Flux",
    "iFApMag",
    "w1Mag",
    "gFApMag",
    "rFApMag",
    "zFApMag",
    "yFApMag",
    "w2Mag",
    "w3Mag",
    "w4Mag",
]
dat = dat[good_cols]
dat.write("/home/s2153246/data/combined_panstarr_allwise_flux.fits", format="fits")
