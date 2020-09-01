import pandas as pd
from astropy.table import Table
from lofarnn.models.dataloaders.utils import get_lotss_objects

dat = get_lotss_objects("/home/s2153246/data/catalogues/pan_allwise.fits")
good_cols = [
    "AllWISE",
    "objID",
    "id",
    "objName",
    "z_best",
    "ra",
    "dec",
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
dat.write("/home/s2153246/data/combined_panstarr_allwise.fits", format="fits")
