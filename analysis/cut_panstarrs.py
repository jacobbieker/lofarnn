import pandas as pd
from astropy.table import Table
from lofarnn.models.dataloaders.utils import get_lotss_objects

dat = get_lotss_objects('/home/jacob/hetdex_ps1_allwise_photoz_v0.6.fits')
good_cols = ['ra',
             'dec',
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
cols = list(dat.columns)
bad_cols = [c for c in cols if not c in good_cols]
dat = dat[good_cols]
dat.write("combined_panstarr_allwise.fits", format="fits")