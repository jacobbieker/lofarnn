import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from lofarnn.models.dataloaders.utils import get_lotss_objects

dat = get_lotss_objects("/home/s2153246/data/dr2_combined.fits")

for col in ["MAG_R",
            "MAG_W1",
            "MAG_W2"]:
    plt.hist(np.nan_to_num(dat[col]), bins=np.arange(np.nanpercentile(dat[col], 2), np.nanpercentile(dat[col], 98), 0.5))
    plt.title(col)
    plt.savefig(f"Legacy_{col}.png", dpi=300)
    print(f"{col}: Max: {np.nanmax(dat[col])}, Min: {np.nanmin(dat[col])}, Median: {np.nanmedian(dat[col])}, "
          f"1st Percentile: {np.nanpercentile(dat[col], 1)}, 5th Percentile: {np.nanpercentile(dat[col], 5)}, 95th Percentile: {np.nanpercentile(dat[col], 95)}, 99th Percentile: {np.nanpercentile(dat[col], 99)}")
exit()
