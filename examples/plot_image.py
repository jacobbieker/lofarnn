from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from lofarnn.data.cutouts import convert_to_valid_color

from lofarnn.models.dataloaders.utils import get_lotss_objects

lobjects = get_lotss_objects("/home/bieker/Downloads/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits")

source = lobjects[lobjects["Source_Name"] == "ILTJ105737.86+561810.0"]

#print(source["RA"])
#print(source["DEC"])
#source_coord = SkyCoord(source["RA"], source["DEC"], unit="deg")
#print(source_coord)
#exit()

onlyfiles = [
    f
    for f in listdir("/home/bieker/LoTSS_DR1_999_Check/COCO/all/")
    if isfile(join("/home/bieker/LoTSS_DR1_999_Check/COCO/all/", f))
]

for f in onlyfiles:
    data = np.load(
        join("/home/bieker/LoTSS_DR1_999_Check/COCO/all/", f), allow_pickle=True, fix_imports=True
    )
    #print(data)
    #exit()
    wcs = data[3]
    img = np.moveaxis(data[0], 0, -1)
    print(img.shape)
    if img.shape[0] >= 8:
        img[0] = convert_to_valid_color(img[0])
        #img[:, :, 0] = convert_to_valid_color(
        #    img[:, :, 0],
        #    clip=True,
        #    lower_clip=0.0,
        #    upper_clip=500,
        #    normalize=True,
        #    scaling=None,
        #)
        #img[:, :, 1] = convert_to_valid_color(
        #    img[:, :, 1],
        #    clip=True,
        #    lower_clip=10.0,
        #    upper_clip=30.0,
        #    normalize=True,
        #    scaling=None,
        #)
        #img[:, :, 2] = convert_to_valid_color(
        #    img[:, :, 2],
        #    clip=True,
        #    lower_clip=10.0,
        #    upper_clip=30.0,
        #    normalize=True,
        #    scaling=None,
        #)
        img = img#[41:-41, 41:-41]
        plt.subplot(projection=wcs)
        plt.title(f.split(".npy")[0])
        plt.xlabel("RA")
        plt.ylabel("DEC")
        plt.imshow(img[:, :, :3])
        plt.savefig(f"{f.split('.npy')[0]}_999.png", dpi=300)
        plt.show()
        plt.cla()
        plt.clf()
