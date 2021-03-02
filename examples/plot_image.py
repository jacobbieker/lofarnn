from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from lofarnn.data.cutouts import convert_to_valid_color
from astropy.io import fits
from astropy.table import Table
from lofarnn.models.dataloaders.utils import get_lotss_objects
from astropy import units as u
from radio_beam import Beam
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel, proj_plane_pixel_scales

lobjects = get_lotss_objects("/home/bieker/Downloads/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits")

gauss_catalog = "/home/bieker/Downloads/LOFAR_HBA_T1_DR1_catalog_v0.99.gaus.fits"
component_catalog = "/home/bieker/Downloads/LOFAR_HBA_T1_DR1_merge_ID_v1.2.comp.fits"
# Use component Name from comp catalog to select gaussian
gauss_cat = Table.read(gauss_catalog).to_pandas()
component_catalog = Table.read(component_catalog).to_pandas()

source = lobjects[lobjects["Source_Name"] == "ILTJ110249.44+502810.4"]
source_name = source["Source_Name"]
print(component_catalog.columns)
print(gauss_cat.columns)
exit()
comp_cat = component_catalog[
    component_catalog.Source_Name == source_name
    ]
non_sources = gauss_cat[
    gauss_cat.Source_Name != str.encode(source_name)
    ]  # Select all those not part of source
print(comp_cat)
exit()
data = fits.open("/home/bieker/Downloads/P10Hetdex-mosaic.fits")
print(data[0].header)
wcs = WCS(data[0].header)
beam = Beam.from_fits_header(data[0].header)
beams_per_pixel = proj_plane_pixel_scales(wcs)[0] ** 2 / (beam.minor * beam.major * 1.1331) * u.beam
print(beams_per_pixel)
print(beam)
print(wcs)
print(proj_plane_pixel_scales(wcs))
#exit()

# Need to convert to Jy from Jy/beam, so multiply by the beam
#exit()
print(source["RA"])
print(source["DEC"])
image = np.nan_to_num(data[0].data)
print(np.min(image))
print(np.max(image))
image *= u.Jy/u.beam #((proj_plane_pixel_scales(wcs)[0]*u.arcsec)**2)
image = image.to(u.Jy/u.beam)
print(np.sum(image))
image = image*beams_per_pixel
print(np.sum(image))
exit()
#print(np.sum(image.to(u.mJy/((proj_plane_pixel_scales(wcs)[0]*u.arcsec)**2), equivalencies=u.beam_angular_area(beam))))
#exit()
#value = source["Total_flux"][0] * u.mJy/((proj_plane_pixel_scales(wcs)[0]*u.arcsec)**2)
#value2 = value.to(u.mJy/u.beam, equivalencies=u.beam_angular_area(beam))
#print(value2)
#print(value.to(u.Jy))
#source_coord = SkyCoord(source["RA"], source["DEC"], unit="deg")
#print(source_coord)
#exit()

onlyfiles = [
    f
    for f in listdir("/home/bieker/LoTSS_DR1_Cleaned_110_Check2_NoCompCheck/COCO/all/")
    if isfile(join("/home/bieker/LoTSS_DR1_Cleaned_110_Check2_NoCompCheck/COCO/all/", f))
]

for f in onlyfiles:
    data = np.load(
        join("/home/bieker/LoTSS_DR1_Cleaned_110_Check2_NoCompCheck/COCO/all/", f), allow_pickle=True, fix_imports=True
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
