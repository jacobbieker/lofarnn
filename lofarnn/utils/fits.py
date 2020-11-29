from typing import Any, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


def flatten(
    f: fits.HDUList,
    x: float,
    y: float,
    size: float,
    hduid: int = 0,
    channel: int = 0,
    freqaxis: int = 3,
    verbose: bool = True,
) -> fits.HDUList:
    """
    Flatten a fits file so that it becomes a 2D image. Return new header and
    data
    This version also makes a sub-image of specified size.
    """

    naxis = f[hduid].header["NAXIS"]
    if naxis < 2:
        raise RuntimeError("Can't make map from this")

    if verbose:
        print(f[hduid].data.shape)
    data_shape = f[hduid].data.shape[-2:]
    by, bx = data_shape
    xmin = int(x - size / 2)
    if xmin < 0:
        xmin = 0
    xmax = int(x + size / 2)
    if xmax > bx:
        xmax = bx
    ymin = int(y - size / 2)
    if ymin < 0:
        ymin = 0
    ymax = int(y + size / 2)
    if ymax > by:
        ymax = by

    if ymax <= ymin or xmax <= xmin:
        # this can only happen if the required position is not on the map
        print(xmin, xmax, ymin, ymax)
        raise RuntimeError("Failed to make subimage! Required position not on the map.")

    w = WCS(f[hduid].header)
    wn = WCS(naxis=2)

    wn.wcs.crpix[0] = w.wcs.crpix[0] - xmin
    wn.wcs.crpix[1] = w.wcs.crpix[1] - ymin
    wn.wcs.cdelt = w.wcs.cdelt[0:2]
    try:
        wn.wcs.pc = w.wcs.pc[0:2, 0:2]
    except AttributeError:
        pass  # pc is not present
    wn.wcs.crval = w.wcs.crval[0:2]
    wn.wcs.ctype[0] = w.wcs.ctype[0]
    wn.wcs.ctype[1] = w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"] = 2

    slice = []
    for i in range(naxis, 0, -1):
        if i == 1:
            slice.append(np.s_[xmin:xmax])
        elif i == 2:
            slice.append(np.s_[ymin:ymax])
        elif i == freqaxis:
            slice.append(channel)
        else:
            slice.append(0)
    if verbose:
        print(slice)

    hdu = fits.PrimaryHDU(f[hduid].data[tuple(slice)], header)
    copy = ("EQUINOX", "EPOCH", "BMAJ", "BMIN", "BPA")
    for k in copy:
        r = f[hduid].header.get(k)
        if r:
            hdu.header[k] = r
    if "TAN" in hdu.header["CTYPE1"]:
        hdu.header["LATPOLE"] = f[hduid].header["CRVAL2"]
    hdulist = fits.HDUList([hdu])
    return hdulist


def extract_subimage(
    filename: str,
    ra: float,
    dec: float,
    size: float,
    hduid: int = 0,
    verbose: bool = True,
) -> fits.HDUList:
    if verbose:
        print("Opening", filename)
    orighdu = fits.open(filename)
    psize = int((size / orighdu[hduid].header["CDELT2"]))
    if verbose:
        print(f"Size in Pixels: {psize}")
        print(f"Size in Arcseconds: {size}")

    ndims = orighdu[hduid].header["NAXIS"]
    pvect = np.zeros((1, ndims))
    lwcs = WCS(orighdu[hduid].header)
    pvect[0][0] = ra
    pvect[0][1] = dec
    imc = lwcs.wcs_world2pix(pvect, 0)
    x = imc[0][0]
    y = imc[0][1]
    hdu = flatten(orighdu, x, y, psize, hduid=hduid, verbose=verbose)
    return hdu


def determine_visible_catalogue_source_and_separation(
    ra: float, dec: float, size: float, catalogue: Table, verbose=False
) -> Tuple[Any, Any, Any, SkyCoord, SkyCoord]:
    """
    Find the sources in the catalogue that are visible in the cutout, and returns a smaller catalogue for that
    :param ra: Radio RA
    :param dec: Radio DEC
    :param wcs: WCS of Radio FITS files
    :param size: Size of cutout in degrees
    :param catalogue: Pan-AllWISE catalogue
    :param l_objects: LOFAR Value Added Catalogue objects
    :return: Subcatalog of catalogue that only contains sources near the radio source in the cutout size, as well as
    SkyCoord of their world coordinates
    """
    try:
        ra_array = np.array(catalogue["RA"], dtype=float)
        dec_array = np.array(catalogue["DEC"], dtype=float)
    except:
        ra_array = np.array(catalogue["ID_ra"], dtype=float)
        dec_array = np.array(catalogue["ID_dec"], dtype=float)
    sky_coords = SkyCoord(ra_array, dec_array, unit="deg")

    source_coord = SkyCoord(ra, dec, unit="deg")
    search_radius = size * u.deg
    d2d = source_coord.separation(sky_coords)
    catalogmask = d2d < search_radius
    idxcatalog = np.where(catalogmask)[0]
    objects = catalogue[idxcatalog]

    try:
        ra_array = np.array(objects["RA"], dtype=float)
        dec_array = np.array(objects["DEC"], dtype=float)
    except:
        ra_array = np.array(objects["ID_ra"], dtype=float)
        dec_array = np.array(objects["ID_dec"], dtype=float)
    sky_coords = SkyCoord(ra_array, dec_array, unit="deg")
    d2d = source_coord.separation(sky_coords)
    angles = source_coord.position_angle(sky_coords)

    return objects, d2d, angles, source_coord, sky_coords


def determine_visible_catalogue_sources(
    ra: float, dec: float, size: float, catalogue: Table, verbose=False
) -> Table:
    """
    Find the sources in the catalogue that are visible in the cutout, and returns a smaller catalogue for that
    :param ra: Radio RA
    :param dec: Radio DEC
    :param wcs: WCS of Radio FITS files
    :param size: Size of cutout in degrees
    :param catalogue: Pan-AllWISE catalogue
    :param l_objects: LOFAR Value Added Catalogue objects
    :return: Subcatalog of catalogue that only contains sources near the radio source in the cutout size, as well as
    SkyCoord of their world coordinates
    """
    try:
        ra_array = np.array(catalogue["ra"], dtype=float)
        dec_array = np.array(catalogue["dec"], dtype=float)
    except:
        ra_array = np.array(catalogue["ID_ra"], dtype=float)
        dec_array = np.array(catalogue["ID_dec"], dtype=float)
    sky_coords = SkyCoord(ra_array, dec_array, unit="deg")

    source_coord = SkyCoord(ra, dec, unit="deg")
    search_radius = size * u.deg
    d2d = source_coord.separation(sky_coords)
    catalogmask = d2d < search_radius
    idxcatalog = np.where(catalogmask)[0]
    objects = catalogue[idxcatalog]

    return objects
