import os

import _pickle as pickle
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

from ..utils.dataset import create_COCO_style_directory_structure
from ..utils.fits import extract_subimage


# sys.path.insert(0, "/home/s2153246/lofar_frcnn_tools")


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


def get_panstarrs_download_name(p_download_loc, f_ra, f_dec, p_size, filter):
    """
    Get the download name of a PanSTARRS fits file in a fixed format
    """
    return f'{p_download_loc}ra={f_ra}_dec={f_dec}_s={p_size}_{filter}.fits'


def create_dict_from_images_and_annotations_coco_version(image_names, source_list, extension,
                                                         image_dir='images', image_destination_dir=None,
                                                         json_dir='', json_name='json_data.pkl', imsize=None):
    """
    :param image_dir: image directory
    :param image_names: image names
    :param image_objects: image object containing (int xmin, int ymin, int xmax, int ymax, string class_name)
    :return:
    """

    assert (len(image_names) == len(source_list))
    # List to store single dict for each image
    dataset_dicts = []
    depth = 0

    # Iterate over all cutouts and their objects (which contain bounding boxes and class labels)
    for i, (image_name, cutout) in enumerate(zip(image_names, source_list)):

        # Get image dimensions and insert them in a python dict
        image_name = image_name + extension
        image_filename = os.path.join(image_dir, image_name)
        image_dest_filename = os.path.join(image_destination_dir, image_name)
        if not imsize is None:
            width, height = imsize, imsize
        elif extension == '.npy':
            im = np.load(image_filename, mmap_mode='r')  # mmap_mode might allow faster read
            width, height, depth = np.shape(im)
        else:
            raise ValueError('Image file format must either be .png, .jpg, .jpeg or .npy')
        size_value = {'width': width, 'height': height, 'depth': depth}

        record = {}

        record["file_name"] = image_dest_filename
        record["image_id"] = i
        record["height"] = height
        record["width"] = width

        # Insert bounding boxes and their corresponding classes
        # print('scale_factor:',cutout.scale_factor)
        objs = []
        cache_list = []
        for s in [cutout.c_source] + cutout.other_sources:
            xmin, ymin, xmax, ymax = s.xmin, s.ymin, s.xmax, s.ymax
            tup = (xmin, ymin, xmax, ymax)
            if tup in cache_list:
                continue
            cache_list.append(tup)
            assert xmax > xmin
            assert ymax > ymin
            assert isinstance(xmin, (int, float))
            assert isinstance(ymin, (int, float))
            assert isinstance(xmax, (int, float))
            assert isinstance(ymax, (int, float))

            ################# Flip x-axis
            old_ymax = ymax * cutout.scale_factor
            old_ymin = ymin * cutout.scale_factor
            ymin = height - old_ymax
            ymax = height - old_ymin
            xmin, xmax = cutout.scale_factor * xmin, cutout.scale_factor * xmax

            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": None,
                # "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    # Write all image dictionaries to file as one json
    json_path = os.path.join(json_dir, json_name)
    with open(json_path, "wb") as outfile:
        pickle.dump(dataset_dicts, outfile)
        # json.dump(dataset_dicts, outfile, indent=4)
    print(f'COCO annotation file created in \'{json_dir}\'.\n')


def pad_with(vector, pad_width, iaxis, kwargs):
    """
    Taken from Numpy documentation, will pad with zeros to make lofar image same size as other image
    :param vector:
    :param pad_width:
    :param iaxis:
    :param kwargs:
    :return:
    """
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def make_layer(value, value_error, size, non_uniform=False):
    """
    Creates a layer based on the value and the error, if non_uniform is True.

    Designed for adding catalogue data to image stacks

    :param value:
    :param value_error:
    :param size:
    :param non_uniform:
    :return:
    """

    if non_uniform:
        return np.random.normal(value, value_error, size=size)
    else:
        return np.full(shape=size, fill_value=value)


def determine_visible_catalogue_sources(ra, dec, wcs, size, catalogue, l_objects, verbose=False):
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
    RAarray = np.array(catalogue['ra'], dtype=float)
    DECarray = np.array(catalogue['dec'], dtype=float)
    sky_coords = SkyCoord(RAarray, DECarray, unit='deg')

    source_coord = SkyCoord(ra, dec, unit='deg')
    other_source = SkyCoord(l_objects['ID_ra'], l_objects['ID_dec'], unit="deg")
    search_radius = size * u.deg
    d2d = source_coord.separation(sky_coords)
    catalogmask = d2d < search_radius
    idxcatalog = np.where(catalogmask)[0]
    objects = catalogue[idxcatalog]

    if verbose:
        print(source_coord)
        print(other_source)
        print(skycoord_to_pixel(other_source, wcs))
        print(search_radius)
        print(len(objects))

    return objects


def make_catalogue_layer(column_name, wcs, shape, catalogue, verbose=False):
    """
    Create a layer based off the data in
    :param column_name: Name in catalogue of data to include
    :param shape: Shape of the image data
    :param wcs: WCS of the Radio data, so catalog data can be translated correctly
    :param catalogue: Catalogue to query
    :return: A Numpy array that holds the information in the correct location
    """

    RAarray = np.array(catalogue['ra'], dtype=float)
    DECarray = np.array(catalogue['dec'], dtype=float)
    sky_coords = SkyCoord(RAarray, DECarray, unit='deg')

    # Now have the objects, need to convert those RA and Decs to pixel coordinates
    layer = np.zeros(shape=shape)
    coords = skycoord_to_pixel(sky_coords, wcs, 0)
    for index, x in enumerate(coords[0]):
        try:
            if ~np.isnan(catalogue[index][column_name]):  # Make sure not putting in NaNs
                layer[int(x)][int(coords[1][index])] = catalogue[index][column_name]
        except Exception as e:
            if verbose:
                print(f"Failed: {e}")
    return layer


def main():
    """
    Build the COCO style dataset from the DR2 fits files, LGZ data, and the PanSTARRS-ALLWISE Catalogs
    :return:
    """
    DR_2_loc = "/run/media/jacob/34b36a2c-5b42-41cd-a1fa-7a09e5414860/lofar-surveys.org/downloads/DR2/mosaics/"
    # DR_2_loc = "/home/s2153246/data/data/LoTSS_DR2/lofar-surveys.org/downloads/DR2/mosaics/"
    fname = '/home/jacob/Development/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    # fname = '/home/s2153246/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits'
    plot_size = "norm"
    l_objects = get_lotss_objects(fname, True)
    l_objects = l_objects[~np.isnan(l_objects['LGZ_Size'])]
    l_objects = l_objects[~np.isnan(l_objects["ID_ra"])]
    mosaic_names = set(l_objects["Mosaic_ID"])

    # Go through each object, creating the cutout and saving to a directory
    cutout_directory = "/home/jacob/Development/lofar_frcnn_tools/cutouts/"
    # cutout_directory = "/home/s2153246/data/cutouts/"
    print(f'{"#" * 80} \nCreate and populate training directories for Detectron 2\n{"#" * 80}')
    # Create a directory structure identical for detectron2
    all_directory, train_directory, val_directory, test_directory, annotations_directory \
        = create_COCO_style_directory_structure(cutout_directory)

    # Now go through each source in l_objects and create a cutout of the fits file
    pan_wise_location = "/home/jacob/hetdex_ps1_allwise_photoz_v0.6.fits"
    # pan_wise_location = "/home/s2153246/data/catalogues/pan_allwise.fits"
    # Open the Panstarrs and WISE catalogue
    pan_wise_catalogue = fits.open(pan_wise_location, memmap=True)
    pan_wise_catalogue = pan_wise_catalogue[1].data
    # exit()

    # Arrays to store the bounding box and othe information
    bounding_box_dict = {}

    for mosaic in mosaic_names:
        # Load the data once, then do multiple cutouts
        lofar_data_location = os.path.join(DR_2_loc, mosaic, "mosaic-blanked.fits")
        lofar_rms_location = os.path.join(DR_2_loc, mosaic, "mosaic.rms.fits")
        try:
            fits.open(lofar_data_location, memmap=True)
            fits.open(lofar_rms_location, memmap=True)
        except:
            print(f"Mosaic {mosaic} does not exist!")
            continue

        mosaic_cutouts = l_objects[l_objects["Mosaic_ID"] == mosaic]
        # Go through each cutout for that mosaic
        for source in mosaic_cutouts:
            img_array = []
            # Get the ra and dec of the radio source
            source_ra = source["RA"]
            source_dec = source["DEC"]
            # Get the size of the cutout needed
            source_size = (source["LGZ_Size"] * 1.5) / 3600.  # in arcseconds converted to archours
            if plot_size == "min1":
                if source_size < (1.0 / 3600.):  # in arcseconds
                    source_size = (1.0 / 3600.)  # in arcseconds
            if plot_size == "only1":
                source_size = (1.0 / 3600.)  # in arcseconds
            try:
                lhdu = extract_subimage(lofar_data_location, source_ra, source_dec, source_size, verbose=False)
                lhdu.writeto(os.path.join(all_directory, source["Source_Name"] + '_radio_DR2.fits'), overwrite=True)
            except:
                print(f"Failed to make data cutout for source: {source['Source_Name']}")
                continue
            try:
                lrms = extract_subimage(lofar_rms_location, source_ra, source_dec, source_size, verbose=False)
                lrms.writeto(os.path.join(all_directory, source["Source_Name"] + '_rms_DR2.fits'), overwrite=True)
            except:
                print(f"Failed to make rms cutout for source: {source['Source_Name']}")
                continue
            img_array.append(lhdu[0].data / lrms[0].data)  # Makes the Radio/RMS channel
            header = lhdu[0].header
            wcs = WCS(header)

            # Now time to get the data from the catalogue and add that in their own channels

            print(f"Image Shape: {img_array[0].data.shape}")
            # Should now be in Radio/RMS, i, W1 format, else we skip it
            # Need from catalog ra, dec, iFApMag, w1Mag, also have a z_best, which might or might not be available for all
            layers = ["iFApMag", "w1Mag"]
            # Get the catalog sources once, to speed things up
            cutout_catalog = determine_visible_catalogue_sources(source_ra, source_dec, wcs, source_size,
                                                                 pan_wise_catalogue, source)
            for layer in layers:
                img_array.append(
                    make_catalogue_layer(layer, wcs, img_array[0].shape, cutout_catalog))

            img_array = np.array(img_array)
            print(img_array.shape)
            img_array = np.moveaxis(img_array, 0, 2)
            # Now save out the combined file
            np.save(os.path.join(all_directory, source['Source_Name']), img_array)

            # Now the sources that exist are saved in the combined folder, so go through all the sources and
            # take the ones that exist, create the bounding box, or segmentation map, of the optical source location


if __name__ == "__main__":
    main()
