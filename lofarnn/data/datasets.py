import multiprocessing
import os
import random
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from itertools import repeat

from lofarnn.utils.coco import create_coco_style_directory_structure
from lofarnn.utils.fits import extract_subimage


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
    try:
        ra_array = np.array(catalogue['ra'], dtype=float)
        dec_array = np.array(catalogue['dec'], dtype=float)
    except:
        ra_array = np.array(catalogue['ID_ra'], dtype=float)
        dec_array = np.array(catalogue['ID_dec'], dtype=float)
    sky_coords = SkyCoord(ra_array, dec_array, unit='deg')

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

    ra_array = np.array(catalogue['ra'], dtype=float)
    dec_array = np.array(catalogue['dec'], dtype=float)
    sky_coords = SkyCoord(ra_array, dec_array, unit='deg')

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


def make_bounding_box(ra, dec, wcs, class_name="Optical source"):
    """
    Creates a bounding box and returns it in (xmin, ymin, xmax, ymax, class_name) format
    :param class_name: Class name for the bounding box
    :param ra: RA of the object to make bounding box
    :param dec: Dec of object
    :param wcs: WCS to convert to pixel coordinates
    :return: Bounding box coordinates for COCO style annotation
    """

    coord = SkyCoord(ra, dec, unit='deg')
    box_center = skycoord_to_pixel(coord, wcs, 0)

    # Now create box, which will be accomplished by taking int to get xmin, ymin, and int + 1 for xmax, ymax
    xmin = int(box_center[0])
    ymin = int(box_center[1])
    ymax = ymin + 1
    xmax = xmin + 1

    return xmin, ymin, xmax, ymax, class_name


def create_cutouts(mosaic, value_added_catalog, pan_wise_catalog, mosaic_location,
                   save_cutout_directory, verbose=False):
    """
    Create cutouts of all sources in a field
    :param mosaic: Name of the field to use
    :param value_added_catalog: The VAC of the LoTSS data release
    :param pan_wise_catalog: The PanSTARRS-ALLWISE catalogue used for Williams, 2018, the LoTSS III paper
    :param mosaic_location: The location of the LoTSS DR2 mosaics
    :param save_cutout_directory: Where to save the cutout npy files
    :param verbose: Whether to print extra information or not
    :return:
    """
    # Load the data once, then do multiple cutouts
    lofar_data_location = os.path.join(mosaic_location, mosaic, "mosaic-blanked.fits")
    lofar_rms_location = os.path.join(mosaic_location, mosaic, "mosaic.rms.fits")
    try:
        fits.open(lofar_data_location, memmap=True)
        fits.open(lofar_rms_location, memmap=True)
    except:
        if verbose:
            print(f"Mosaic {mosaic} does not exist!")

    mosaic_cutouts = value_added_catalog[value_added_catalog["Mosaic_ID"] == mosaic]
    # Go through each cutout for that mosaic
    for source in mosaic_cutouts:
        img_array = []
        # Get the ra and dec of the radio source
        source_ra = source["RA"]
        source_dec = source["DEC"]
        # Get the size of the cutout needed
        source_size = (source["LGZ_Size"] * 1.5) / 3600.  # in arcseconds converted to archours
        try:
            lhdu = extract_subimage(lofar_data_location, source_ra, source_dec, source_size, verbose=False)
        except:
            if verbose:
                print(f"Failed to make data cutout for source: {source['Source_Name']}")
            continue
        try:
            lrms = extract_subimage(lofar_rms_location, source_ra, source_dec, source_size, verbose=False)
        except:
            if verbose:
                print(f"Failed to make rms cutout for source: {source['Source_Name']}")
            continue
        img_array.append(lhdu[0].data / lrms[0].data)  # Makes the Radio/RMS channel
        header = lhdu[0].header
        wcs = WCS(header)

        # Now time to get the data from the catalogue and add that in their own channels
        if verbose:
            print(f"Image Shape: {img_array[0].data.shape}")
        # Should now be in Radio/RMS, i, W1 format, else we skip it
        # Need from catalog ra, dec, iFApMag, w1Mag, also have a z_best, which might or might not be available for all
        layers = ["iFApMag", "w1Mag"]
        # Get the catalog sources once, to speed things up
        cutout_catalog = determine_visible_catalogue_sources(source_ra, source_dec, wcs, source_size,
                                                             pan_wise_catalog, source)
        # Now determine if there are other sources in the area
        other_visible_sources = determine_visible_catalogue_sources(source_ra, source_dec, wcs, source_size,
                                                                    mosaic_cutouts, source)
        for layer in layers:
            img_array.append(
                make_catalogue_layer(layer, wcs, img_array[0].shape, cutout_catalog))

        img_array = np.array(img_array)
        if verbose:
            print(img_array.shape)
        img_array = np.moveaxis(img_array, 0, 2)
        # Include another array giving the bounding box for the source
        bounding_boxes = []
        source_bounding_box = make_bounding_box(source['ID_ra'], source['ID_dec'], wcs)
        bounding_boxes.append(source_bounding_box)
        # Now go through and for any other sources in the field of view, add those
        for other_source in other_visible_sources:
            other_bbox = make_bounding_box(other_source['ID_ra'], other_source['ID_dec'],
                                           wcs, class_name="Other Optical Source")
            if other_bbox[0] != bounding_boxes[0][0] and other_bbox[1] != bounding_boxes[0][
                1]:  # Make sure not same one
                if other_bbox[0] >= 0 and other_bbox[1] >= 0 and other_bbox[2] < img_array.shape[0] and other_bbox[3] < \
                        img_array.shape[1]:
                    bounding_boxes.append(other_bbox)  # Only add the bounding box if it is within the image shape
        # Now save out the combined file
        bounding_boxes = np.array(bounding_boxes)
        if verbose:
            print(bounding_boxes)
        combined_array = [img_array, bounding_boxes]
        try:
            np.save(os.path.join(save_cutout_directory, source['Source_Name']), combined_array)
        except Exception as e:
            if verbose:
                print(f"Failed to save: {e}")


def create_fixed_cutouts(mosaic, value_added_catalog, pan_wise_catalog, mosaic_location,
                         save_cutout_directory, size_arcseconds=200, verbose=False):
    """
    Create cutouts of all sources in a field, with the same size cutout. This means that each image
    will have at least one source, but potentially more, if the cutouts are large enough.

    Each arcsecond is 2/3 of a pixel i.e. 300 arcseconds = 200 pixels
    :param mosaic: Name of the field to use
    :param value_added_catalog: The VAC of the LoTSS data release
    :param pan_wise_catalog: The PanSTARRS-ALLWISE catalogue used for Williams, 2018, the LoTSS III paper
    :param mosaic_location: The location of the LoTSS DR2 mosaics
    :param save_cutout_directory: Where to save the cutout npy files
    :param size_arcseconds: The size, in arcseconds, of the cutouts (determines pixel size from that)
    :param verbose: Whether to print extra information or not
    :return:
    """
    # Load the data once, then do multiple cutouts
    lofar_data_location = os.path.join(mosaic_location, mosaic, "mosaic-blanked.fits")
    lofar_rms_location = os.path.join(mosaic_location, mosaic, "mosaic.rms.fits")
    try:
        fits.open(lofar_data_location, memmap=True)
        fits.open(lofar_rms_location, memmap=True)
    except:
        if verbose:
            print(f"Mosaic {mosaic} does not exist!")

    mosaic_cutouts = value_added_catalog[value_added_catalog["Mosaic_ID"] == mosaic]
    # Go through each cutout for that mosaic
    for source in mosaic_cutouts:
        img_array = []
        # Get the ra and dec of the radio source
        source_ra = source["RA"]
        source_dec = source["DEC"]
        # Get the size of the cutout needed
        source_size = size_arcseconds / 3600.  # in arcseconds converted to archours
        try:
            lhdu = extract_subimage(lofar_data_location, source_ra, source_dec, source_size, verbose=False)
        except:
            if verbose:
                print(f"Failed to make data cutout for source: {source['Source_Name']}")
            continue
        try:
            lrms = extract_subimage(lofar_rms_location, source_ra, source_dec, source_size, verbose=False)
        except:
            if verbose:
                print(f"Failed to make rms cutout for source: {source['Source_Name']}")
            continue
        img_array.append(lhdu[0].data / lrms[0].data)  # Makes the Radio/RMS channel
        header = lhdu[0].header
        wcs = WCS(header)

        # Now time to get the data from the catalogue and add that in their own channels
        if verbose:
            print(f"Image Shape: {img_array[0].data.shape}")
        # Should now be in Radio/RMS, i, W1 format, else we skip it
        # Need from catalog ra, dec, iFApMag, w1Mag, also have a z_best, which might or might not be available for all
        layers = ["iFApMag", "w1Mag"]
        # Get the catalog sources once, to speed things up
        cutout_catalog = determine_visible_catalogue_sources(source_ra, source_dec, wcs, source_size,
                                                             pan_wise_catalog, source)

        # Now determine if there are other sources in the area
        other_visible_sources = determine_visible_catalogue_sources(source_ra, source_dec, wcs, source_size,
                                                                    mosaic_cutouts, source)
        for layer in layers:
            img_array.append(
                make_catalogue_layer(layer, wcs, img_array[0].shape, cutout_catalog))

        img_array = np.array(img_array)
        if verbose:
            print(img_array.shape)
        img_array = np.moveaxis(img_array, 0, 2)
        # Include another array giving the bounding box for the source
        bounding_boxes = []
        source_bounding_box = make_bounding_box(source['ID_ra'], source['ID_dec'], wcs)
        bounding_boxes.append(source_bounding_box)
        # Now go through and for any other sources in the field of view, add those
        for other_source in other_visible_sources:
            other_bbox = make_bounding_box(other_source['ID_ra'], other_source['ID_dec'],
                                           wcs, class_name="Other Optical Source")
            if other_bbox[0] != bounding_boxes[0][0] and other_bbox[1] != bounding_boxes[0][
                1]:  # Make sure not same one
                if other_bbox[0] >= 0 and other_bbox[1] >= 0 and other_bbox[2] < img_array.shape[0] and other_bbox[3] < \
                        img_array.shape[1]:
                    bounding_boxes.append(other_bbox)  # Only add the bounding box if it is within the image shape
        # Now save out the combined file
        bounding_boxes = np.array(bounding_boxes)
        if verbose:
            print(bounding_boxes)
        combined_array = [img_array, bounding_boxes]
        try:
            np.save(os.path.join(save_cutout_directory, source['Source_Name']), combined_array)
        except Exception as e:
            if verbose:
                print(f"Failed to save: {e}")


def create_fixed_source_dataset(cutout_directory, pan_wise_location,
                                value_added_catalog_location, dr_two_location, fixed_size=300,
                                use_multiprocessing=False,
                                num_threads=os.cpu_count()):
    """
    Creates fixed size dataset using LGZ data

    :param cutout_directory: Directory to store the cutouts
    :param pan_wise_location: The location of the PanSTARRS-ALLWISE catalog
    :param value_added_catalog_location: Location of the LoTSS Value Added Catalog
    :param dr_two_location: The location of the LoTSS DR2 Mosaic Locations
    :param fixed_size: The size of the cutouts, in arcseconds default to 300 arcseconds
    :param use_multiprocessing: Whether to use multiprocessing
    :param num_threads: Number of threads to use, if multiprocessing is true
    :return:
    """

    l_objects = get_lotss_objects(value_added_catalog_location, True)
    l_objects = l_objects[~np.isnan(l_objects['LGZ_Size'])]
    l_objects = l_objects[~np.isnan(l_objects["ID_ra"])]
    mosaic_names = set(l_objects["Mosaic_ID"])

    # Go through each object, creating the cutout and saving to a directory
    # Create a directory structure identical for detectron2
    all_directory, train_directory, val_directory, test_directory, annotations_directory \
        = create_coco_style_directory_structure(cutout_directory)

    # Now go through each source in l_objects and create a cutout of the fits file
    # Open the Panstarrs and WISE catalogue
    pan_wise_catalogue = fits.open(pan_wise_location, memmap=True)
    pan_wise_catalogue = pan_wise_catalogue[1].data

    if use_multiprocessing:
        pool = multiprocessing.Pool(num_threads)
        pool.starmap(create_fixed_cutouts, zip(mosaic_names, repeat(l_objects), repeat(pan_wise_catalogue),
                                               repeat(dr_two_location), repeat(all_directory),
                                               repeat(fixed_size), repeat(True)))
    else:
        for mosaic in mosaic_names:
            create_fixed_cutouts(mosaic=mosaic, value_added_catalog=l_objects, pan_wise_catalog=pan_wise_catalogue,
                                 mosaic_location=dr_two_location,
                                 save_cutout_directory=all_directory,
                                 size_arcseconds=fixed_size,
                                 verbose=True)


def create_variable_source_dataset(cutout_directory, pan_wise_location,
                                   value_added_catalog_location, dr_two_location, use_multiprocessing=False,
                                   num_threads=os.cpu_count()):
    """
    Create variable sized cutouts (hardcoded to 1.5 times the LGZ_Size) for each of the cutouts

    :param cutout_directory: Directory to store the cutouts
    :param pan_wise_location: The location of the PanSTARRS-ALLWISE catalog
    :param value_added_catalog_location: Location of the LoTSS Value Added Catalog
    :param dr_two_location: The location of the LoTSS DR2 Mosaic Locations
    :param use_multiprocessing: Whether to use multiprocessing
    :param num_threads: Number of threads to use, if multiprocessing is true
    :return:
    """

    l_objects = get_lotss_objects(value_added_catalog_location, True)
    l_objects = l_objects[~np.isnan(l_objects['LGZ_Size'])]
    l_objects = l_objects[~np.isnan(l_objects["ID_ra"])]
    mosaic_names = set(l_objects["Mosaic_ID"])

    # Go through each object, creating the cutout and saving to a directory
    # Create a directory structure identical for detectron2
    all_directory, train_directory, val_directory, test_directory, annotations_directory \
        = create_coco_style_directory_structure(cutout_directory)

    # Now go through each source in l_objects and create a cutout of the fits file
    # Open the Panstarrs and WISE catalogue
    pan_wise_catalogue = fits.open(pan_wise_location, memmap=True)
    pan_wise_catalogue = pan_wise_catalogue[1].data

    if use_multiprocessing:
        pool = multiprocessing.Pool(num_threads)
        pool.starmap(create_cutouts, zip(mosaic_names, repeat(l_objects), repeat(pan_wise_catalogue),
                                         repeat(dr_two_location), repeat(all_directory),
                                         repeat(True)))
    else:
        for mosaic in mosaic_names:
            create_cutouts(mosaic=mosaic, value_added_catalog=l_objects, pan_wise_catalog=pan_wise_catalogue,
                           mosaic_location=dr_two_location,
                           save_cutout_directory=all_directory,
                           verbose=True)


def split_data(image_directory, split=(0.6, 0.8)):
    """
    Split up the data and return which images should go to which train, test, val directory
    :param image_directory: The directory where all the images are located, i.e. the "all" directory
    :param train_directory: Where to store the training data
    :param val_directory: Where to store the validation data
    :param test_directory: Where to store the test data
    :param split: Tuple of train, val, test split,
    in form of (train fraction, val fraction), with the rest being put in test directory
    :return: A dict containing which images go to which directory
    """

    image_paths = Path(image_directory).rglob("*.npy")
    image_paths = random.shuffle(list(image_paths))
    train_images = image_paths[:int(len(image_paths)*split[0])]
    val_images = image_paths[int(len(image_paths)*split[0]):int(len(image_paths)*split[1])]
    test_images = image_paths[int(len(image_paths)*split[1]):]

    return {"train": train_images,
            "val": val_images,
            "test": test_images}