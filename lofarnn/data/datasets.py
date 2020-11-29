import multiprocessing
import os
from itertools import repeat
from typing import List, Union, Optional, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from scipy.ndimage.filters import gaussian_filter

from lofarnn.models.dataloaders.utils import get_lotss_objects
from lofarnn.utils.coco import create_coco_style_directory_structure
from lofarnn.utils.fits import extract_subimage, determine_visible_catalogue_sources
from lofarnn.visualization.cutouts import plot_three_channel_debug


def pad_with(
    vector: np.ndarray, pad_width: Union[int, List[int]], iaxis: int, **kwargs
):
    """
    Taken from Numpy documentation, will pad with zeros to make lofar image same size as other image
    :param vector:
    :param pad_width:
    :param iaxis:
    :param kwargs:
    :return:
    """
    pad_value = kwargs.get("padder", 0)
    vector[: pad_width[0]] = pad_value
    vector[-pad_width[1] :] = pad_value
    return vector


def make_layer(
    value: float, value_error: float, size: Tuple[int], non_uniform: bool = False
):
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


def make_catalogue_layer(
    column_name: str,
    wcs: WCS,
    shape: Union[Tuple[int], int],
    catalogue: Table,
    gaussian: bool = None,
    verbose: bool = False,
):
    """
    Create a layer based off the data in
    :param column_name: Name in catalogue of data to include
    :param shape: Shape of the image data
    :param wcs: WCS of the Radio data, so catalog data can be translated correctly
    :param catalogue: Catalogue to query
    :param gaussian: Whether to smooth the point values with a gaussian
    :return: A Numpy array that holds the information in the correct location
    """

    ra_array = np.array(catalogue["ra"], dtype=float)
    dec_array = np.array(catalogue["dec"], dtype=float)
    sky_coords = SkyCoord(ra_array, dec_array, unit="deg")

    # Now have the objects, need to convert those RA and Decs to pixel coordinates
    layer = np.zeros(shape=shape)
    coords = skycoord_to_pixel(sky_coords, wcs, 0)
    for index, x in enumerate(coords[0]):
        try:
            if (
                ~np.isnan(catalogue[index][column_name])
                and catalogue[index][column_name] > 0.0
            ):  # Make sure not putting in NaNs
                layer[int(np.floor(coords[0][index]))][
                    int(np.floor(coords[1][index]))
                ] = catalogue[index][column_name]
        except Exception as e:
            if verbose:
                print(f"Failed: {e}")
    if gaussian is not None:
        layer = gaussian_filter(layer, sigma=gaussian)
    return layer


def make_proposal_boxes(wcs: WCS, catalogue: Table, gaussian: Optional[bool] = None):
    """
   Create Faster RCNN proposal boxes for all sources in the image

   The sky_coords seems to be swapped x and y on the boxes, so should be swapped here too
   :param wcs: WCS of the Radio data, so catalog data can be translated correctly
   :param catalogue: Catalogue to query
   :param gaussian: Whether to smooth the point values with a gaussian
   :return: A Numpy array that holds the information in the correct location
   """

    ra_array = np.array(catalogue["ra"], dtype=float)
    dec_array = np.array(catalogue["dec"], dtype=float)
    sky_coords = SkyCoord(ra_array, dec_array, unit="deg")

    # Now have the objects, need to convert those RA and Decs to pixel coordinates
    proposals = []
    coords = skycoord_to_pixel(sky_coords, wcs, 0)
    for index, x in enumerate(coords[0]):
        try:
            proposals.append(
                make_bounding_box(
                    ra_array[index],
                    dec_array[index],
                    wcs=wcs,
                    class_name="Proposal Box",
                    gaussian=gaussian,
                )
            )
        except Exception as e:
            print(f"Failed Proposal: {e}")
    return proposals


def make_bounding_box(
    ra: Union[float, str],
    dec: Union[float, str],
    wcs: WCS,
    class_name: str = "Optical source",
    gaussian: Optional[bool] = None,
):
    """
    Creates a bounding box and returns it in (xmin, ymin, xmax, ymax, class_name) format
    :param class_name: Class name for the bounding box
    :param ra: RA of the object to make bounding box
    :param dec: Dec of object
    :param wcs: WCS to convert to pixel coordinates
    :param gaussian: Whether gaussian is being used, in which case the box is not int'd but left as a float, and the
    width of the gaussian is used for the width of the bounding box, if it is being used, instead of 'None', it should
    be the width of the Gaussian
    :return: Bounding box coordinates for COCO style annotation
    """
    source_skycoord = SkyCoord(ra, dec, unit="deg")
    box_center = skycoord_to_pixel(source_skycoord, wcs, 0)
    if gaussian is None:
        # Now create box, which will be accomplished by taking int to get xmin, ymin, and int + 1 for xmax, ymax
        xmin = int(np.floor(box_center[0])) - 0.5
        ymin = int(np.floor(box_center[1])) - 0.5
        ymax = ymin + 1
        xmax = xmin + 1
    else:
        xmin = int(np.floor(box_center[0])) - gaussian
        ymin = int(np.floor(box_center[1])) - gaussian
        xmax = np.ceil(int(np.floor(box_center[0])) + gaussian)
        ymax = np.ceil(int(np.floor(box_center[1])) + gaussian)

    return xmin, ymin, xmax, ymax, class_name, box_center


def create_cutouts(
    mosaic: Union[str, List[str], set],
    value_added_catalog: Union[Table, str],
    pan_wise_catalog: Union[Table, str],
    component_catalog: Union[Table, str],
    mosaic_location: str,
    save_cutout_directory: str,
    gaussian: Optional[bool] = None,
    all_channels: Optional[bool] = False,
    source_size: Optional[bool] = None,
    verbose: Optional[bool] = False,
):
    """
    Create cutouts of all sources in a field

    :param mosaic: Name of the field to use
    :param value_added_catalog: The VAC of the LoTSS data release
    :param pan_wise_catalog: The PanSTARRS-ALLWISE catalogue used for Williams, 2018, the LoTSS III paper
    :param mosaic_location: The location of the LoTSS DR2 mosaics
    :param save_cutout_directory: Where to save the cutout npy files
    :param all_channels: Whether to include all possible channels (grizy,W1,2,3,4 bands) in npy file or just (radio,i,W1)
    :param fixed_size: Whether to use fixed size cutouts, in arcseconds, or the LGZ size (default: LGZ)
    :param verbose: Whether to print extra information or not
    :return:
    """
    lofar_data_location = os.path.join(mosaic_location, mosaic, "mosaic-blanked.fits")
    lofar_rms_location = os.path.join(mosaic_location, mosaic, "mosaic.rms.fits")
    if gaussian is False:
        gaussian = None
    if type(pan_wise_catalog) == str:
        print("Trying To Open")
        pan_wise_catalog = fits.open(pan_wise_catalog, memmap=True)
        pan_wise_catalog = pan_wise_catalog[1].data
        print("Opened Catalog")
    # Load the data once, then do multiple cutouts
    try:
        fits.open(lofar_data_location, memmap=True)
        fits.open(lofar_rms_location, memmap=True)
    except:
        if verbose:
            print(f"Mosaic {mosaic} does not exist!")

    mosaic_cutouts = value_added_catalog[value_added_catalog["Mosaic_ID"] == mosaic]
    # Go through each cutout for that mosaic
    for l, source in enumerate(mosaic_cutouts):
        if not os.path.exists(
            os.path.join(save_cutout_directory, f"{source['Source_Name']}.npy")
        ):
            img_array = []
            # Get the ra and dec of the radio source
            source_ra = source["RA"]
            source_dec = source["DEC"]
            # Get the size of the cutout needed
            if source_size is None or source_size is False:
                source_size = (
                    source["LGZ_Size"] * 1.5
                ) / 3600.0  # in arcseconds converted to archours
                source_size *= np.sqrt(2)
            print(source_size)
            try:
                lhdu = extract_subimage(
                    lofar_data_location,
                    source_ra,
                    source_dec,
                    source_size,
                    verbose=verbose,
                )
            except:
                if verbose:
                    print(
                        f"Failed to make data cutout for source: {source['Source_Name']}"
                    )
                continue
            try:
                lrms = extract_subimage(
                    lofar_rms_location,
                    source_ra,
                    source_dec,
                    source_size,
                    verbose=verbose,
                )
            except:
                if verbose:
                    print(
                        f"Failed to make rms cutout for source: {source['Source_Name']}"
                    )
                continue
            img_array.append(lhdu[0].data / lrms[0].data)  # Makes the Radio/RMS channel
            header = lhdu[0].header
            wcs = WCS(header)
            # Now change source_size to size of cutout, or root(2)*source_size so all possible sources are included
            # exit()

            # Now time to get the data from the catalogue and add that in their own channels
            if verbose:
                print(f"Image Shape: {img_array[0].data.shape}")
            # Should now be in Radio/RMS, i, W1 format, else we skip it
            # Need from catalog ra, dec, iFApMag, w1Mag, also have a z_best, which might or might not be available for all
            if all_channels:
                layers = [
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
            else:
                layers = ["iFApMag", "w1Mag"]
            # Get the catalog sources once, to speed things up
            # cuts size in two to only get sources that fall within the cutout, instead of ones that go twice as large
            cutout_catalog = determine_visible_catalogue_sources(
                source_ra, source_dec, source_size / 2, pan_wise_catalog
            )
            # Now determine if there are other sources in the area
            other_visible_sources = determine_visible_catalogue_sources(
                source_ra, source_dec, source_size / 2, mosaic_cutouts
            )

            # Now make proposal boxes
            proposal_boxes = np.asarray(
                make_proposal_boxes(
                    wcs, img_array[0].shape, cutout_catalog, gaussian=gaussian
                )
            )
            for layer in layers:
                tmp = make_catalogue_layer(
                    layer, wcs, img_array[0].shape, cutout_catalog, gaussian=gaussian
                )
                img_array.append(tmp)

            img_array = np.array(img_array)
            if verbose:
                print(img_array.shape)
            img_array = np.moveaxis(img_array, 0, 2)
            # Include another array giving the bounding box for the source
            bounding_boxes = []
            try:
                source_bbox = make_bounding_box(
                    source["ID_ra"], source["ID_dec"], wcs, gaussian=gaussian
                )
                assert source_bbox[1] >= 0
                assert source_bbox[0] >= 0
                assert source_bbox[3] < img_array.shape[0]
                assert source_bbox[2] < img_array.shape[1]
                source_bounding_box = list(source_bbox)
                bounding_boxes.append(source_bounding_box)
            except:
                print("Source not in bounds")
                continue
            if verbose:
                plot_three_channel_debug(
                    img_array, bounding_boxes, 1, bounding_boxes[0][5]
                )
            # Now segmentation map
            source_components = component_catalog[
                component_catalog["Source_Name"] == source["Source_Name"]
            ]
            # Now go through and for any other sources in the field of view, add those
            for other_source in other_visible_sources:
                other_components = component_catalog[
                    component_catalog["Source_Name"] == other_source["Source_Name"]
                ]
                other_bbox = make_bounding_box(
                    other_source["ID_ra"],
                    other_source["ID_dec"],
                    wcs,
                    class_name="Other Optical Source",
                    gaussian=gaussian,
                )
                if ~np.isclose(other_bbox[0], bounding_boxes[0][0]) and ~np.isclose(
                    other_bbox[1], bounding_boxes[0][1]
                ):  # Make sure not same one
                    if (
                        other_bbox[1] >= 0
                        and other_bbox[0] >= 0
                        and other_bbox[3] < img_array.shape[0]
                        and other_bbox[2] < img_array.shape[1]
                    ):
                        bounding_boxes.append(
                            list(other_bbox)
                        )  # Only add the bounding box if it is within the image shape

            # Now save out the combined file
            bounding_boxes = np.array(bounding_boxes)

            # Save out the IDs of the cutout catalog and other catalog, for reverse indexing into optical catalogs from
            # results

            if verbose:
                print(bounding_boxes)
            combined_array = [
                img_array,
                bounding_boxes,
                proposal_boxes,
                wcs,
            ]
            try:
                np.save(
                    os.path.join(save_cutout_directory, source["Source_Name"]),
                    combined_array,
                )
            except Exception as e:
                if verbose:
                    print(f"Failed to save: {e}")
        else:
            print(f"Skipped: {l}")


def create_source_dataset(
    cutout_directory: str,
    pan_wise_location: str,
    value_added_catalog_location: str,
    component_catalog_location: str,
    dr_two_location: str,
    gaussian: Optional[bool] = None,
    all_channels: Optional[bool] = False,
    fixed_size: Optional[Union[int, float]] = None,
    filter_lgz: bool = True,
    verbose: bool = False,
    use_multiprocessing: bool = False,
    strict_filter: bool = False,
    filter_optical: bool = True,
    no_source: bool = False,
    num_threads: Optional[int] = os.cpu_count(),
):
    """

    :param cutout_directory: Directory to store the cutouts
    :param pan_wise_location: The location of the PanSTARRS-ALLWISE catalog
    :param value_added_catalog_location: Location of the LoTSS Value Added Catalog
    :param dr_two_location: The location of the LoTSS DR2 Mosaic Locations
    :param gaussian: Whether to spread out data layers with Gaussian of specified width
    :param use_multiprocessing: Whether to use multiprocessing
    :param num_threads: Number of threads to use, if multiprocessing is true
    :param strict_filter: Use the same filtering as for Jelle's subsample, with total flux > 10 mJy, and size > 15 arcseconds
    :param filter_optical: Whether to filter out sources with only optical sources or not
    :param filter_lgz: Whether to filter on LGZ_Size
    :return:
    """
    l_objects = get_lotss_objects(value_added_catalog_location, False)
    print(len(l_objects))
    if filter_lgz:
        l_objects = l_objects[~np.isnan(l_objects["LGZ_Size"])]
        print(len(l_objects))
    if filter_optical:
        if no_source:
            l_objects = l_objects[np.isnan(l_objects["ID_ra"])]
            l_objects = l_objects[np.isnan(l_objects["ID_dec"])]
        else:
            l_objects = l_objects[~np.isnan(l_objects["ID_ra"])]
            l_objects = l_objects[~np.isnan(l_objects["ID_dec"])]
        print(len(l_objects))
    else:  # Otherwise, remove those with optical IDs
        l_objects = l_objects[np.isnan(l_objects["ID_ra"])]
        l_objects = l_objects[np.isnan(l_objects["ID_dec"])]
        print(len(l_objects))
    if strict_filter:
        l_objects = l_objects[l_objects["LGZ_Size"] > 15.0]
        l_objects = l_objects[l_objects["Total_flux"] > 10.0]
        print(len(l_objects))
    mosaic_names = set(l_objects["Mosaic_ID"])

    comp_catalog = get_lotss_objects(component_catalog_location, False)

    # Go through each object, creating the cutout and saving to a directory
    # Create a directory structure identical for detectron2
    (
        all_directory,
        train_directory,
        val_directory,
        test_directory,
        annotations_directory,
    ) = create_coco_style_directory_structure(cutout_directory)

    # Now go through each source in l_objects and create a cutout of the fits file
    # Open the Panstarrs and WISE catalogue
    if fixed_size is False:
        fixed_size = None

    if use_multiprocessing:
        pool = multiprocessing.Pool(num_threads)
        pool.starmap(
            create_cutouts,
            zip(
                mosaic_names,
                repeat(l_objects),
                repeat(pan_wise_location),
                repeat(comp_catalog),
                repeat(dr_two_location),
                repeat(all_directory),
                repeat(gaussian),
                repeat(all_channels),
                repeat(fixed_size),
                repeat(verbose),
            ),
        )
    else:
        create_cutouts(
            mosaic_names,
            l_objects,
            pan_wise_location,
            comp_catalog,
            dr_two_location,
            all_directory,
            gaussian,
            all_channels,
            fixed_size,
            verbose,
        )
