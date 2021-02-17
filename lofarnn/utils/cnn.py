import os
import pickle
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import List, Any, Optional, Union, Tuple

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D

from lofarnn.data.cutouts import augment_image_and_bboxes, convert_to_valid_color
from lofarnn.models.dataloaders.utils import get_lotss_objects
from lofarnn.utils.common import create_coco_style_directory_structure, split_data
from lofarnn.utils.fits import determine_visible_catalogue_source_and_separation


def make_single_cnn_set(
    image_names: List[Path],
    record_list: List[Any],
    set_number: int,
    image_destination_dir: Optional[str],
    pan_wise_location: str = "",
    bands: List[str] = (
        "iFApMag",
        "w1Mag",
        "gFApMag",
        "rFApMag",
        "zFApMag",
        "yFApMag",
        "w2Mag",
        "w3Mag",
        "w4Mag",
    ),
    resize: Optional[Union[int, List[int]]] = None,
    rotation: Optional[Union[List[float], float]] = None,
    convert: bool = False,
    vac_catalog_location: str = "",
    normalize: bool = True,
    **kwargs,
):
    pan_wise_catalog = fits.open(pan_wise_location, memmap=True)
    pan_wise_catalog = pan_wise_catalog[1].data
    vac_catalog = get_lotss_objects(vac_catalog_location)
    for i, image_name in enumerate(image_names):
        # Get image dimensions and insert them in a python dict
        record_dest_filename = os.path.join(
            image_destination_dir, image_name.stem + f".record.{normalize}.npy"
        )
        if convert:
            image_dest_filename = os.path.join(
                image_destination_dir, image_name.stem + f".cnn.{set_number}.png"
            )
        else:
            if rotation is not None and rotation.any() > 0:
                image_dest_filename = os.path.join(
                    image_destination_dir,
                    image_name.stem + f".cnn.{set_number}.{normalize}.npy",
                )
                wcs_dest_filename = os.path.join(
                    image_destination_dir,
                    image_name.stem + f".cnn.{set_number}.{normalize}.wcs.npy",
                )
                record_dest_filename = os.path.join(
                    image_destination_dir,
                    image_name.stem + f".record.{set_number}.{normalize}.npy",
                )
            else:
                image_dest_filename = os.path.join(
                    image_destination_dir, image_name.stem + f".cnn.{normalize}.npy"
                )
                wcs_dest_filename = os.path.join(
                    image_destination_dir, image_name.stem + f".cnn.{normalize}.wcs.npy"
                )
        if not os.path.exists(os.path.join(image_dest_filename)):
            (image, cutouts, proposal_boxes, wcs) = np.load(
                image_name, allow_pickle=True
            )  # mmap_mode might allow faster read
            print(image.shape)
            image = np.moveaxis(image, 0, 2)
            cutout = Cutout2D(
                image[:, :, 0],
                position=(int(image.shape[0] / 2), int(image.shape[1] / 2)),
                size=(int(image.shape[0]), int(image.shape[1])),
                wcs=wcs,
            )
            wcs = cutout.wcs
            image = np.nan_to_num(image)
            # Need this to convert the bbox coordinates into the correct format
            (image, cutouts, proposal_boxes,) = augment_image_and_bboxes(
                image,
                cutouts=cutouts,
                proposal_boxes=proposal_boxes,
                angle=0,
                new_size=resize,
                verbose=False,
            )
            width, height, depth = np.shape(image)

            # First R (Radio) channel
            image = image[:, :, 0]
            image_clip = np.copy(image)
            image_none = np.copy(image)
            image = convert_to_valid_color(
                image,
                clip=True,
                lower_clip=0.0,
                upper_clip=1000,
                normalize=normalize,
                scaling="sqrt",
            )
            image_clip = convert_to_valid_color(
                image_clip,
                clip=True,
                lower_clip=0.0,
                upper_clip=1000,
                normalize=normalize,
                scaling=None,
            )
            image_none = convert_to_valid_color(
                image_none,
                clip=False,
                normalize=False,
                scaling="sqrt",
            )
            image = np.ma.filled(
                image, fill_value=0.0
            )  # convert back from masked array to normal array
            image_clip = np.ma.filled(
                image_clip, fill_value=0.0
            )  # convert back from masked array to normal array
            image_none = np.ma.filled(
                image_none, fill_value=0.0
            )  # convert back from masked array to normal array
            # Now restack into 3 channel image
            image = np.dstack((image, image_clip, image_none))
            image = np.nan_to_num(image)  # Only take radio
            np.save(image_dest_filename, image)  # Save to the final destination
            np.save(wcs_dest_filename, wcs)
        else:
            image = np.load(image_dest_filename)
            wcs = np.load(wcs_dest_filename, allow_pickle=True)
            height, width, depth = np.shape(image)

        record = {
            "file_name": image_dest_filename,
            "image_id": i,
            "height": height,
            "width": width,
            "depth": depth,
        }
        if not os.path.exists(os.path.join(record_dest_filename)):
            source = vac_catalog[vac_catalog["Source_Name"] == image_name.stem]
            # All optical sources in 150 arcsecond radius of the point
            (
                objects,
                distances,
                angles,
                source_coords,
                sky_coords,
            ) = determine_visible_catalogue_source_and_separation(
                source["RA"],
                source["DEC"],
                source[kwargs.get("size_name", "LGZ_Size")] * 1.5 / 3600.0,
                pan_wise_catalog,
            )
            # Sort from closest to farthest distance
            idx = np.argsort(distances)
            objects = objects[idx]
            distances = distances[idx]
            angles = angles[idx]
            sky_coords = sky_coords[idx]
            optical_sources = []
            optical_labels = []
            for j, obj in enumerate(objects):
                optical_sources.append([])
                if (
                    obj["objID"] == source["objID"].data
                    and obj["AllWISE"] == source["AllWISE"].data
                ):
                    optical_labels.append(1)  # Optical Source
                else:
                    optical_labels.append(0)
                optical_sources[-1].append(obj["objID"])
                optical_sources[-1].append(obj["AllWISE"])
                optical_sources[-1].append(obj["ra"])
                optical_sources[-1].append(obj["dec"])
                optical_sources[-1].append(distances[j])
                optical_sources[-1].append(angles[j])
                optical_sources[-1].append(obj["z_best"])
                for layer in bands:
                    value = np.nan_to_num(obj[layer])
                    if normalize:  # Scale to between 0 and 1 for 10 to 28 magnitude
                        value = np.clip(value, 10.0, 28.0)
                        value = (value - 10.0) / (28.0 - 10.0)
                    optical_sources[-1].append(value)
            record["optical_sources"] = optical_sources
            record["optical_labels"] = optical_labels
            record["source_skycoord"] = source_coords
            record["optical_skycoords"] = sky_coords
            record["wcs"] = wcs
            if rotation is not None:
                record["rotation"] = rotation[set_number]
            else:
                record["rotation"] = 0.0
            np.save(record_dest_filename, record)
        else:
            record = np.load(record_dest_filename, fix_imports=True, allow_pickle=True)

        # Now add the labels, so need to know which optical source is the true one
        record_list.append(record)


def create_cnn_annotations(
    image_names,
    image_destination_dir=None,
    json_dir="",
    json_name="json_data.pkl",
    pan_wise_location="",
    resize=None,
    rotation=None,
    convert=False,
    bands: List[str] = (
        "iFApMag",
        "w1Mag",
        "gFApMag",
        "rFApMag",
        "zFApMag",
        "yFApMag",
        "w2Mag",
        "w3Mag",
        "w4Mag",
    ),
    vac_catalog_location="",
    normalize=True,
    rotation_names=None,
    verbose=False,
):
    """
    Creates the annotations for the COCO-style dataset from the npy files available, and saves the images in the correct
    directory
    :param image_names: Image names, i.e., the source names
    :param image_destination_dir: The directory the images will end up in
    :param json_dir: The directory where to put the JSON annotation file
    :param json_name: The name of the JSON file
    :param bands: The bands to include in the source
    :param rotation: Whether to rotate the images or not, if given as a tuple, it is taken as rotate each image by that amount,
    if a single float, then rotates images randomly between -rotation,rotation 50 times
    :param convert: Whether to convert to PNG files (default), or leave them as NPY files
    :return:
    """

    if rotation is not None:
        if type(rotation) == tuple:
            num_copies = len(rotation)
        else:
            num_copies = 3
    else:
        num_copies = 1
    if rotation_names is not None:
        # Rotate these specific sources ~ 2.5 times more (generally multicomponent ones)
        extra_rotates = []
        t = []
        for i, name in enumerate(image_names):
            if name.stem in rotation_names:
                extra_rotates.append(i)
                t.append(rotation_names)
        extra_rotates = np.asarray(extra_rotates)
        image_names = np.asarray(image_names)
        extra_names = image_names[extra_rotates]
        mask = np.ones(len(image_names), np.bool)
        mask[extra_rotates] = 0
        single_names = image_names[mask]
    else:
        single_names = image_names
        extra_names = []
    # List to store single dict for each image
    dataset_dicts = []
    if num_copies > 1:
        manager = Manager()
        pool = Pool(processes=os.cpu_count())
        L = manager.list()
        rotation = np.linspace(0, 170, num_copies)
        [
            pool.apply_async(
                make_single_cnn_set,
                args=[
                    single_names,
                    L,
                    m,
                    image_destination_dir,
                    pan_wise_location,
                    bands,
                    resize,
                    rotation,
                    convert,
                    vac_catalog_location,
                    normalize,
                ],
            )
            for m in range(num_copies)
        ]
        print(len(L))
        # Now do the same for the extra copies, but with more rotations, ~2.5 to equal out multi and single comp sources
        num_multi_copies = int(np.ceil(num_copies * 2.5))
        print(f"Num Multi Copies: {num_multi_copies}")
        multi_rotation = np.linspace(0, 170, num_multi_copies)
        [
            pool.apply_async(
                make_single_cnn_set,
                args=[
                    extra_names,
                    L,
                    m,
                    image_destination_dir,
                    pan_wise_location,
                    bands,
                    resize,
                    multi_rotation,
                    convert,
                    bands,
                    vac_catalog_location,
                    normalize,
                ],
            )
            for m in range(num_multi_copies)
        ]
        pool.close()
        pool.join()
        print(len(L))
        print(f"Length of L: {len(L)}")
        for element in L:
            dataset_dicts.append(element)
        print(f"Length of Dataset Dict: {len(dataset_dicts)}")
        # Write all image dictionaries to file as one json
        json_path = os.path.join(json_dir, json_name)
        with open(json_path, "wb") as outfile:
            pickle.dump(dataset_dicts, outfile)
        if verbose:
            print(f"CNN annotation file created in '{json_dir}'.\n")
        return 0  # Returns to doesnt go through it again

    # Iterate over all cutouts and their objects (which contain bounding boxes and class labels)
    manager = Manager()
    pool = Pool(processes=os.cpu_count())
    L = manager.list()
    for name in image_names:
        # x = pool.apply_async(
        make_single_cnn_set(
            [name],
            L,
            0,
            image_destination_dir,
            pan_wise_location,
            bands,
            resize,
            None,
            convert,
            vac_catalog_location,
            normalize,
        )
        # )
        # x.get()
    pool.close()
    pool.join()
    print(len(L))
    print(f"Length of L: {len(L)}")
    for element in L:
        dataset_dicts.append(element)
    # Write all image dictionaries to file as one json
    json_path = os.path.join(json_dir, json_name)
    with open(json_path, "wb") as outfile:
        pickle.dump(dataset_dicts, outfile)
    if verbose:
        print(f"CNN annotation file created in '{json_dir}'.\n")


def create_cnn_dataset(
    root_directory: str,
    counterpart_catalog: str = "",
    split_fraction: float = 0.2,
    resize: Optional[Union[Tuple[int], int]] = None,
    rotation: Optional[Union[List[float], float]] = None,
    convert: bool = True,
    bands: List[str] = (
        "iFApMag",
        "w1Mag",
        "gFApMag",
        "rFApMag",
        "zFApMag",
        "yFApMag",
        "w2Mag",
        "w3Mag",
        "w4Mag",
    ),
    vac_catalog: str = "",
    normalize: bool = True,
    subset: str = "",
    multi_rotate_only: Optional[Union[List[str], str]] = None,
    verbose: bool = False,
    **kwargs,
):
    """
    Create COCO directory structure, if it doesn't already exist, split the image data, and save it to the correct
    directories, and create the COCO annotation file to be loaded into Detectron2, or other similar models
    :param split_fraction: Fraction of the data for the test set. the validation set is rolled into the test set.
    :param root_directory: root directory for the COCO dataset
    :param bands: The bands to include in the source
    :param resize: Image size to resize to, or None if not resizing
    :param convert: Whether to convert npy files to png, or to keep them in the original format, useful for SourceMapper
    :param verbose: Whether to print more data to stdout or not
    :param subset: Whether to limit ones to only the fluxlimit sources, if not empty, should be path to list of source filepaths to use
    :return:
    """

    (
        all_directory,
        train_directory,
        val_directory,
        test_directory,
        annotations_directory,
    ) = create_coco_style_directory_structure(root_directory, verbose=verbose)

    # Gather data from all_directory
    data_split = split_data(
        all_directory, val_split=split_fraction, test_split=split_fraction
    )
    if subset:
        # Keep only those already in the subset
        subset = np.load(subset, allow_pickle=True)
        for d in ["train", "test", "val"]:
            data_split[d] = data_split[d][np.isin(data_split[d], subset)]
        annotations_directory = os.path.join(annotations_directory, "subset")
    if multi_rotate_only:
        l_objects = get_lotss_objects(multi_rotate_only, False)
        # Get all multicomponent sources
        l_objects = l_objects[
            l_objects[kwargs.get("association_name", "LGZ_Assoc")] > 1
        ]
        multi_names = l_objects["Source_Name"].data
    else:
        multi_names = None
    if len(data_split["val"]) > 0:
        create_cnn_annotations(
            data_split["val"],
            json_dir=annotations_directory,
            image_destination_dir=val_directory,
            json_name=f"cnn_val_norm{normalize}_extra.pkl",
            pan_wise_location=counterpart_catalog,
            resize=resize,
            rotation=None,
            convert=convert,
            normalize=normalize,
            bands=bands,
            vac_catalog_location=vac_catalog,
            rotation_names=multi_names,
            verbose=verbose,
        )
    create_cnn_annotations(
        data_split["train"],
        json_dir=annotations_directory,
        image_destination_dir=train_directory,
        json_name=f"cnn_train_test_norm{normalize}_extra.pkl",
        pan_wise_location=counterpart_catalog,
        resize=resize,
        rotation=None,
        convert=convert,
        normalize=normalize,
        bands=bands,
        vac_catalog_location=vac_catalog,
        rotation_names=multi_names,
        verbose=verbose,
    )
    create_cnn_annotations(
        data_split["test"],
        json_dir=annotations_directory,
        image_destination_dir=test_directory,
        json_name=f"cnn_test_norm{normalize}_extra.pkl",
        pan_wise_location=counterpart_catalog,
        resize=resize,
        rotation=None,
        convert=convert,
        normalize=normalize,
        bands=bands,
        vac_catalog_location=vac_catalog,
        rotation_names=multi_names,
        verbose=verbose,
    )
    create_cnn_annotations(
        data_split["train"],
        json_dir=annotations_directory,
        image_destination_dir=train_directory,
        json_name=f"cnn_train_norm{normalize}_extra.pkl",
        pan_wise_location=counterpart_catalog,
        resize=resize,
        rotation=rotation,
        convert=convert,
        normalize=normalize,
        bands=bands,
        vac_catalog_location=vac_catalog,
        rotation_names=multi_names,
        verbose=verbose,
    )
