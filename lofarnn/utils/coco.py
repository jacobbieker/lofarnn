import os
import pickle
from multiprocessing import Pool, Manager
from pathlib import Path
from typing import Optional, List, Tuple, Union, Any

import numpy as np
from PIL import Image
from detectron2.structures import BoxMode

from lofarnn.data.cutouts import convert_to_valid_color, augment_image_and_bboxes
from lofarnn.models.dataloaders.utils import get_lotss_objects
from lofarnn.utils.common import create_coco_style_directory_structure, split_data


def make_single_coco_annotation_set(
    image_names: List[Path],
    record_list: List[Any],
    set_number: int,
    image_destination_dir: Optional[str],
    multiple_bboxes: bool = True,
    resize: Optional[Union[Tuple[int], int]] = None,
    rotation: Optional[Union[List[float], float]] = None,
    convert: bool = True,
    all_channels: bool = False,
    precomputed_proposals: bool = False,
    normalize: bool = True,
):
    """
    For use with multiprocessing, goes through and does one rotation for the COCO annotations
    :param image_names: Image names to load and use for generating dataset
    :param record_list: Array to add the sources to
    :param set_number:
    :param image_destination_dir: The destination directory of the images
    :param resize: What to resize to
    :param rotation: How much to rotate
    :param convert: Whether to convert to PNG and normalize to between 0 and 255 for all channels
    :param all_channels: Whether to use all 10 channels, or just radio, iband, W1 band
    :param precomputed_proposals: Whether to create precomputed proposals
    :param normalize: Whether to normalize input data between 0 and 1
    :return:
    """
    for i, image_name in enumerate(image_names):
        # Get image dimensions and insert them in a python dict
        if convert:
            image_dest_filename = os.path.join(
                image_destination_dir, image_name.stem + f".{set_number}.png"
            )
        else:
            if rotation is not None and rotation.any() > 0:
                image_dest_filename = os.path.join(
                    image_destination_dir, image_name.stem + f".{set_number}.npy"
                )
            else:
                image_dest_filename = os.path.join(
                    image_destination_dir, image_name.stem + f".npy"
                )
        (image, cutouts, proposal_boxes, wcs) = np.load(
            image_name, allow_pickle=True
        )  # mmap_mode might allow faster read
        image = np.nan_to_num(image)
        # Change order to H,W,C for imgaug
        if rotation is not None:
            if isinstance(rotation, (list, tuple, np.ndarray)):
                (image, cutouts, proposal_boxes,) = augment_image_and_bboxes(
                    image,
                    cutouts=cutouts,
                    proposal_boxes=proposal_boxes,
                    angle=rotation[set_number],
                    new_size=resize,
                )
            else:
                (image, cutouts, proposal_boxes,) = augment_image_and_bboxes(
                    image,
                    cutouts=cutouts,
                    proposal_boxes=proposal_boxes,
                    angle=np.random.uniform(-rotation, rotation),
                    new_size=resize,
                )
        else:
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
        if all_channels and depth != 10:
            continue

        # First R (Radio) channel
        image[:, :, 0] = convert_to_valid_color(
            image[:, :, 0],
            clip=True,
            lower_clip=0.0,
            upper_clip=1000,
            normalize=normalize,
            scaling=None,
        )
        for layer in range(1, image.shape[2]):
            image[:, :, layer] = convert_to_valid_color(
                image[:, :, layer],
                clip=True,
                lower_clip=10.0,
                upper_clip=28.0,
                normalize=normalize,
                scaling=None,
            )
        if not os.path.exists(os.path.join(image_dest_filename)):
            if convert:
                image = np.nan_to_num(image)
                image = (255.0 * image).astype(np.uint8)
                # If converting, only take the first three layers, generally Radio, i band, W1 band
                pil_im = Image.fromarray(image[:, :, :3], "RGB")
                pil_im.save(image_dest_filename)
            else:
                image = np.nan_to_num(image)
                np.save(image_dest_filename, image)  # Save to the final destination
        if all_channels:
            rec_depth = 10
        else:
            rec_depth = 3
        record = {
            "file_name": image_dest_filename,
            "image_id": i,
            "height": height,
            "width": width,
            "depth": rec_depth,
        }

        # Insert bounding boxes and their corresponding classes
        objs = []
        # check if there is no optical source
        try:
            if len(cutouts) > 0:  # There is an optical source
                if not multiple_bboxes:
                    cutouts = [
                        cutouts[0]
                    ]  # Only take the first one, the main optical source
                for source_num, bbox in enumerate(cutouts):
                    assert float(bbox[2]) >= float(bbox[0])
                    assert float(bbox[3]) >= float(bbox[1])

                    if bbox[4] == "Other Optical Source":
                        category_id = 0
                    else:
                        category_id = 0
                    obj = {
                        "bbox": [
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": category_id,  # For Optical Source
                        "iscrowd": 0,
                    }
                    objs.append(obj)
        except Exception as e:
            print(e)
            print("No Optical source found")
        if precomputed_proposals:
            record["proposal_boxes"] = proposal_boxes
            record["proposal_objectness_logits"] = np.ones(len(proposal_boxes))
            record["proposal_bbox_mode"] = BoxMode.XYXY_ABS
        record["annotations"] = objs
        record_list.append(record)


def create_coco_annotations(
    image_names: Union[List[Path], List[str]],
    image_destination_dir: Optional[str],
    json_dir: str = "",
    json_name: str = "json_data.pkl",
    multiple_bboxes: bool = True,
    resize: Optional[Union[Tuple[int], int]] = None,
    rotation: Optional[Union[List[float], float]] = None,
    convert: bool = True,
    all_channels: bool = False,
    precomputed_proposals: bool = False,
    normalize: bool = True,
    rotation_names: Optional[List[str]] = None,
    verbose: bool = False,
):
    """
    Creates the annotations for the COCO-style dataset from the npy files available, and saves the images in the correct
    directory
    :param image_names: Image names, i.e., the source names
    :param image_destination_dir: The directory the images will end up in
    :param json_dir: The directory where to put the JSON annotation file
    :param json_name: The name of the JSON file
    :param multiple_bboxes: Whether to use multiple bounding boxes, or only the first, for
    example, to only use the main source Optical source, or include others that fall within the
    defined area
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
            print(name.stem)
            print(rotation_names)
            if name.stem in rotation_names:
                extra_rotates.append(i)
                t.append(rotation_names)
        print(f"Matched Names: {t}")
        print(f"Indicies: {extra_rotates} out of {image_names}")
        extra_rotates = np.asarray(extra_rotates)
        image_names = np.asarray(image_names)
        extra_names = image_names[extra_rotates]
        mask = np.ones(len(image_names), np.bool)
        mask[extra_rotates] = 0
        single_names = image_names[mask]
    else:
        single_names = image_names
        extra_names = []
    print(f"Extra Names: {len(extra_names)}")
    print(f"Single Names: {len(single_names)}")
    print(f"Extra Names: {extra_names}")
    print(f"Single Names: {single_names}")
    # List to store single dict for each image
    dataset_dicts = []
    if num_copies > 1:
        manager = Manager()
        pool = Pool(processes=os.cpu_count())
        L = manager.list()
        rotation = np.linspace(0, 180, num_copies)
        [
            pool.apply_async(
                make_single_coco_annotation_set,
                args=[
                    single_names,
                    L,
                    m,
                    image_destination_dir,
                    multiple_bboxes,
                    resize,
                    rotation,
                    convert,
                    all_channels,
                    precomputed_proposals,
                    normalize,
                ],
            )
            for m in range(num_copies)
        ]
        print(len(L))
        # Now do the same for the extra copies, but with more rotations, ~2.5 to equal out multi and single comp sources
        num_multi_copies = int(np.ceil(num_copies * 2.5))
        print(f"Num Multi Copies: {num_multi_copies}")
        multi_rotation = np.linspace(0, 180, num_multi_copies)
        [
            pool.apply_async(
                make_single_coco_annotation_set,
                args=[
                    extra_names,
                    L,
                    m,
                    image_destination_dir,
                    multiple_bboxes,
                    resize,
                    multi_rotation,
                    convert,
                    all_channels,
                    precomputed_proposals,
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
            print(f"COCO annotation file created in '{json_dir}'.\n")
        return 0  # Returns to doesnt go through it again

    # Iterate over all cutouts and their objects (which contain bounding boxes and class labels)
    for m in range(num_copies):
        make_single_coco_annotation_set(
            image_names=image_names,
            record_list=dataset_dicts,
            set_number=m,
            image_destination_dir=image_destination_dir,
            multiple_bboxes=multiple_bboxes,
            resize=resize,
            rotation=rotation,
            convert=convert,
            all_channels=all_channels,
            precomputed_proposals=precomputed_proposals,
            normalize=normalize,
        )
    # Write all image dictionaries to file as one json
    json_path = os.path.join(json_dir, json_name)
    with open(json_path, "wb") as outfile:
        pickle.dump(dataset_dicts, outfile)
    if verbose:
        print(f"COCO annotation file created in '{json_dir}'.\n")


def create_coco_dataset(
    root_directory: str,
    multiple_bboxes: bool = False,
    split_fraction: float = 0.2,
    resize: Optional[Union[Tuple[int], int]] = None,
    rotation: Optional[Union[List[float], float]] = None,
    convert: bool = True,
    all_channels: bool = False,
    precomputed_proposals: bool = False,
    normalize: bool = True,
    subset: str = "",
    multi_rotate_only: Optional[List[str]] = None,
    verbose: bool = False,
):
    """
    Create COCO directory structure, if it doesn't already exist, split the image data, and save it to the correct
    directories, and create the COCO annotation file to be loaded into Detectron2, or other similar models
    :param split_fraction: Fraction of the data for the test set. the validation set is rolled into the test set.
    :param root_directory: root directory for the COCO dataset
    :param multiple_bboxes: Whether to include multiple bounding boxes, or only the main source
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
        l_objects = l_objects[l_objects["LGZ_Assoc"] > 1]
        multi_names = l_objects["Source_Name"].data
    else:
        multi_names = None
    create_coco_annotations(
        data_split["train"],
        json_dir=annotations_directory,
        image_destination_dir=train_directory,
        json_name=f"json_train_prop{precomputed_proposals}_all{all_channels}_multi{multiple_bboxes}_norm{normalize}.pkl",
        multiple_bboxes=multiple_bboxes,
        resize=resize,
        rotation=rotation,
        convert=convert,
        normalize=normalize,
        all_channels=all_channels,
        precomputed_proposals=precomputed_proposals,
        rotation_names=multi_names,
        verbose=verbose,
    )
    create_coco_annotations(
        data_split["train"],
        json_dir=annotations_directory,
        image_destination_dir=train_directory,
        json_name=f"json_train_test_prop{precomputed_proposals}_all{all_channels}_multi{multiple_bboxes}_norm{normalize}.pkl",
        multiple_bboxes=multiple_bboxes,
        resize=resize,
        rotation=None,
        convert=convert,
        normalize=normalize,
        all_channels=all_channels,
        precomputed_proposals=precomputed_proposals,
        rotation_names=None,
        verbose=verbose,
    )
    if len(data_split["val"]) > 0:
        create_coco_annotations(
            data_split["val"],
            json_dir=annotations_directory,
            image_destination_dir=val_directory,
            json_name=f"json_val_prop{precomputed_proposals}_all{all_channels}_multi{multiple_bboxes}_norm{normalize}.pkl",
            multiple_bboxes=multiple_bboxes,
            resize=resize,
            rotation=None,
            convert=convert,
            normalize=normalize,
            all_channels=all_channels,
            precomputed_proposals=precomputed_proposals,
            rotation_names=None,
            verbose=verbose,
        )
    create_coco_annotations(
        data_split["test"],
        json_dir=annotations_directory,
        image_destination_dir=test_directory,
        json_name=f"json_test_prop{precomputed_proposals}_all{all_channels}_multi{multiple_bboxes}_norm{normalize}.pkl",
        multiple_bboxes=multiple_bboxes,
        resize=resize,
        rotation=None,
        convert=convert,
        normalize=normalize,
        all_channels=all_channels,
        precomputed_proposals=precomputed_proposals,
        verbose=verbose,
    )
