import os
import pickle
import random
from pathlib import Path

import numpy as np
from PIL import Image


def mkdirs_safe(directory_list):
    """When given a list containing directories,
    checks if these exist, if not creates them."""
    assert isinstance(directory_list, list)
    for directory in directory_list:
        os.makedirs(directory, exist_ok=True)


def create_recursive_directories(prepend_path, current_dir, dictionary):
    """Give it a base, a current_dir to create and a dictionary of stuff yet to create.
    See create_VOC_style_directory_structure for a use case scenario."""
    mkdirs_safe([os.path.join(prepend_path, current_dir)])
    if dictionary[current_dir] is None:
        return
    else:
        for key in dictionary[current_dir].keys():
            create_recursive_directories(os.path.join(prepend_path, current_dir), key,
                                         dictionary[current_dir])


def create_coco_style_directory_structure(root_directory, suffix='', verbose=False):
    """
     We will create a directory structure identical to that of the CLARAN
     The root directory is the directory in which the directory named 'RGZdevkit' will be placed.
     The structure contained will be as follows:
     LGZ_COCOstyle{suffix}/
        |-- Annotations/
             |-- *.json (Annotation files)
        |-- all/ (train,test,val split directory)
             |-- *.png (Image files)
        |-- train/ (train,test,val split directory)
             |-- *.png (Image files)
        |-- val/ (train,test,val split directory)
             |-- *.png (Image files)
        |-- test/ (train,test,val split directory)
             |-- *.png (Image files)
    """
    directories_to_make = {f'COCO{suffix}':
                               {'annotations': None,
                                'all': None,
                                'train': None,
                                'val': None,
                                'test': None}}
    create_recursive_directories(root_directory, f'COCO{suffix}', directories_to_make)
    if verbose:
        print(f'COCO style directory structure created in \'{root_directory}\'.\n')
    all_directory, train_directory, val_directory, test_directory, annotations_directory = \
        os.path.join(root_directory, f'COCO{suffix}', 'all'), \
        os.path.join(root_directory, f'COCO{suffix}', 'train'), \
        os.path.join(root_directory, f'COCO{suffix}', 'val'), \
        os.path.join(root_directory, f'COCO{suffix}', 'test'), \
        os.path.join(root_directory, f'COCO{suffix}', 'annotations')
    return all_directory, train_directory, val_directory, test_directory, annotations_directory


def create_coco_annotations(image_names,
                            image_destination_dir=None,
                            json_dir='', json_name='json_data.pkl',
                            multiple_bboxes=True, resize=None, verbose=False):
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
    :return:
    """

    # List to store single dict for each image
    dataset_dicts = []

    # Iterate over all cutouts and their objects (which contain bounding boxes and class labels)
    for i, image_name in enumerate(image_names):
        # Get image dimensions and insert them in a python dict
        image_dest_filename = os.path.join(image_destination_dir, image_name.name)
        image, cutouts = np.load(image_name, allow_pickle=True)  # mmap_mode might allow faster read
        if resize is not None:
            # Resize the image and boxes
            for index, box in enumerate(cutouts):
                cutouts[index] = scale_box(image, box, resize)
            image = resize_array(image, resize, resize)
        width, height, depth = np.shape(image)
        np.save(image_dest_filename, image)  # Save to the final destination
        record = {"file_name": image_dest_filename, "image_id": i, "height": height, "width": width}

        # Insert bounding boxes and their corresponding classes
        # print('scale_factor:',cutout.scale_factor)
        objs = []
        cache_list = []
        if not multiple_bboxes:
            cutouts = [cutouts[0]]  # Only take the first one, the main optical source
        for bbox in cutouts:
            if bbox in cache_list:
                continue
            cache_list.append(bbox)
            assert float(bbox[2]) > float(bbox[0])
            assert float(bbox[3]) > float(bbox[1])

            if bbox[4] == "Other Optical Source":
                category_id = 1
            else:
                category_id = 0

            obj = {
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "bbox_mode": None,
                # "segmentation": [poly],
                "category_id": category_id,
                "iscrowd": 0
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    # Write all image dictionaries to file as one json
    json_path = os.path.join(json_dir, json_name)
    with open(json_path, "wb") as outfile:
        pickle.dump(dataset_dicts, outfile)
    if verbose:
        print(f'COCO annotation file created in \'{json_dir}\'.\n')


def create_coco_dataset(root_directory, multiple_bboxes=False, split_fraction=(0.6,0.8), resize=None, verbose=False):
    """
    Create COCO directory structure, if it doesn't already exist, split the image data, and save it to the correct
    directories, and create the COCO annotation file to be loaded into Detectron2, or other similar models
    :param split_fraction: Tuple of train, val, test split,
    in form of (train fraction, val fraction), with the rest being put in test directory
    :param root_directory: root directory for the COCO dataset
    :param multiple_bboxes: Whether to include multiple bounding boxes, or only the main source
    :param resize: Image size to resize to, or None if not resizing
    :param verbose: Whether to print more data to stdout or not
    :return:
    """

    all_directory, train_directory, val_directory, test_directory, annotations_directory \
        = create_coco_style_directory_structure(root_directory, verbose=verbose)

    # Gather data from all_directory
    data_split = split_data(all_directory, split=split_fraction)

    create_coco_annotations(data_split["train"],
                            json_dir=annotations_directory,
                            image_destination_dir=train_directory,
                            json_name=f"json_train.pkl",
                            multiple_bboxes=multiple_bboxes,
                            resize=resize,
                            verbose=verbose)
    create_coco_annotations(data_split["val"],
                            json_dir=annotations_directory,
                            image_destination_dir=val_directory,
                            json_name=f"json_val.pkl",
                            multiple_bboxes=multiple_bboxes,
                            resize=resize,
                            verbose=verbose)
    create_coco_annotations(data_split["test"],
                            json_dir=annotations_directory,
                            image_destination_dir=test_directory,
                            json_name=f"json_test.pkl",
                            multiple_bboxes=multiple_bboxes,
                            resize=resize,
                            verbose=verbose)


def split_data(image_directory, split=(0.6, 0.8)):
    """
    Split up the data and return which images should go to which train, test, val directory
    :param image_directory: The directory where all the images are located, i.e. the "all" directory
    :param split: Tuple of train, val, test split,
    in form of (train fraction, val fraction), with the rest being put in test directory
    :return: A dict containing which images go to which directory
    """

    image_paths = Path(image_directory).rglob("*.npy")
    im_paths = []
    for p in image_paths:
        im_paths.append(p)
    random.shuffle(im_paths)
    train_images = im_paths[:int(len(im_paths)*split[0])]
    val_images = im_paths[int(len(im_paths)*split[0]):int(len(im_paths)*split[1])]
    test_images = im_paths[int(len(im_paths)*split[1]):]

    return {"train": train_images,
            "val": val_images,
            "test": test_images}


def resize_array(arr, width, height, interpolation=Image.BILINEAR):
    """Resizes numpy array to a specified width and height using specified interpolation"""
    return np.array(Image.fromarray(arr).resize((width,height),interpolation))


def scale_box(arr, bounding_box, new_size):
    scale_factor = arr.shape[0] / new_size
    bounding_box[1] = bounding_box[1]*scale_factor
    bounding_box[3] = bounding_box[3]*scale_factor
    bounding_box[0] = bounding_box[0]*scale_factor
    bounding_box[2] = bounding_box[2]*scale_factor
    return bounding_box
