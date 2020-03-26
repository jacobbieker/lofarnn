import os
import pickle
import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from detectron2.structures import BoxMode

from lofarnn.data.cutouts import convert_to_valid_color, RandomRotate, Rotate
from lofarnn.visualization.cutouts import plot_three_channel_debug


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
                            multiple_bboxes=True, resize=None, rotation=None, verbose=False):
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
    :return:
    """

    if rotation is not None:
        if type(rotation) == tuple:
            num_copies = len(rotation)
        else:
            num_copies = 50
    else:
        num_copies = 1
    # List to store single dict for each image
    dataset_dicts = []
    # Iterate over all cutouts and their objects (which contain bounding boxes and class labels)
    for m in range(num_copies):
        for i, image_name in enumerate(image_names):
            # Get image dimensions and insert them in a python dict
            image_dest_filename = os.path.join(image_destination_dir, image_name.stem + f"{m}.png")
            image, cutouts = np.load(image_name, allow_pickle=True)  # mmap_mode might allow faster read
            prev_shape = image.shape[0]
            if rotation is not None:
                if type(rotation) == tuple:
                    transforms = Rotate(angle=rotation[m])
                    tmp_bboxes = []
                    for index, bbox in enumerate(cutouts):
                        # convert to proper coordinates
                        tmp_bbox = [float(bbox[1]), float(bbox[0]), float(bbox[3]), float(bbox[2])]
                        tmp_bboxes.append(tmp_bbox)
                    image, tmp_bboxes = transforms(image, np.asarray(tmp_bboxes))
                    for index, _ in enumerate(cutouts):
                        # Convert back to the messed up coordinates that are expected
                        cutouts[index] = [float(tmp_bboxes[index][1]), float(tmp_bboxes[index][0]),
                                          float(tmp_bboxes[index][3]), float(tmp_bboxes[index][2]),
                                          cutouts[index][4], cutouts[index][5]]
            else:
                transforms = RandomRotate(angle=rotation)
                tmp_bboxes = []
                for index, bbox in enumerate(cutouts):
                    # convert to proper coordinates
                    tmp_bbox = [float(bbox[1]), float(bbox[0]), float(bbox[3]), float(bbox[2])]
                    tmp_bboxes.append(tmp_bbox)
                image, tmp_bboxes = transforms(image, np.asarray(tmp_bboxes))
                for index, _ in enumerate(cutouts):
                    # Convert back to the messed up coordinates that are expected
                    cutouts[index] = [float(tmp_bboxes[index][1]), float(tmp_bboxes[index][0]),
                                      float(tmp_bboxes[index][3]), float(tmp_bboxes[index][2]),
                                      cutouts[index][4], cutouts[index][5]]
            if resize is not None:
                # Resize the image and boxes
                for index, box in enumerate(cutouts):
                    print(box)
                    cutouts[index] = scale_box(image, box, resize)
                image = resize_array(image, resize)
            width, height, depth = np.shape(image)
            # Rescale to between 0 and 1
            scale_size = image.shape[0] / prev_shape
            if True:
                plot_three_channel_debug(image, cutouts, 1, cutouts[0][5],
                                         save_path=os.path.join("/home/jacob/Development/LOFAR-ML/data/",
                                                                image_name.name + ".jpg"))
            # First R channel
            image[:, :, 0] = convert_to_valid_color(image[:, :, 0], clip=True, lower_clip=0.0, upper_clip=1000,
                                                    normalize=True, scaling=None)
            image[:, :, 1] = convert_to_valid_color(image[:, :, 1], clip=True, lower_clip=0., upper_clip=25.,
                                                    normalize=True, scaling=None)
            image[:, :, 2] = convert_to_valid_color(image[:, :, 2], clip=True, lower_clip=0., upper_clip=25.,
                                                    normalize=True, scaling=None)
            image = (255.0 * image).astype(np.uint8)
            im = Image.fromarray(image, 'RGB')
            if True:
                plot_three_channel_debug(image, cutouts, scale_size, cutouts[0][5],
                                         save_path=os.path.join("/home/jacob/Development/LOFAR-ML/data/",
                                                                image_name.name + ".png"))
            im.save(image_dest_filename)
            # np.save(image_dest_filename, image)  # Save to the final destination
            record = {"file_name": image_dest_filename, "image_id": i, "height": height, "width": width}

            # Insert bounding boxes and their corresponding classes
            # print('scale_factor:',cutout.scale_factor)
            objs = []
            if not multiple_bboxes:
                cutouts = [cutouts[0]]  # Only take the first one, the main optical source
            for bbox in cutouts:
                assert float(bbox[2]) > float(bbox[0])
                assert float(bbox[3]) > float(bbox[1])

                if bbox[4] == "Other Optical Source":
                    category_id = 1
                else:
                    category_id = 0

                obj = {
                    "bbox": [float(bbox[1]), float(bbox[0]), float(bbox[3]), float(bbox[2])],
                    "bbox_mode": BoxMode.XYXY_ABS,
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


def create_coco_dataset(root_directory, multiple_bboxes=False, split_fraction=(0.6, 0.8), resize=None, verbose=False):
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
                            all_directory=all_directory,
                            verbose=verbose)
    create_coco_annotations(data_split["val"],
                            json_dir=annotations_directory,
                            image_destination_dir=val_directory,
                            json_name=f"json_val.pkl",
                            multiple_bboxes=multiple_bboxes,
                            resize=resize,
                            all_directory=all_directory,
                            verbose=verbose)
    create_coco_annotations(data_split["test"],
                            json_dir=annotations_directory,
                            image_destination_dir=test_directory,
                            json_name=f"json_test.pkl",
                            multiple_bboxes=multiple_bboxes,
                            resize=resize,
                            all_directory=all_directory,
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
    train_images = im_paths[:int(len(im_paths) * split[0])]
    val_images = im_paths[int(len(im_paths) * split[0]):int(len(im_paths) * split[1])]
    test_images = im_paths[int(len(im_paths) * split[1]):]

    return {"train": train_images,
            "val": val_images,
            "test": test_images}


def resize_array(arr, new_size):
    """Resizes numpy array to a specified width and height using specified interpolation"""
    scale_factor = new_size / arr.shape[0]
    return cv2.resize(arr, dsize=(new_size, new_size), interpolation=cv2.INTER_LINEAR)


def scale_box(arr, bounding_box, new_size):
    scale_factor = new_size / arr.shape[0]
    bounding_box[1] = float(bounding_box[1]) * scale_factor
    bounding_box[3] = float(bounding_box[3]) * scale_factor
    bounding_box[0] = float(bounding_box[0]) * scale_factor
    bounding_box[2] = float(bounding_box[2]) * scale_factor
    bounding_box[5] = (float(bounding_box[5][0]) * scale_factor, float(bounding_box[5][1]) * scale_factor)
    return bounding_box


def get_pixel_mean_and_std(image_paths):
    """
    Get the channelwise mean and std dev of all the images
    :param image_paths: Paths to the images
    :return:
    """
    r = []
    g = []
    b = []
    r_max = 0
    r_min = 1000
    g_max = 0
    g_min = 1000
    b_max = 0
    b_min = 1000
    for image in image_paths:
        data = np.load(image, allow_pickle=True)[0]
        if np.min(data[:, :, 0]) < r_min:
            r_min = np.min(data[:, :, 0])
        if np.min(data[:, :, 1]) < g_min:
            g_min = np.min(data[:, :, 1])
        if np.min(data[:, :, 2]) < b_min:
            b_min = np.min(data[:, :, 2])
        if np.max(data[:, :, 0]) > r_max:
            r_max = np.max(data[:, :, 0])
        if np.max(data[:, :, 1]) > g_max:
            g_max = np.max(data[:, :, 0])
        if np.max(data[:, :, 1]) > b_max:
            b_max = np.max(data[:, :, 0])
        r_val = np.reshape(data[:, :, 0], -1)
        g_val = np.reshape(data[:, :, 1], -1)
        b_val = np.reshape(data[:, :, 2], -1)

    # print(f"R Mean: {r_mean}, {r_std} \n G Mean: {g_mean}, {g_std} \n B Mean: {b_mean}, {b_std}")

    return (r_max, r_min), (g_max, g_min), (b_max, b_min)
