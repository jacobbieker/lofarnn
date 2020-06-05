import os
import pickle
import random
from pathlib import Path
from zlib import crc32

import numpy as np
import cv2
from PIL import Image
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt
from pycocotools import mask

from lofarnn.data.cutouts import convert_to_valid_color, augment_image_and_bboxes
from lofarnn.visualization.cutouts import plot_three_channel_debug
from imgaug.augmentables.bbs import BoundingBox
from multiprocessing import Pool, Process, Manager, Queue


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


def make_single_coco_annotation_set(image_names, L, m,
                                    image_destination_dir=None,
                                    multiple_bboxes=True, resize=None,
                                    rotation=None, convert=True,
                                    all_channels=False, precomputed_proposals=False, segmentation=False, normalize=True, box_seg=True, stats=[],
                                    verbose=False):
    """
    For use with multiprocessing, goes through and does one rotation for the COCO annotations
    :param box_seg: Whether to have segmentation maps and bounding boxes for sources as well as radio components. I.e. bounding boxes with 2 classes:
    Optical source with segmentation of the entire bounding box, and radio component with bounding box of the entire? image and segmentation map inside that
    :param image_names: Image names to load and use for generating dataset
    :param L: Array to add the sources to
    :param m:
    :param image_destination_dir: The destination directory of the images
    :param multiple_bboxes: Whether to include multiple bounding boxes and segmentation maps
    :param resize: What to resize to
    :param rotation: How much to rotate
    :param convert: Whether to convert to PNG and normalize to between 0 and 255 for all channels
    :param all_channels: Whether to use all 10 channels, or just radio, iband, W1 band
    :param precomputed_proposals: Whether to create precomputed proposals
    :param segmentation: Whether to do segmentation or not, if True, or 5, then uses the 5 sigma segmentation maps, if 3, uses the 3 sigma maps
    :param normalize: Whether to normalize input data between 0 and 1
    :param stats:
    :param verbose:
    :return:
    """
    for i, image_name in enumerate(image_names):
        # Get image dimensions and insert them in a python dict
        if convert:
            image_dest_filename = os.path.join(image_destination_dir, image_name.stem + f".{m}.png")
        else:
            if rotation is not None and rotation > 0:
                image_dest_filename = os.path.join(image_destination_dir, image_name.stem + f".{m}.npy")
            else:
                image_dest_filename = os.path.join(image_destination_dir, image_name.stem + f".npy")
        if multiple_bboxes:
            segmap_dest_filename = os.path.join(image_destination_dir, image_name.stem + f".semseg.multi.png")
        else:
            segmap_dest_filename = os.path.join(image_destination_dir, image_name.stem + f".semseg.png")
        image, cutouts, proposal_boxes, segmentation_maps_five, segmentation_maps_three = np.load(image_name,
                                                                    allow_pickle=True)  # mmap_mode might allow faster read
        if segmentation == 3:
            segmentation_maps = segmentation_maps_three.astype(np.uint8)
        else:
            segmentation_maps = segmentation_maps_five.astype(np.uint8)
        prev_shape = image.shape[0]
        image = np.nan_to_num(image)
        print(segmentation_maps[0].shape)
        # Change order to H,W,C for imgaug
        segmentation_maps = np.moveaxis(segmentation_maps,0,-1)
        print(segmentation_maps[0].shape)
        if rotation is not None:
            if type(rotation) == tuple:
                image, cutouts, proposal_boxes, segmentation_maps = augment_image_and_bboxes(image,
                                                                                             cutouts=cutouts,
                                                                                             proposal_boxes=proposal_boxes,
                                                                                             segmentation_maps=segmentation_maps,
                                                                                             angle=rotation[m],
                                                                                             new_size=resize)
            else:
                image, cutouts, proposal_boxes, segmentation_maps = augment_image_and_bboxes(image,
                                                                                             cutouts=cutouts,
                                                                                             proposal_boxes=proposal_boxes,
                                                                                             segmentation_maps=segmentation_maps,
                                                                                             angle=np.random.uniform(
                                                                                                 -rotation, rotation),
                                                                                             new_size=resize)
        else:
            # Need this to convert the bbox coordinates into the correct format
            image, cutouts, proposal_boxes, segmentation_maps = augment_image_and_bboxes(image, cutouts=cutouts,
                                                                                         proposal_boxes=proposal_boxes,
                                                                                         segmentation_maps=segmentation_maps,
                                                                                         angle=0,
                                                                                         new_size=resize)
        width, height, depth = np.shape(image)
        # Move the segmentation maps back to original order
        segmentation_maps = np.moveaxis(segmentation_maps, -1, 0)
        print(segmentation_maps[0].shape)
        if all_channels and depth != 10:
            continue

        # Rescale to between 0 and 1
        scale_size = image.shape[0] / prev_shape
        # First R (Radio) channel
        image[:, :, 0] = convert_to_valid_color(image[:, :, 0], clip=True, lower_clip=0.0, upper_clip=1000,
                                                normalize=normalize, scaling=None)
        for layer in range(image.shape[2]):
            image[:, :, layer] = convert_to_valid_color(image[:, :, layer], clip=True, lower_clip=0.,
                                                        upper_clip=25.,
                                                        normalize=normalize, scaling=None)
        if convert:
            image = np.nan_to_num(image)
            image = (255.0 * image).astype(np.uint8)
            # If converting, only take the first three layers, generally Radio, i band, W1 band
            pil_im = Image.fromarray(image[:, :, :3], 'RGB')
            pil_im.save(image_dest_filename)
        else:
            image = np.nan_to_num(image)
            np.save(image_dest_filename, image)  # Save to the final destination
        if all_channels:
            rec_depth = 10
        else:
            rec_depth = 3
        record = {"file_name": image_dest_filename, "image_id": i, "height": height, "width": width,
                  "depth": rec_depth}

        # Get segmentation maps for bounding boxes
        box_seg_maps = []
        if box_seg and len(cutouts) > 0:
            # Make segmentation map of inside the bounding box as the source
            for i, bbox in enumerate(cutouts):
                box_seg_map = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
                instance_bbox = BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3])
                for i in range(instance_bbox.x1_int, instance_bbox.x2_int):
                    for j in range(instance_bbox.y1_int, instance_bbox.y2_int):
                        box_seg_map[i][j] = 1
                box_seg_maps.append(box_seg_map)
        # Insert bounding boxes and their corresponding classes
        objs = []
        # check if there is no optical source
        try:
            if len(cutouts) > 0:  # There is an optical source
                if not multiple_bboxes:
                    cutouts = [cutouts[0]]  # Only take the first one, the main optical source
                for source_num, bbox in enumerate(cutouts):
                    assert float(bbox[2]) >= float(bbox[0])
                    assert float(bbox[3]) >= float(bbox[1])

                    if bbox[4] == "Other Optical Source":
                        category_id = 0
                    else:
                        category_id = 0
                    stats.append(BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3]).area)

                    obj = {
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": category_id, # For Optical Source
                        "iscrowd": 0
                    }
                    if box_seg:
                        # Now have to create bounding box of segmentation map
                        #TODO maybe make the bounding box smaller around segmentation map
                        seg_obj = {
                            "bbox": [0.0,0.0,float(image.shape[0]-1), float(image.shape[1]-1)],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": 1, # For Radio Component
                            "iscrowd": 0,
                            "segmentation": mask.encode(np.asarray(segmentation_maps[source_num], order="F"))
                        }
                        objs.append(seg_obj)

                    if segmentation and not box_seg:
                        obj["segmentation"] = mask.encode(np.asarray(segmentation_maps[source_num], order="F"))
                    elif box_seg:
                        obj["segmentation"] = mask.encode(np.asarray(box_seg_maps[source_num], order="F"))
                    objs.append(obj)
        except Exception as e:
            print(e)
            print("No Optical source found")
        if precomputed_proposals:
            if box_seg:
                semseg_prop = np.asarray([[0.0,0.0,float(image.shape[0]-1), float(image.shape[1]-1)] for _ in cutouts])
                proposal_boxes = np.concatenate([proposal_boxes, semseg_prop])
            record["proposal_boxes"] = proposal_boxes
            record["proposal_objectness_logits"] = np.ones(len(proposal_boxes))  # TODO Not sure this is right
            record["proposal_bbox_mode"] = BoxMode.XYXY_ABS
        record["annotations"] = objs
        if segmentation:
            # Now save out the ground truth semantic segmentation mask
            if not multiple_bboxes:
                ground_truth_mask = segmentation_maps[0] # Choose the first one, the primary source
                if box_seg:
                    # Have to add ground truth for the source segmentations
                    np.add(ground_truth_mask, box_seg_maps[0])
            else:
                # Have to go through and add all the other segmaps into a single one, leaving out full background one
                ground_truth_mask = np.zeros(segmentation_maps[0].shape)
                for source_num, source_segmap in enumerate(segmentation_maps[:-1]):
                    source_segmap = np.where(source_segmap > 0, source_num+1, 0)
                    np.add(ground_truth_mask, source_segmap)
                if box_seg and len(box_seg_maps) > 0:
                    for source_num, source_segmap in enumerate(box_seg_maps):
                        source_segmap = np.where(source_segmap > 0, len(segmentation_maps) + source_num + 1, 0)
                        np.add(ground_truth_mask, source_segmap)
            # Now save out the ground truth map
            ground_truth_mask = Image.fromarray(ground_truth_mask)
            ground_truth_mask.save(segmap_dest_filename)
            record["sem_seg_file_name"] = segmap_dest_filename
        L.append(record)


def create_coco_annotations(image_names,
                            image_destination_dir=None,
                            json_dir='', json_name='json_data.pkl',
                            multiple_bboxes=True, resize=None, rotation=None, convert=True, all_channels=False,
                            precomputed_proposals=False, segmentation=False, normalize=True,
                            verbose=False):
    """
    Creates the annotations for the COCO-style dataset from the npy files available, and saves the images in the correct
    directory
    :param segmentation: Whether to include the segmentation maps or not
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
            num_copies = 50
    else:
        num_copies = 1
    # List to store single dict for each image
    dataset_dicts = []
    if num_copies > 1:
        manager = Manager()
        pool = Pool(processes=os.cpu_count())
        L = manager.list()
        bbox_size = manager.list()
        [pool.apply_async(make_single_coco_annotation_set,
                          args=[image_names, L, m, image_destination_dir, multiple_bboxes, resize, rotation, convert,
                                all_channels, precomputed_proposals, segmentation, normalize, True, bbox_size,
                                verbose]) for m in range(num_copies)]
        pool.close()
        pool.join()
        print(len(L))
        print(np.mean(bbox_size))
        print(np.std(bbox_size))
        print(np.max(bbox_size))
        print(np.min(bbox_size))
        for element in L:
            dataset_dicts.append(element)
        # Write all image dictionaries to file as one json
        json_path = os.path.join(json_dir, json_name)
        with open(json_path, "wb") as outfile:
            pickle.dump(dataset_dicts, outfile)
        if verbose:
            print(f'COCO annotation file created in \'{json_dir}\'.\n')
        return 0  # Returns to doesnt go through it again

    # Iterate over all cutouts and their objects (which contain bounding boxes and class labels)
    bbox_size = []
    for m in range(num_copies):
        make_single_coco_annotation_set(image_names, dataset_dicts, m, image_destination_dir, multiple_bboxes, resize,
                                        rotation, convert, all_channels, precomputed_proposals, segmentation, normalize, True, bbox_size,
                                        verbose)
    # Write all image dictionaries to file as one json
    # print(np.mean(bbox_size))
    # print(np.std(bbox_size))
    # print(np.max(bbox_size))
    # print(np.min(bbox_size))
    plt.hist(bbox_size, bins=50)
    plt.show()
    json_path = os.path.join(json_dir, json_name)
    with open(json_path, "wb") as outfile:
        pickle.dump(dataset_dicts, outfile)
    if verbose:
        print(f'COCO annotation file created in \'{json_dir}\'.\n')


def create_coco_dataset(root_directory, multiple_bboxes=False, split_fraction=0.2, resize=None, rotation=None,
                        convert=True, all_channels=False, precomputed_proposals=False, segmentation=False, normalize=True,
                        verbose=False):
    """
    Create COCO directory structure, if it doesn't already exist, split the image data, and save it to the correct
    directories, and create the COCO annotation file to be loaded into Detectron2, or other similar models
    :param split_fraction: Fraction of the data for the test set. the validation set is rolled into the test set.
    :param root_directory: root directory for the COCO dataset
    :param multiple_bboxes: Whether to include multiple bounding boxes, or only the main source
    :param resize: Image size to resize to, or None if not resizing
    :param convert: Whether to convert npy files to png, or to keep them in the original format, useful for SourceMapper
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
                            json_name=f"json_train_prop{precomputed_proposals}_all{all_channels}_multi{multiple_bboxes}_seg{segmentation}_norm{normalize}.pkl",
                            multiple_bboxes=multiple_bboxes,
                            resize=resize,
                            rotation=rotation,
                            convert=convert,
                            segmentation=segmentation,
                            normalize=normalize,
                            all_channels=all_channels,
                            precomputed_proposals=precomputed_proposals,
                            verbose=verbose)
    if len(data_split["val"]) > 0:
        create_coco_annotations(data_split["val"],
                                json_dir=annotations_directory,
                                image_destination_dir=val_directory,
                                json_name=f"json_val_prop{precomputed_proposals}_all{all_channels}_multi{multiple_bboxes}_seg{segmentation}_norm{normalize}.pkl",
                                multiple_bboxes=multiple_bboxes,
                                resize=resize,
                                rotation=rotation,
                                convert=convert,
                                segmentation=segmentation,
                                normalize=normalize,
                                all_channels=all_channels,
                                precomputed_proposals=precomputed_proposals,
                                verbose=verbose)
    create_coco_annotations(data_split["test"],
                            json_dir=annotations_directory,
                            image_destination_dir=test_directory,
                            json_name=f"json_test_prop{precomputed_proposals}_all{all_channels}_multi{multiple_bboxes}_seg{segmentation}_norm{normalize}.pkl",
                            multiple_bboxes=multiple_bboxes,
                            resize=resize,
                            rotation=rotation,
                            convert=convert,
                            segmentation=segmentation,
                            normalize=normalize,
                            all_channels=all_channels,
                            precomputed_proposals=precomputed_proposals,
                            verbose=verbose)


# Function to check test set's identifier.
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


# Function to split train/test
def split_train_test_by_id(data, test_ratio):
    in_test_set = np.asarray([test_set_check(crc32(str(x).split("/")[-1].encode()), test_ratio) for x in data])
    return data[~in_test_set], data[in_test_set]


def split_data(image_directory, split=0.2):
    """
    Split up the data and return which images should go to which train, test, val directory
    :param image_directory: The directory where all the images are located, i.e. the "all" directory
    :param split: Fraction of the data for the test set. the validation set is rolled into the test set.
    :return: A dict containing which images go to which directory
    """

    image_paths = Path(image_directory).rglob("*.npy")
    im_paths = []
    for p in image_paths:
        im_paths.append(p)
    print(len(im_paths))
    train_images, test_images = split_train_test_by_id(np.asarray(im_paths), split)
    print(len(train_images))
    print(len(test_images))
    return {"train": train_images,
            "val": [],
            "test": test_images}


def get_all_single_image_std_mean(image, num_layers, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6,
                                  layer_7, layer_8, layer_9):
    try:
        data = Image.open(image).convert('RGB')
    except:
        try:
            data = np.nan_to_num(np.load(image, allow_pickle=True))
        except:
            print("Failed")
    image = np.array(data)
    layer_0.append(image[:, :, 0])
    layer_1.append(image[:, :, 1])
    layer_2.append(image[:, :, 2])
    if num_layers > 3:
        layer_3.append(image[:, :, 3])
        layer_4.append(image[:, :, 4])
        layer_5.append(image[:, :, 5])
        layer_6.append(image[:, :, 6])
        layer_7.append(image[:, :, 7])
        layer_8.append(image[:, :, 8])
        layer_9.append(image[:, :, 9])


def get_all_pixel_mean_and_std_multi(image_paths, num_layers=3):
    """
    Get the channelwise mean and std dev of all the images
    :param image_paths: Paths to the images
    :return:
    """
    manager = Manager()
    layer_0 = manager.list()
    layer_1 = manager.list()
    layer_2 = manager.list()
    layer_3 = manager.list()
    layer_4 = manager.list()
    layer_5 = manager.list()
    layer_6 = manager.list()
    layer_7 = manager.list()
    layer_8 = manager.list()
    layer_9 = manager.list()
    pool = Pool(processes=os.cpu_count())
    [pool.apply_async(get_all_single_image_std_mean,
                      args=[image, num_layers, layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7,
                            layer_8, layer_9]) for
     image
     in image_paths]
    pool.close()
    pool.join()
    layer_0 = np.concatenate([np.array(i) for i in layer_0])
    layer_2 = np.concatenate([np.array(i) for i in layer_2])
    layer_1 = np.concatenate([np.array(i) for i in layer_1])
    if num_layers > 3:
        layer_3 = np.concatenate([np.array(i) for i in layer_3])
        layer_4 = np.concatenate([np.array(i) for i in layer_4])
        layer_5 = np.concatenate([np.array(i) for i in layer_5])
        layer_6 = np.concatenate([np.array(i) for i in layer_6])
        layer_7 = np.concatenate([np.array(i) for i in layer_7])
        layer_8 = np.concatenate([np.array(i) for i in layer_8])
        layer_9 = np.concatenate([np.array(i) for i in layer_9])
    print(layer_0.shape)
    print(f"Layer 0 Mean: {np.mean(layer_0)} Std: {np.std(layer_0)}")
    print(f"Layer 1 Mean: {np.mean(layer_1)} Std: {np.std(layer_1)}")
    print(f"Layer 2 Mean: {np.mean(layer_2)} Std: {np.std(layer_2)}")
    if num_layers > 3:
        print(f"Layer 3 Mean: {np.mean(layer_3)} Std: {np.std(layer_3)}")
        print(f"Layer 4 Mean: {np.mean(layer_4)} Std: {np.std(layer_4)}")
        print(f"Layer 5 Mean: {np.mean(layer_5)} Std: {np.std(layer_5)}")
        print(f"Layer 6 Mean: {np.mean(layer_6)} Std: {np.std(layer_6)}")
        print(f"Layer 7 Mean: {np.mean(layer_7)} Std: {np.std(layer_7)}")
        print(f"Layer 8 Mean: {np.mean(layer_8)} Std: {np.std(layer_8)}")
        print(f"Layer 9 Mean: {np.mean(layer_9)} Std: {np.std(layer_9)}")
        print(f"[[{np.round(np.mean(layer_0), 5)},{np.round(np.mean(layer_1), 5)},{np.round(np.mean(layer_2), 5)},"
              f"{np.round(np.mean(layer_3), 5)},{np.round(np.mean(layer_4), 5)},{np.round(np.mean(layer_5), 5)},"
              f"{np.round(np.mean(layer_6), 5)},{np.round(np.mean(layer_7), 5)},{np.round(np.mean(layer_8), 5)},"
              f"{np.round(np.mean(layer_9), 5)}]]")
        print(f"[[{np.round(np.std(layer_0), 5)},{np.round(np.std(layer_1), 5)},{np.round(np.std(layer_2), 5)},"
              f"{np.round(np.std(layer_3), 5)},{np.round(np.std(layer_4), 5)},{np.round(np.std(layer_5), 5)},"
              f"{np.round(np.std(layer_6), 5)},{np.round(np.std(layer_7), 5)},{np.round(np.std(layer_8), 5)},"
              f"{np.round(np.std(layer_9), 5)}]]")
    else:
        print(f"[[{np.round(np.mean(layer_0), 5)},{np.round(np.mean(layer_1), 5)},{np.round(np.mean(layer_2), 5)}]]")
        print(f"[[{np.round(np.std(layer_0), 5)},{np.round(np.std(layer_1), 5)},{np.round(np.std(layer_2), 5)}]]")
    print(layer_0.shape)
