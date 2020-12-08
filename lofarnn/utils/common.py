import os
from pathlib import Path
from typing import Any, Union, List, Tuple, Dict
from zlib import crc32

import numpy as np


def mkdirs_safe(directory_list: list):
    """When given a list containing directories,
    checks if these exist, if not creates them."""
    assert isinstance(directory_list, list)
    for directory in directory_list:
        os.makedirs(directory, exist_ok=True)


def create_recursive_directories(prepend_path: str, current_dir: str, dictionary: dict):
    """Give it a base, a current_dir to create and a dictionary of stuff yet to create.
    See create_VOC_style_directory_structure for a use case scenario."""
    mkdirs_safe([os.path.join(prepend_path, current_dir)])
    if dictionary[current_dir] is None:
        return
    else:
        for key in dictionary[current_dir].keys():
            create_recursive_directories(
                os.path.join(prepend_path, current_dir), key, dictionary[current_dir]
            )


def create_coco_style_directory_structure(
    root_directory: str, suffix: str = "", verbose: bool = False
):
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
    directories_to_make = {
        f"COCO{suffix}": {
            "annotations": None,
            "all": None,
            "train": None,
            "val": None,
            "test": None,
        }
    }
    create_recursive_directories(root_directory, f"COCO{suffix}", directories_to_make)
    if verbose:
        print(f"COCO style directory structure created in '{root_directory}'.\n")
    (
        all_directory,
        train_directory,
        val_directory,
        test_directory,
        annotations_directory,
    ) = (
        os.path.join(root_directory, f"COCO{suffix}", "all"),
        os.path.join(root_directory, f"COCO{suffix}", "train"),
        os.path.join(root_directory, f"COCO{suffix}", "val"),
        os.path.join(root_directory, f"COCO{suffix}", "test"),
        os.path.join(root_directory, f"COCO{suffix}", "annotations"),
    )
    return (
        all_directory,
        train_directory,
        val_directory,
        test_directory,
        annotations_directory,
    )


def test_set_check(identifier: Any, test_ratio: float) -> float:
    return crc32(np.int64(identifier)) & 0xFFFFFFFF < test_ratio * 2 ** 32


def split_train_test_by_id(
    data: Union[np.ndarray, List[str]], test_ratio: float
) -> Tuple[Union[Union[str, List[str]], Any], Union[list, Any]]:
    in_test_set = np.asarray(
        [
            test_set_check(crc32(str(x).split("/")[-1].encode()), test_ratio)
            for x in data
        ]
    )
    return data[~in_test_set], data[in_test_set]


def split_data(
    image_directory: str, val_split: float = 0.2, test_split: float = 0.2
) -> Dict[str, List[str]]:
    """
    Split up the data and return which images should go to which train, test, val directory
    :param image_directory: The directory where all the images are located, i.e. the "all" directory
    :param test_split: Fraction of the data for the test set. the validation set is rolled into the test set.
    :param val_split: Fraction of data in validation set
    :return: A dict containing which images go to which directory
    """

    image_paths = Path(image_directory).rglob("*.npy")
    im_paths = []
    for p in image_paths:
        im_paths.append(p)
    print(len(im_paths))
    train_images, test_images = split_train_test_by_id(
        np.asarray(im_paths), val_split + test_split
    )
    val_images, test_images = split_train_test_by_id(test_images, val_split)
    print(len(train_images))
    print(len(val_images))
    print(len(test_images))
    return {"train": train_images, "val": val_images, "test": test_images}
