from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import torch
import copy
import numpy as np


def source_mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = np.load(dataset_dict["file_name"], allow_pickle=True)
    image, transforms = T.apply_transform_gens([T.Resize((200, 200))], image)
    dataset_dict["image"] = torch.as_tensor(image.astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict