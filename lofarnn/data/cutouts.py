from typing import Union, Optional, List, Tuple, Any

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from astropy.visualization import (
    MinMaxInterval,
    SqrtStretch,
    ManualInterval,
    ImageNormalize,
)
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from skimage.transform import rotate


def convert_to_valid_color(
    image_color: np.ndarray,
    clip: bool = False,
    lower_clip: float = 0.0,
    upper_clip: float = 1.0,
    normalize: bool = False,
    scaling: Optional[str] = None,
    simple_norm: bool = False,
) -> np.ndarray:
    """
    Convert the channel to a valid 0-1 range for RGB images
    """

    if simple_norm:
        interval = MinMaxInterval()
        norm = ImageNormalize()
        return norm(interval(image_color))

    if clip:
        image_color = np.clip(image_color, lower_clip, upper_clip)
    if normalize:
        interval = ManualInterval(lower_clip, upper_clip)
    else:
        interval = MinMaxInterval()
    if scaling == "sqrt":
        stretch = SqrtStretch()
        image_color = stretch(interval(image_color))
    else:
        norm = ImageNormalize()
        image_color = norm(interval(image_color))

    return image_color


def rotate_image_stacks(
    image_stack: np.ndarray, rotation_angle: Union[float, int] = 0
) -> np.ndarray:
    image_stack = rotate(
        image_stack,
        -rotation_angle,
        resize=True,
        center=None,
        order=1,
        mode="constant",
        cval=0,
        clip=True,
        preserve_range=True,
    )
    return image_stack


def make_bounding_box(
    source_location: Union[List[List[Union[float, int]]], np.ndarray]
) -> Tuple[
    int,
    int,
    int,
    int,
    Union[List[Union[float, int]], Any],
    Union[List[Union[float, int]], Any],
]:
    """
    Creates a bounding box and returns it in (xmin, ymin, xmax, ymax, class_name, source location) format
    for use with the source location in pixel values
    :return: Bounding box coordinates for COCO style annotation
    """
    # Now create box, which will be accomplished by taking int to get xmin, ymin, and int + 1 for xmax, ymax
    xmin = int(np.floor(source_location[5][0])) - 1
    ymin = int(np.floor(source_location[5][1])) - 1
    ymax = ymin + 2
    xmax = xmin + 2

    return xmin, ymin, xmax, ymax, source_location[4], source_location[5]


def augment_image_and_bboxes(
    image: np.ndarray,
    cutouts: Union[List[Tuple[float]], np.ndarray],
    proposal_boxes: Union[List[Tuple[float]], np.ndarray],
    angle: Optional[Union[float, int]],
    new_size: Optional[Union[int, Tuple[int]]],
    verbose: bool = False,
) -> Tuple[Any, Union[List[Tuple[float]], np.ndarray], np.ndarray]:
    bounding_boxes = []
    prop_boxes = []
    seg_boxes = []
    new_crop = int(
        np.ceil(image.shape[0] / np.sqrt(2))
    )  # Is 200 for 282, which is what is wanted, and works for others
    seq = iaa.Sequential(
        [
            iaa.Affine(rotate=angle),
            iaa.CropToFixedSize(width=new_crop, height=new_crop, position="center"),
        ]
    )
    for cutout in cutouts:
        bounding_boxes.append(
            BoundingBox(
                cutout[1] + 0.5, cutout[0] + 0.5, cutout[3] + 0.5, cutout[2] + 0.5
            )
        )
    for pbox in proposal_boxes:
        prop_boxes.append(
            BoundingBox(pbox[1] + 0.5, pbox[0] + 0.5, pbox[3] + 0.5, pbox[2] + 0.5)
        )

    bbs = BoundingBoxesOnImage(bounding_boxes, shape=image.shape)
    pbbs = BoundingBoxesOnImage(prop_boxes, shape=image.shape)
    if seg_boxes:
        sbbs = BoundingBoxesOnImage(seg_boxes, shape=image.shape)
        _, sbbs = seq(image=image, bounding_boxes=sbbs)
    _, bbs = seq(image=image, bounding_boxes=bbs)
    image, pbbs = seq(image=image, bounding_boxes=pbbs)
    # Rescale image and bounding boxes
    if type(new_size) == int:
        image_rescaled = ia.imresize_single_image(image, (new_size, new_size))
    else:
        image_rescaled = ia.imresize_single_image(
            image, (image.shape[0], image.shape[1])
        )
    bbs_rescaled = bbs.on(image_rescaled)
    pbbs_rescaled = pbbs.on(image_rescaled)
    if seg_boxes:
        sbbs_rescaled = sbbs.on(image_rescaled)
        sbbs_rescaled = sbbs_rescaled.remove_out_of_image(
            partly=False
        ).clip_out_of_image()

    # Remove bounding boxes that go out of bounds
    pbbs_rescaled = pbbs_rescaled.remove_out_of_image(partly=False).clip_out_of_image()
    # But only clip source bounding boxes that are partly out of frame, so that no sources are lost
    bbs_rescaled = bbs_rescaled.remove_out_of_image(partly=False).clip_out_of_image()
    # Also remove those and clip of segmentation maps
    for index, bbox in enumerate(bbs_rescaled):
        cutouts[index][0] = bbox.x1
        cutouts[index][1] = bbox.y1
        cutouts[index][2] = bbox.x2
        cutouts[index][3] = bbox.y2
    # Convert proposal boxes as well
    pbs = []
    for index, bbox in enumerate(pbbs_rescaled):
        pbs.append(np.asarray((bbox.x1, bbox.y1, bbox.x2, bbox.y2)))
    pbs = np.asarray(pbs)
    sbs = []
    try:
        for index, bbox in enumerate(sbbs_rescaled):
            sbs.append(np.asarray((bbox.x1, bbox.y1, bbox.x2, bbox.y2)))
    except:
        if verbose:
            print("Empty sbbs_rescaled")
    sbs = np.asarray(sbs)
    return image_rescaled, cutouts, pbs
