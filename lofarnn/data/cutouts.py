import numpy as np
from skimage.transform import rotate
from astropy.visualization import MinMaxInterval, SqrtStretch, ManualInterval, ImageNormalize
import cv2
import imgaug.augmenters as iaa


def convert_to_valid_color(image_color, clip=False, lower_clip=0.0, upper_clip=1.0, normalize=False, scaling=None,
                           simple_norm=False):
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


def rotate_image_stacks(image_stack, rotation_angle=0):
    image_stack = rotate(image_stack, -rotation_angle, resize=True, center=None,
                         order=1, mode='constant', cval=0, clip=True,
                         preserve_range=True)
    return image_stack


def make_bounding_box(source_location):
    """
    Creates a bounding box and returns it in (xmin, ymin, xmax, ymax, class_name, source location) format
    for use with the source location in pixel values
    :param class_name: Class name for the bounding box
    :return: Bounding box coordinates for COCO style annotation
    """
    # Now create box, which will be accomplished by taking int to get xmin, ymin, and int + 1 for xmax, ymax
    xmin = int(np.floor(source_location[5][0])) - 1
    ymin = int(np.floor(source_location[5][1])) - 1
    ymax = ymin + 2
    xmax = xmin + 2

    return xmin, ymin, xmax, ymax, source_location[4], source_location[5]

def augment_image_and_bboxes(image, cutouts, angle):
    bounding_boxes = []
    for cutout in cutouts:
        bounding_boxes.append((cutout[1], cutout[0], cutout[3], cutout[2]))
    #bounding_boxes = np.asarray(bounding_boxes)
    image_aug, bbox_aug = iaa.Affine(rotate=angle)(image=image, bounding_boxes=bounding_boxes)
    for index, bbox in enumerate(bbox_aug):
        cutouts[index][0] = bbox[0]
        cutouts[index][1] = bbox[1]
        cutouts[index][2] = bbox[2]
        cutouts[index][3] = bbox[3]
    return image_aug, cutouts