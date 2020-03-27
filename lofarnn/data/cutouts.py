import numpy as np
from skimage.transform import rotate
from astropy.visualization import MinMaxInterval, SqrtStretch, ManualInterval, ImageNormalize
import cv2


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


def rotate_image_and_bboxes(image, source_locations, angle):
    """
    Rotate and transform image, and recompute bounding box for the new source location
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image

    # add ones
    for index, location in enumerate(source_locations):
        v = [location[1] + 1.5, location[0] + 1.5, 1]
        source_loc = [location[5][0], location[5][1], 1]
        # transform points
        transformed_points = np.dot(M, v)
        source_loc = np.dot(M, source_loc)
        source_locations[index][5] = (transformed_points[1], transformed_points[0])
        source_locations[index] = make_bounding_box(source_locations[index])

    return cv2.warpAffine(image, M, (nW, nH)), source_locations


def nearest_nonzero_idx(a, x, y):
    idx = np.argwhere(a)

    # If (x,y) itself is also non-zero, we want to avoid those, so delete that
    # But, if we are sure that (x,y) won't be non-zero, skip the next step
    idx = idx[~(idx == [x, y]).all(1)]

    return idx[((idx - [x, y]) ** 2).sum(1).argmin()]


def find_bounding_box(image, source_location):
    """
    Gets the closest island of values greater than 0, and then makes the bounding box the size of that island
    """
    image = image[image > 0.0]
    nearest_value = nearest_nonzero_idx(image, source_location[0], source_location[1])


import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


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
