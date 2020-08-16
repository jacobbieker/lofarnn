import numpy as np
from skimage.transform import rotate
from astropy.visualization import (
    MinMaxInterval,
    SqrtStretch,
    ManualInterval,
    ImageNormalize,
)
import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapOnImage, SegmentationMapsOnImage


def convert_to_valid_color(
    image_color,
    clip=False,
    lower_clip=0.0,
    upper_clip=1.0,
    normalize=False,
    scaling=None,
    simple_norm=False,
):
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


import matplotlib.pyplot as plt


def augment_image_and_bboxes(
    image,
    cutouts,
    proposal_boxes,
    segmentation_maps,
    segmentation_proposals,
    angle,
    new_size=None,
    crop_size=None,
    verbose=False,
):
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
    for i, sbox in enumerate(segmentation_proposals):
        seg_boxes.append(
            BoundingBox(float(sbox[0]), float(sbox[1]), float(sbox[2]), float(sbox[3]))
        )

    bbs = BoundingBoxesOnImage(bounding_boxes, shape=image.shape)
    pbbs = BoundingBoxesOnImage(prop_boxes, shape=image.shape)
    if seg_boxes:
        sbbs = BoundingBoxesOnImage(seg_boxes, shape=image.shape)
        _, sbbs = seq(image=image, bounding_boxes=sbbs)
    if segmentation_maps.any():
        segmaps = SegmentationMapsOnImage(segmentation_maps, shape=image.shape)
        _, segmaps_rescaled = seq(image=image, segmentation_maps=segmaps)
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
    if segmentation_maps.any():
        segmaps_rescaled = segmaps.resize(
            (image_rescaled.shape[0], image_rescaled.shape[1])
        )

    # Remove bounding boxes that go out of bounds
    pbbs_rescaled = pbbs_rescaled.remove_out_of_image(partly=False).clip_out_of_image()
    # But only clip source bounding boxes that are partly out of frame, so that no sources are lost
    bbs_rescaled = bbs_rescaled.remove_out_of_image(partly=False).clip_out_of_image()
    # Also remove those and clip of segmentation maps
    # Draw image before/after rescaling and with rescaled bounding boxes
    if verbose:
        image = image[:, :, :3]
        image_rescaled = image_rescaled[:, :, :3]
        image_bbs = bbs.draw_on_image(image, size=1, alpha=1)
        # image_bbs = sbbs.draw_on_image(image_bbs, size=1, alpha=1)
        # image_bbs = segmaps.draw_on_image(image_bbs)[0]
        image_rescaled_bbs = bbs_rescaled.draw_on_image(
            image_rescaled, size=1, alpha=1, color=(255, 255, 255)
        )
        # image_rescaled_bbs = sbbs_rescaled.draw_on_image(
        #    image_rescaled_bbs, size=1, alpha=1, color=(0, 255, 255)
        # )
        # image_rescaled_bbs = segmaps_rescaled.draw_on_image(image_rescaled_bbs)[0]
        plt.imshow(image_bbs)
        plt.title("Before")
        plt.show()
        # plt.savefig("before_rot.png")
        # plt.cla()
        # plt.clf()
        plt.imshow(image_rescaled_bbs)
        plt.title("After")
        # plt.savefig("after_rot.png")
        # plt.cla()
        # plt.clf()
        plt.show()
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
    if segmentation_maps.any():
        return image_rescaled, cutouts, pbs, segmaps_rescaled.get_arr(), sbs
    else:
        return image_rescaled, cutouts, pbs, np.asarray([]), sbs
