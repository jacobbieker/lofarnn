import numpy as np
from skimage.transform import rotate
from astropy.visualization import MinMaxInterval, SqrtStretch, ManualInterval, ImageNormalize


def convert_to_valid_color(image_color, clip=False, lower_clip=0.0, upper_clip=1.0, normalize=False, scaling=None, simple_norm=False):
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