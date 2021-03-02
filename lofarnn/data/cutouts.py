from typing import Union, Optional, List, Tuple, Any

import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.visualization import (
    MinMaxInterval,
    SqrtStretch,
    ManualInterval,
    ImageNormalize,
)
from astropy.visualization import PercentileInterval
from astropy.wcs.utils import skycoord_to_pixel
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from skimage.transform import rotate
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.table import Table

flatten = lambda t: [item for sublist in t for item in sublist]


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
    cutouts: Union[List[Tuple[float]], np.ndarray, None],
    proposal_boxes: Union[List[Tuple[float]], np.ndarray, None],
    angle: Optional[Union[float, int]],
    new_size: Optional[Union[int, Tuple[int]]],
    verbose: bool = False,
) -> Tuple[Any, Union[List[Tuple[float]], np.ndarray], np.ndarray]:
    bounding_boxes = []
    prop_boxes = []
    seg_boxes = []
    new_crop = int(
        np.ceil(image.shape[0])
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


def FWHM_to_sigma_for_gaussian(fwhm):
    """Given a FWHM returns the sigma of the normal distribution."""
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def extract_gaussian_parameters_from_component_catalogue(
    pandas_cat,
    wcs,
    arcsec_per_pixel=1.5,
    PA_offset_degree=90,
    maj_min_in_arcsec=True,
    peak_flux_is_in_mJy=True,
):
    # Create skycoords for the center locations of all gaussians
    c = SkyCoord(pandas_cat.RA, pandas_cat.DEC, unit="deg")

    # transform ra, decs to pixel coordinates
    if maj_min_in_arcsec:
        deg2arcsec = 1
    else:
        deg2arcsec = 3600
    if peak_flux_is_in_mJy:
        mJy2Jy = 1000
    else:
        mJy2Jy = 1
    pixel_locs = skycoord_to_pixel(c, wcs, origin=0, mode="all")
    gaussians = [
        models.Gaussian2D(
            row.Peak_flux / mJy2Jy,
            x,
            y,
            FWHM_to_sigma_for_gaussian(row.Maj * deg2arcsec / arcsec_per_pixel),
            FWHM_to_sigma_for_gaussian(row.Min * deg2arcsec / arcsec_per_pixel),
            theta=np.deg2rad(row.PA + PA_offset_degree),
        )
        for ((irow, row), x, y) in zip(
            pandas_cat.iterrows(), pixel_locs[0], pixel_locs[1]
        )
    ]
    return gaussians


def subtract_gaussians_from_data(gaussians, astropy_cutout):
    # Create indices
    yi, xi = np.indices(astropy_cutout.shape)

    model = np.zeros(astropy_cutout.shape)
    for g in gaussians:
        model += g(xi, yi)
    residual = astropy_cutout - model
    return model, residual


# Gauss_dict could be constructed as follows
"""
# Load gaussian component cat
gauss_cat = pd.read_hdf(os.environ['LOTSS_GAUSS_CATALOGUE'])
# Turn Gauss cat into dict
gauss_dict = {s:[] for s in gauss_cat['Source_Name'].values}
for s,idx in zip(gauss_cat['Source_Name'].values, gauss_cat.index):
    gauss_dict[s].append(idx)
"""


def remove_unresolved_sources_from_view(
    source_name: str,
    min_ra: float,
    max_ra: float,
    min_dec: float,
    max_dec: float,
    image: np.ndarray,
    wcs: WCS,
    gauss_catalog: str,
    component_catalog: Table,
    debug: bool = False,
):
    """Given a path to a fits file and the corresponding cutout object,
    for all sources in the cutout object marked as unresolved we will find
    the constituent gaussian components and subtract those from the fits image.
    Finally we write the image back to the fits file."""

    relevant_idxs = []

    # Load gaussian component cat
    gauss_cat = Table.read(gauss_catalog).to_pandas()
    component_catalog = component_catalog.to_pandas()
    # Turn Gauss cat into dict
    gauss_dict = {str(s, "utf-8"): [] for s in gauss_cat["Source_Name"].values}
    for s, idx in zip(gauss_cat["Source_Name"].values, gauss_cat.index):
        gauss_dict[str(s, "utf-8")].append(idx)
    # print(gauss_dict)
    # For each unresolved source
    # UNnresolved source is from a special cutout lofarnn_things stuff, have to change for here
    # Mostly need to change input, take all those that have the same Source Name areas, and keep those, subtract out
    # All others in cutout that don't have the same Source Name as the source
    box_dim = (
        (gauss_cat["RA"] >= min_ra)
        & (gauss_cat["RA"] <= max_ra)
        & (gauss_cat["DEC"] >= min_dec)
        & (gauss_cat["DEC"] <= max_dec)
    )
    # Get accompanying value added catalogue datarow
    compcat_subset = gauss_cat[box_dim]
    box_dim = (
            (component_catalog["RA"] >= min_ra)
            & (component_catalog["RA"] <= max_ra)
            & (component_catalog["DEC"] >= min_dec)
            & (component_catalog["DEC"] <= max_dec)
    )
    component_cat = component_catalog[box_dim]
    print(f"Source: {source_name} {str.encode(source_name)}")
    #print(f"Compcat: {compcat_subset.Source_Name}")
    #for _ in range(len(compcat_subset.Source_Name)):
        #print(f"Equal: {compcat_subset.Source_Name != str.encode(source_name)}")
    non_sources = compcat_subset[
        compcat_subset.Source_Name != str.encode(source_name)
    ]  # Select all those not part of source
    comp_non_sources = component_cat[
        component_cat.Source_Name != source_name
        ]  # Select all those not part of source
    comp_sources = component_cat[
        component_cat.Source_Name == source_name
        ]
    print(f"Component Sources: {len(comp_sources)}")
    print(f"Component Non Sources: {len(comp_non_sources)}")
    print(f"Gaussian Non Sources: {len(non_sources)}")
    comp_dict = {s: [] for s in comp_sources["Source_Name"].values}
    for s, idx in zip(comp_sources["Source_Name"].values, comp_sources.index):
        comp_dict[s].append(idx)
    if len(non_sources) >= 1:
        for unresolved_source in non_sources["Source_Name"]:
            # Get relevant catalogue entries
            print(unresolved_source)
            try:
                same_one = comp_dict[str(unresolved_source, "utf-8")]
            except KeyError:
                # Only want ones that are not components
                relevant_idxs.append(gauss_dict[str(unresolved_source, "utf-8")])
                pass
            #if str(unresolved_source, "utf-8") in comp_non_sources: # So Gauss either diff name, or in comp
            #    relevant_idxs.append(gauss_dict[str(unresolved_source, "utf-8")])

        # Create gaussians
        relevant_idxs = flatten(relevant_idxs)
        gaussians = extract_gaussian_parameters_from_component_catalogue(
            gauss_cat.loc[relevant_idxs], wcs
        )
        # Subtract them from the data
        model, residual = subtract_gaussians_from_data(gaussians, image)
        #image = residual

        # Debug visualization
        if True:
            norm = ImageNormalize(
                image, interval=PercentileInterval(99.0), stretch=SqrtStretch()
            )
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(image, norm=norm)
            ax[1].imshow(residual, norm=norm)
            ax[2].imshow(model, norm=norm)
            plt.savefig(f"{source_name}.png", dpi=300)
            plt.cla()
            plt.clf()
    else:
        residual = image
    return residual, image


def is_image_artifact(
    image: np.ndarray, central_size: int = 10, threshold: Union[float, int] = 1.0
) -> bool:
    """Check if the center of the image is empty, suggesting that the Gaussian is an artifact, remove"""
    img_center_h = int(image.shape[0] / 2)
    img_center_w = int(image.shape[1] / 2)
    center = image[
        img_center_h - central_size : img_center_h + central_size,
        img_center_w - central_size : img_center_w + central_size,
    ]
    if np.sum(center) <= threshold:
        return True
    else:
        return False


def get_zoomed_image(
    image: np.ndarray, rms_img: np.ndarray, wcs: WCS, threshold: float = 0.9
) -> [np.ndarray, WCS, int, Tuple[int, int]]:
    """Gets the image where 90% of the flux is, returns the min bounding box around 90% of the flux"""

    total_flux = np.sum(image)

    img_center_h = int(image.shape[0] / 2)
    img_center_w = int(image.shape[1] / 2)

    current_flux = 0.0
    central_size = 15
    print(threshold)
    # Convert threshold from mJy to Jy, same as image
    while current_flux <= threshold:
        print(f"Current Flux: {current_flux} <= {threshold}")
        central_size += 1
        if img_center_w - central_size < 0:
            # Too large, not have all the flux, so just return the original image, not the residual
            return image, wcs, None, None, rms_img
        center = image[
            img_center_h - central_size : img_center_h + central_size,
            img_center_w - central_size : img_center_w + central_size,
        ]
        current_flux = np.sum(center)

    cutout = Cutout2D(
        image,
        position=(img_center_h, img_center_w),
        size=(2 * central_size, 2 * central_size),
        wcs=wcs,
    )
    rms_cutout = Cutout2D(
        rms_img,
        position=(img_center_h, img_center_w),
        size=(2 * central_size, 2 * central_size),
        wcs=wcs,
    )

    return cutout.data, cutout.wcs, central_size, (img_center_h, img_center_w), rms_cutout.data
