import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits

from analysis.plot_data_stats import get_lotss_objects
from lofarnn.utils.fits import determine_visible_catalogue_source_and_separation

"""

Set of baseline models to compare the other models against, includes:

1. Model which only chooses the closest point
2. Model that chooses the flux-weighted closest point, using the inverse of the total flux
3. Model that chooses the flux-weighted closest point, based on the combined flux of W1 and i-band

All models are designed to work with the multisource setup, to speed up the process

"""


def closest_point_model(data):
    return torch.from_numpy(np.array([[1]]))


pan_wise_catalog = fits.open("/home/jacob/combined_panstarr_allwise_flux.fits", memmap=True)
pan_wise_catalog = pan_wise_catalog[1].data
vac_catalog = get_lotss_objects("/run/media/jacob/SSD_Backup/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2_restframe.fits")

def flux_weighted_model(names):
    soln = []
    for name in names:
        source = vac_catalog[vac_catalog["Source_Name"] == name]
        # All optical sources in 150 arcsecond radius of the point
        (
            objects,
            distances,
            angles,
            source_coords,
            sky_coords,
        ) = determine_visible_catalogue_source_and_separation(
            source["RA"], source["DEC"], 150.0 / 3600, pan_wise_catalog
        )
        # Sort from closest to farthest distance
        idx = np.argsort(distances)
        objects = objects[idx]
        distances = distances[idx]
        flux_layers = ["iFApFlux",
                       "w1Flux",
                       "gFApFlux",
                       "rFApFlux",
                       "zFApFlux",
                       "yFApFlux",
                       "w2Flux",
                       "w3Flux",
                       "w4Flux", ]
        flux = 0
        opticals = []
        for j, obj in enumerate(objects):
            for layer in flux_layers:
                value = np.nan_to_num(obj[layer])
                flux = np.nansum(flux, value)
            weighted = distances[j] * (1/flux+1e-7) # Larger flux = smaller value = chosen first
            opticals.append(weighted) # In same order as distance, so should be the same
        soln.append(np.argmin(opticals))

    # Return the smallest value, so smallest distance * 1/total flux
    return torch.from_numpy(np.array(soln))


class ClosestPointModel(nn.Module):

    def __init__(self, num_image_layers, num_sources, config):
        super(ClosestPointModel, self).__init__()

    def forward(self, image, data):
        # Return the label of 1, as its the closest point in the non-shuffled multi
        result = np.array([[1]])
        return torch.from_numpy(result)


class FluxWeightedModel(nn.Module):

    def __init__(self):
        super(FluxWeightedModel, self).__init__()

    def forward(self, image, data):

        # Sum up the last 9 elements as the total flux
        fluxes = data[:, 3:]
        distances = data[:,0]
        total_fluxes = np.sum(fluxes, axis=1)
        values = distances/total_fluxes
        # Remove first empty one
        values = values[1:]
        # Return the smallest value, so smallest distance * 1/total flux
        return torch.from_numpy(np.argmin(values))
