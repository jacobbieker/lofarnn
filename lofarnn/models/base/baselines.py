import numpy as np
import torch
import torch.nn as nn

"""

Set of baseline models to compare the other models against, includes:

1. Model which only chooses the closest point
2. Model that chooses the flux-weighted closest point, using the inverse of the total flux
3. Model that chooses the flux-weighted closest point, based on the combined flux of W1 and i-band

All models are designed to work with the multisource setup, to speed up the process

"""


def closest_point_model(data):
    return torch.from_numpy(np.array([[1]]))


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
        distances = data[:, 0]
        total_fluxes = np.sum(fluxes, axis=1)
        values = distances / total_fluxes
        # Remove first empty one
        values = values[1:]
        # Return the smallest value, so smallest distance * 1/total flux
        return torch.from_numpy(np.argmin(values))
