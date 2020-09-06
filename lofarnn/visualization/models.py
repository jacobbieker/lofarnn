import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from captum.attr import (
    Occlusion,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    Saliency,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    visualization
)

"""

Set of visualizations from a model to determine why those models look at what they do

"""


def score_func(o):
    return F.softmax(o, dim=1)


def visuaize_maps(model, inputs, labels, title, second_occlusion=(2,2)):
    """
    Visualizes the average of the inputs, or the single input, using various different XAI approaches
    """
    model.zero_grad()
    model.eval()

    occ = Occlusion(model)
    saliency = Saliency(model)
    igrad = IntegratedGradients(model)
    deep_lift = DeepLift(model)
    grad_shap = GradientShap(model)
    targets = model(inputs[0], inputs[1])
    occ_out = occ.attribute(inputs, sliding_window_shapes=((5,5), second_occlusion), target=labels)
    saliency_out = saliency.attribute(inputs, target=labels, abs=False)
    igrad_out = igrad.attribute(inputs, target=labels)
    deep_lift_out = deep_lift.attribute(inputs, target=labels)
    #grad_shap_out = grad_shap.attribute(inputs, baselines=torch.randn(20,3,200,200), target=labels)

    inputs = (np.transpose(inputs[0].cpu().detach().numpy(), (1, 2, 0)), np.transpose(inputs[1].cpu().detach().numpy(), (1, 2, 0)))
    fig, axes = plt.subplots(2, 5)

    (fig, axes[0,0]) = visualization.visualize_image_attr(occ_out[0], inputs[0], title="Original Image", method="original_image", plt_fig_axis=(fig, axes[0,0]), use_pyplot=False)
    (fig, axes[0,1]) = visualization.visualize_image_attr(occ_out[0], inputs[0], title="Occlusion (5x5)", plt_fig_axis=(fig, axes[0,1]), use_pyplot=False)
    (fig, axes[0,2]) = visualization.visualize_image_attr(saliency_out[0], inputs[0], title="Saliency", plt_fig_axis=(fig, axes[0,2]), use_pyplot=False)
    (fig, axes[0,3]) = visualization.visualize_image_attr(igrad_out[0], inputs[0], title="Integrated Grad", plt_fig_axis=(fig, axes[0,3]), use_pyplot=False)
    (fig, axes[0,4]) = visualization.visualize_image_attr(deep_lift_out[0], inputs[0], title="DeepLIFT", plt_fig_axis=(fig, axes[0,4]), use_pyplot=False)
    ##### Second Input Labels ############################################################################################
    (fig, axes[1,0]) = visualization.visualize_image_attr(occ_out[1], inputs[1], title="Original Image", method="original_image", plt_fig_axis=(fig, axes[1,0]), use_pyplot=False)
    (fig, axes[1,1]) = visualization.visualize_image_attr(occ_out[1], inputs[1], title="Occlusion (2x2)", plt_fig_axis=(fig, axes[1,1]), use_pyplot=False)
    (fig, axes[1,2]) = visualization.visualize_image_attr(saliency_out[1], inputs[1], title="Saliency", plt_fig_axis=(fig, axes[1,2]), use_pyplot=False)
    (fig, axes[1,3]) = visualization.visualize_image_attr(igrad_out[1], inputs[1], title="Integrated Grad", plt_fig_axis=(fig, axes[1,3]), use_pyplot=False)
    (fig, axes[1,4]) = visualization.visualize_image_attr(deep_lift_out[1], inputs[1], title="DeepLIFT", plt_fig_axis=(fig, axes[1,4]), use_pyplot=False)

    fig.title(title)
    fig.show()
