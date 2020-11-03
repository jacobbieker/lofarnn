import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

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
    visualization,
    ShapleyValueSampling,
)

"""

Set of visualizations from a model to determine why those models look at what they do

"""


def score_func(o):
    return F.softmax(o, dim=1)


def convert_to_image(inputs):
    return (
        np.transpose(inputs[0].cpu().detach().numpy(), (0, 2, 3, 1)),
        inputs[1].cpu().detach().numpy().reshape((1, 12, 1)),
    )


def convert_to_image_multi(inputs):
    return (
        np.transpose(inputs[0].cpu().detach().numpy(), (0, 2, 3, 1)),
        np.transpose(inputs[1].cpu().detach().numpy(), (0, 2, 3, 1))[0],
    )


def visuaize_maps(
    model, inputs, labels, title, second_occlusion=(1, 2, 2), baselines=(0, 0)
):
    """
    Visualizes the average of the inputs, or the single input, using various different XAI approaches
    """
    single = inputs[1].ndim == 2
    model.zero_grad()
    model.eval()
    occ = Occlusion(model)
    saliency = Saliency(model)
    saliency = NoiseTunnel(saliency)
    igrad = IntegratedGradients(model)
    igrad_2 = NoiseTunnel(igrad)
    # deep_lift = DeepLift(model)
    grad_shap = ShapleyValueSampling(model)
    output = model(inputs[0], inputs[1])
    output = F.softmax(output, dim=-1).argmax(dim=1, keepdim=True)
    labels = F.softmax(labels, dim=-1).argmax(dim=1, keepdim=True)
    if np.all(labels.cpu().numpy() == 1):
        return
    if True:
        targets = labels
    else:
        targets = output
    print(targets)
    correct = targets.cpu().numpy() == labels.cpu().numpy()
    #if correct:
    #   return
    occ_out = occ.attribute(
        inputs,
        baselines=baselines,
        sliding_window_shapes=((1, 5, 5), second_occlusion),
        target=targets,
    )
    # occ_out2 = occ.attribute(inputs, sliding_window_shapes=((1,20,20), second_occlusion), strides=(8,1), target=targets)
    saliency_out = saliency.attribute(
        inputs, nt_type="smoothgrad_sq", n_samples=5, target=targets, abs=False
    )
    # igrad_out = igrad.attribute(inputs, target=targets, internal_batch_size=1)
    igrad_out = igrad_2.attribute(
        inputs,
        baselines=baselines,
        target=targets,
        n_samples=5,
        nt_type="smoothgrad_sq",
        internal_batch_size=1,
    )
    # deep_lift_out = deep_lift.attribute(inputs, target=targets)
    grad_shap_out = grad_shap.attribute(inputs, baselines=baselines, target=targets)

    if single:
        inputs = convert_to_image(inputs)
        occ_out = convert_to_image(occ_out)
        saliency_out = convert_to_image(saliency_out)
        igrad_out = convert_to_image(igrad_out)
        #grad_shap_out = convert_to_image(grad_shap_out)
    else:
        inputs = convert_to_image_multi(inputs)
        occ_out = convert_to_image_multi(occ_out)
        saliency_out = convert_to_image_multi(saliency_out)
        igrad_out = convert_to_image_multi(igrad_out)
        grad_shap_out = convert_to_image_multi(grad_shap_out)
    fig, axes = plt.subplots(2, 5)
    (fig, axes[0, 0]) = visualization.visualize_image_attr(
        occ_out[0][0],
        inputs[0][0],
        title="Original Image",
        method="original_image",
        show_colorbar=True,
        plt_fig_axis=(fig, axes[0, 0]),
        use_pyplot=False,
    )
    (fig, axes[0, 1]) = visualization.visualize_image_attr(
        occ_out[0][0],
        None,
        sign="all",
        title="Occ (5x5)",
        show_colorbar=True,
        plt_fig_axis=(fig, axes[0, 1]),
        use_pyplot=False,
    )
    (fig, axes[0, 2]) = visualization.visualize_image_attr(
        saliency_out[0][0],
        None,
        sign="all",
        title="Saliency",
        show_colorbar=True,
        plt_fig_axis=(fig, axes[0, 2]),
        use_pyplot=False,
    )
    (fig, axes[0, 3]) = visualization.visualize_image_attr(
        igrad_out[0][0],
        None,
        sign="all",
        title="Integrated Grad",
        show_colorbar=True,
        plt_fig_axis=(fig, axes[0, 3]),
        use_pyplot=False,
    )
    (fig, axes[0, 4]) = visualization.visualize_image_attr(
        grad_shap_out[0],
        None,
        title="GradSHAP",
        show_colorbar=True,
        plt_fig_axis=(fig, axes[0, 4]),
        use_pyplot=False,
    )
    ##### Second Input Labels #########################################################################################
    (fig, axes[1, 0]) = visualization.visualize_image_attr(
        occ_out[1],
        inputs[1],
        title="Original Aux",
        method="original_image",
        show_colorbar=True,
        plt_fig_axis=(fig, axes[1, 0]),
        use_pyplot=False,
    )
    (fig, axes[1, 1]) = visualization.visualize_image_attr(
        occ_out[1],
        None,
        sign="all",
        title="Occ (1x1)",
        show_colorbar=True,
        plt_fig_axis=(fig, axes[1, 1]),
        use_pyplot=False,
    )
    (fig, axes[1, 2]) = visualization.visualize_image_attr(
        saliency_out[1],
        None,
        sign="all",
        title="Saliency",
        show_colorbar=True,
        plt_fig_axis=(fig, axes[1, 2]),
        use_pyplot=False,
    )
    (fig, axes[1, 3]) = visualization.visualize_image_attr(
        igrad_out[1],
        None,
        sign="all",
        title="Integrated Grad",
        show_colorbar=True,
        plt_fig_axis=(fig, axes[1, 3]),
        use_pyplot=False,
    )
    (fig, axes[1, 4]) = visualization.visualize_image_attr(
        grad_shap_out[1],
        None,
        title="GradSHAP",
        show_colorbar=True,
        plt_fig_axis=(fig, axes[1, 4]),
        use_pyplot=False,
    )

    fig.suptitle(
        title + f" Label: {labels.cpu().numpy()} Pred: {targets.cpu().numpy()}"
    )
    plt.savefig(
        f"{title}_{'single' if single else 'multi'}_{'Failed' if correct else 'Success'}_baseline{baselines[0]}.png",
        dpi=300,
    )
    plt.clf()
    plt.cla()
