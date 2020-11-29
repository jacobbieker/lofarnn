import torch
import torch.nn as nn
import torch.nn.functional as F

from lofarnn.models.base.resnet import ResNet, Bottleneck

"""

Taken from https://discuss.pytorch.org/t/concatenating-observations-that-include-image-pose-and-sensor-readings/41084

This is for having a CNN look at the radio data and generate feature maps, and then in the last step of the classifier,
goes through and adds in the optical source features, like distance from radio center, multispectral magnitudes, etc.

"""


class RadioSingleSourceModel(nn.Module):
    """
    Model for classifying whether a single optical source is the radio source's source
    """

    def __init__(self, num_image_layers, len_aux_data, config):
        super(RadioSingleSourceModel, self).__init__()
        self.config = config

        # Image net
        self.cnn = ResNet(
            num_image_layers,
            Bottleneck,
            [3, 4, 23, 3],
            num_classes=2,
            zero_init_residual=False,
            groups=32,
            width_per_group=8,
            keep_fc=True,
        )
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, self.config["fc_out"])

        # Source net
        self.source1 = nn.Linear(len_aux_data, self.config["fc_final"])
        self.source2 = nn.Linear(self.config["fc_final"], self.config["fc_out"])

        # Combined
        self.fc1 = nn.Linear(
            self.config["fc_out"] + self.config["fc_out"], self.config["fc_final"]
        )
        self.fc2 = nn.Linear(self.config["fc_final"], 2)

    def forward(self, image, data):
        x1 = self.cnn(image)
        if self.config["act"] == "leaky":
            x2 = F.leaky_relu(self.source1(data))
        elif self.config["act"] == "elu":
            x2 = F.elu(self.source1(data))
        else:
            x2 = F.relu(self.source1(data))
        x2 = self.source2(x2)

        # Combine them
        x = torch.cat((x1, x2), dim=1)
        if self.config["act"] == "leaky":
            x = F.leaky_relu(self.fc1(x))
        elif self.config["act"] == "elu":
            x = F.elu(self.fc1(x))
        else:
            x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RadioMultiSourceModel(nn.Module):
    """
    Model for classifying which of a set number of sources is the optical source
    """

    def __init__(self, num_image_layers, num_sources, config):
        super(RadioMultiSourceModel, self).__init__()
        self.config = config

        # Image net - ResNet 101 32x8d
        self.cnn = ResNet(
            num_image_layers,
            Bottleneck,
            [3, 4, 23, 3],
            num_classes=num_sources,
            zero_init_residual=False,
            groups=32,
            width_per_group=8,
            keep_fc=True,
        )
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, self.config["fc_out"])

        # Source net - smaller ResNet-50 to hold aux-data x num_sources 'image' to classify which of those souces is the source
        self.cnn2 = ResNet(
            num_image_layers,
            Bottleneck,
            [3, 4, 6, 3],
            num_classes=num_sources,
            keep_fc=True,
        )
        self.cnn2.fc = nn.Linear(self.cnn2.fc.in_features, self.config["fc_out"])

        # Combined
        self.fc1 = nn.Linear(
            self.config["fc_out"] + self.config["fc_out"], self.config["fc_final"]
        )
        self.fc2 = nn.Linear(self.config["fc_final"], num_sources)

    def forward(self, image, data):
        x1 = self.cnn(image)
        x2 = self.cnn2(data)

        # Combine them
        x = torch.cat((x1, x2), dim=1)
        if self.config["act"] == "leaky":
            x = F.leaky_relu(self.fc1(x))
        elif self.config["act"] == "elu":
            x = F.elu(self.fc1(x))
        else:
            x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RadioMultiSourceEnsembleModel(nn.Module):
    """
    Model for classifying which of a set number of sources is the optical source
    Uses an Ensemble of multiple networks per input to give the output
    """

    def __init__(self, num_image_layers, num_sources, config):
        super(RadioMultiSourceEnsembleModel, self).__init__()
        self.config = config

        # Image net - ResNet 101 32x8d
        self.cnn = ResNet(
            num_image_layers,
            Bottleneck,
            [3, 4, 23, 3],
            num_classes=num_sources,
            zero_init_residual=False,
            groups=32,
            width_per_group=8,
            keep_fc=True,
        )
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, self.config["fc_out"])

        # Source net - ResNet 101 32x8d
        self.cnn2 = ResNet(
            num_image_layers,
            Bottleneck,
            [3, 4, 23, 3],
            num_classes=num_sources,
            zero_init_residual=False,
            groups=32,
            width_per_group=8,
            keep_fc=True,
        )
        self.cnn2.fc = nn.Linear(self.cnn2.fc.in_features, self.config["fc_out"])

        # Combined
        self.fc1 = nn.Linear(
            self.config["fc_out"] + self.config["fc_out"], self.config["fc_final"]
        )
        self.fc2 = nn.Linear(self.config["fc_final"], num_sources)

    def forward(self, image, data):
        x1 = self.cnn(image)
        x2 = self.cnn2(data)

        # Combine them
        x = torch.cat((x1, x2), dim=1)
        if self.config["act"] == "leaky":
            x = F.leaky_relu(self.fc1(x))
        elif self.config["act"] == "elu":
            x = F.elu(self.fc1(x))
        else:
            x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def f1_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, is_training=False
) -> torch.Tensor:
    """Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


class F1_Loss(nn.Module):
    """Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    """

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred.item())).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean().item()
