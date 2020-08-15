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
    def __init__(self, num_image_layers, len_aux_data):
        super(RadioSingleSourceModel, self).__init__()

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
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)

        # Source net
        self.source1 = nn.Linear(len_aux_data, 64)
        self.source2 = nn.Linear(64, 128)

        # Combined
        self.fc1 = nn.Linear(128 + 128, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, image, data):
        x1 = self.cnn(image)
        x2 = data
        x2 = F.relu(self.source1(x2))
        x2 = self.source2(x2)

        # Combine them
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RadioMultiSourceModel(nn.Module):
    def __init__(self, num_image_layers, num_sources):
        super(RadioMultiSourceModel, self).__init__()

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
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)

        # Source net - smaller ResNet-50 to hold aux-data x num_sources 'image' to classify which of those souces is the source
        self.cnn2 = ResNet(
            num_image_layers,
            Bottleneck,
            [3, 4, 6, 3],
            num_classes=num_sources,
            keep_fc=True,
        )
        self.cnn2.fc = nn.Linear(self.cnn2.fc.in_features, 128)

        # Combined
        self.fc1 = nn.Linear(128 + 128, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, image, data):
        x1 = self.cnn(image)
        x2 = self.cnn2(data)

        # Combine them
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# So 10 mangitudes + distance  = 11 total, could include any more optical source data?
model = RadioSingleSourceModel(1, 11)

batch_size = 2
image = torch.randn(batch_size, 3, 299, 299)
data = torch.randn(batch_size, 10)

output = model(image, data)
