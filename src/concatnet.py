import torch
import torch.nn as nn
import torch.nn.functional as F

from features import Features
from classifier import Classifier


class ConcatNet(nn.Module):
    def __init__(self, num_channels: int):
        super(ConcatNet, self).__init__()
        self.rgb_features = Features(num_channels=num_channels)
        self.nir_features = Features(num_channels=num_channels)
        self.concat_cls = Classifier(num_channels=512)

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        rgb = self.rgb_features(rgb)
        nir = self.nir_features(nir)

        concatenation = torch.cat((F.relu(rgb), F.relu(nir)), dim=1)
        concatenation = concatenation.view(concatenation.size(0), -1)
        concatenation = self.concat_cls(concatenation)

        return concatenation
