"""
Base Backbone
=============
Minimal base class for pluggable backbones.

A backbone maps a flat input state vector of size ``input_dim`` to a feature
vector of size ``output_dim``. Subclasses implement ``forward``.
"""

import torch.nn as nn


class BaseBackbone(nn.Module):
    """
    Base class for feature-extraction backbones.

    Args:
        input_dim: Dimension of the input state vector.
        output_dim: Dimension of the produced feature vector.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError
