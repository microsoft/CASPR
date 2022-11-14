# coding: utf-8
"""CASPR mlp base class."""

import torch
import torch.nn as nn

from caspr.models.dense_bn_dropout import DenseBnDropout


class MLP(nn.Module): # noqa: W0223
    def __init__(self, # noqa: R0913
                 input_size,
                 lin_layer_sizes,
                 lin_layer_dropouts,
                 output_size,
                 use_sigmoid=False):
        """Initialize model with params."""

        super().__init__()

        self.output_size = output_size
        self.use_sigmoid = use_sigmoid

        # final linear layers just before prediction
        self.dense_bn_dropout = DenseBnDropout(
            lin_layer_sizes=lin_layer_sizes, lin_layer_dropouts=lin_layer_dropouts, input_size=input_size)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

    def forward(self, inp):
        """Run a forward pass of model over the data."""
        inp = self.dense_bn_dropout(inp)
        out = self.output_layer(inp)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out
