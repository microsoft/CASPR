"""CNN based layer base class."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAggregation(nn.Module): # noqa: W0223
    """Initialise a CNN based layer that reduces the size of our input.

    It treates the sequential input like an image with a single channel and performs learned aggregation
    """

    def __init__(self, kernel_size=(3, 3), stride=(2, 2), max_pool_size=(2, 2), dropout_size=0.):
        """Initiliase the cnn layers.

        Args:
            kernel_size : Tuple which determines the size of the cnn kernel
            stride : Tuple which determines the size of the strides in the x and y direction
            max_pool_size = Tuple which determines the size of the max_pooling function
            dropout_size = Value of dropout added after entire processing
        """
        super().__init__()
        self.in_channels = 1
        self.out_channels = 1

        self.conv_layer = nn.Conv2d(in_channels=self.in_channels,
                                    out_channels=self.out_channels, kernel_size=kernel_size, stride=stride)
        self.max_pool = nn.MaxPool2d(max_pool_size)
        self.conv_dropout = nn.Dropout(dropout_size)

    def forward(self, input_tensor):
        """Run a forward pass of model over the data."""

        # The CNN by default accepts the input as (batch_size, in_channels, height_img, width_img).
        # We treat the sequential input as an image but we need an additional dimension to correspond to in_channels
        # Therefore we need to unsqueeze a dimension out here

        input_tensor = torch.unsqueeze(input_tensor, 1)

        input_tensor = F.tanh(self.conv_layer(input_tensor))
        input_tensor = self.max_pool(input_tensor)

        # The CNN by default outputs as (batch_size, out_channels, height_img, width_img).
        # We need to squeeze away the dimension we had added earlier to remain consistent

        input_tensor = input_tensor.squeeze(1)
        output_tensor = self.conv_dropout(input_tensor)

        return output_tensor
