"""CASPR base dense layer class."""

import torch.nn as nn
import torch.nn.functional as F


class DenseBnDropout(nn.Module):  # noqa: W0223
    """Dense Layers w/ dropout and batch-normalization.

    A module comprising of a sequential structure of [Linear -> Batch Normalisation -> Dropout]
        used for multiple iterations through it
    When the input is a 3D tensor - batch_size x seq_len x features
    When the input is a 2D tensor - batch_size x features
    """

    def __init__(self, lin_layer_sizes, lin_layer_dropouts, input_size):
        """Initiliasing the layers.

        Args:
            lin_layer_sizes (list) = sizes of the linear layers being using across multiple iterations
            lin_layer_dropouts (list) = values of the dropout layers across multiple iterations
            input_size (integer) = size of the input tensor - batch_size x 'input_size' x seq_len
        """

        super().__init__()
        first_lin_layer = nn.Linear(input_size, lin_layer_sizes[0])
        self.lin_layers = nn.ModuleList([first_lin_layer] +
                                        [nn.Linear(lin_layer_sizes[i],
                                                   lin_layer_sizes[i + 1])
                                         for i in range(len(lin_layer_sizes) - 1)])
        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        self.dropout_layers = nn.ModuleList([nn.Dropout(p) for p in lin_layer_dropouts])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

    def forward(self, input_tensor):
        """Run a forward pass of model over the data."""
        is_seq = input_tensor.ndim == 3

        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.dropout_layers, self.bn_layers):
            input_tensor = F.relu(lin_layer(input_tensor))
            if is_seq:
                # permute to adjust for the BN internal structure
                input_tensor = input_tensor.permute(0, 2, 1)

            input_tensor = bn_layer(input_tensor)

            if is_seq:
                # permute back to maintain the original structure required for linear layer
                input_tensor = input_tensor.permute(0, 2, 1)

            input_tensor = dropout_layer(input_tensor)

        output_tensor = input_tensor
        return output_tensor
