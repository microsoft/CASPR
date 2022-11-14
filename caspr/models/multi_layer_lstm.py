"""CASPR LSTM base class."""

import torch
import torch.nn as nn


class MultiLayerLSTM(nn.Module):  # noqa: W0223
    """Encapsulates the Pytorch LSTM.

    Added functionality of aggregation / concatenation in cases of
    bidirectional and multi-layered LSTM's

    It outputs the original outputs of the lstm along with an aggregated output vector
    """

    def __init__(self, input_size, hidden_size, dropout=0., num_layers=1, bidirectional=False):  # noqa: R0913
        """Initialise the pytorch LSTM layer.

        Args:
            input_size = The size of the input in the lstm. This represents the number of input features
            hidden_size = the hidden size of the lstm
            dropout = the dropout layers between the multiple layers of the lstm (works only when we use a
                multi-layered lstm)
            num_layers = num_layers of the lstm
            bidirectional = represents the type of the lstm
        """
        super().__init__()
        self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                  bidirectional=bidirectional, dropout=dropout)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # Linear Layers post LSTM
        self.lin_layer_lstm_to_dense = nn.Linear(
            self.num_directions*self.hidden_size, self.hidden_size)

    def forward(self, input_tensor, hidden_state=None):
        """Run a forward pass of model over the data."""
        batch_size = input_tensor.size()[0]
        device = input_tensor.device

        if hidden_state is not None:
            h0 = hidden_state
            c0 = torch.zeros(self.num_directions*self.num_layers, batch_size, self.hidden_size).to(device)
            output, (hn, cn) = self.lstm_layer(input_tensor, (h0, c0))
        else:
            output, (hn, cn) = self.lstm_layer(input_tensor)

        hn = hn.view(self.num_layers, self.num_directions, -
                     1, self.hidden_size)
        cn = cn.view(self.num_layers, self.num_directions, -
                     1, self.hidden_size)

        if self.num_directions > 1:
            seq_inp = self.lin_layer_lstm_to_dense(torch.cat(
                [hn[self.num_layers-1, 0], hn[self.num_layers-1, -1]], 1).unsqueeze(0))
        else:
            seq_inp = self.lin_layer_lstm_to_dense(
                hn[self.num_layers-1, 0]).unsqueeze(0)

        seq_inp = seq_inp.reshape(seq_inp.size()[1], seq_inp.size()[2])

        return output, (hn[self.num_layers-1, 0, :, :], cn[self.num_layers-1, 0, :, :]), seq_inp
