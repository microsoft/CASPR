"""CASPR LSTM decoder base class."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_attention_embedding_decoder(nn.Module): # noqa: W0223
    """Simple LSTM decoder."""

    def __init__(self, # noqa: R0913
                 input_dim,
                 hidden_size,
                 output_dim,
                 num_classes,
                 num_layers=1):
        """Initialize model with params."""
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.output_dim = output_dim

        # LSTM layer
        self.lstm_layer = nn.LSTM(
            input_dim, hidden_size, num_layers, batch_first=True)

        self.linear = nn.Linear(self.hidden_size, output_dim)

        self.output = nn.ModuleList([nn.Linear(self.hidden_size, num_class) for num_class in self.num_classes])
        self.hidden = None

    def forward(self, inp, hidden):
        """Forward pass through LSTM layer.

        shape of lstm_out: [input_size, batch_size, hidden_dim]
        shape of self.hidden: (a, b), where a and b both
        have shape (num_layers, batch_size, hidden_dim).
        """
        inp = inp.view(inp.shape[0], 1, -1)
        self.hidden = hidden

        lstm_out, self.hidden = self.lstm_layer(inp, self.hidden)
        decoder_out = (torch.tanh(lstm_out[:, -1, :]))

        y_pred = self.linear(decoder_out)
        out_cont = F.relu(y_pred)
#         out_cat = self.output(decoder_out)
        out_cat = [  # across all rows and column i -  useful for batches
            output_layer(decoder_out) for i, output_layer in enumerate(self.output)
        ]
#         out_cat = torch.cat(out_cat, -1)
#         print(out_cat.shape)

        return out_cont, self.hidden, out_cont, out_cat
