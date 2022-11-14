"""CASPR embedding layer base class."""

import numpy as np
import torch
import torch.nn as nn


class CategoricalEmbedding(nn.Module):  # noqa: W0223
    """Define embedding layers to convert categorical variable values to continuous embeddings.

    Uses pytorch defined nn.Embedding layers
    The incoming data for this class has 3 dimensions - dim(1) is the number of time steps in the sequence
    when used for a seq variable
    When being used for non-seq variable - data has 2 dimensions
    """

    def __init__(self,  # noqa: R0913
                 emb_dims, emb_dropout, is_seq=False, pretrained_vecs=None, freeze_pretrained=True):
        """Initialise the emb layer class.

        Args:
            emb_dims: A list of tuple (x, y) which contains the input for the nn.Embedding layer
            emb_dropout : The dropout value for the layers applied after concatenation of all the embeddings
            is_seq = determines if this layer has been initialised for sequential or non-sequential data
            pretrained_vecs = The tensor which contains the pretrained values. For variables for which we dont have the
                vecs we initialise the nn.Embedding layer and backpropagate through them
            freeze_pretrained This boolean label determines if we freeze the pretrained embeddings and dont
              backpropagate through them
        """

        super().__init__()

        self.emb_dims = emb_dims
        self.is_seq = is_seq
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        if pretrained_vecs is not None and len(emb_dims) > 0:
            for i, v in enumerate(pretrained_vecs):
                if v is not None:
                    self.emb_layers[i] = nn.Embedding.from_pretrained(v, freeze=freeze_pretrained)
        self.num_classes = [x for x, _ in emb_dims]
        self.emb_size = np.sum([y for _, y in emb_dims], dtype=np.int32)
        self.emb_dropout_layer = nn.Dropout(emb_dropout)

    def forward(self, cat_data):
        """Run a forward pass of model over the data."""
        cat_data = cat_data.long()
        # across all rows and column i - useful for batches
        cat_inp = [emb_layer(cat_data[..., i]) for i, emb_layer in enumerate(self.emb_layers)]
        cat_inp = torch.cat(cat_inp, -1)
        cat_inp = self.emb_dropout_layer(cat_inp)
        return cat_inp
