"""CASPR transformer base class."""

import torch
import torch.nn as nn

from caspr.models.attention_mechanisms import MultiHeadAttentionLayer


class TransformerEncoderLayer(nn.Module): # noqa: W0223  # noqa: W0223
    """TransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        hid_dim: the hidden size of the encoder
        n_heads: the number of heads in the multi-head attention layers
        pf_dim: the dimension of the feedforward network model
        dropout: the dropout value
        device: the device on which the model is running
    """

    def __init__(self,  # noqa: R0913
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        """Initialize model with params."""

        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """Run a forward pass of model over the data."""

        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class TransformerEncoder(nn.Module): # noqa: W0223  # noqa: W0223
    """TransformerEncoder is a stack of N encoder layers.

    Args:
        hid_dim: the hidden size of the encoder.
        n_layers: the number of sub-encoder-layers in the encoder
        n_heads: the number of heads in the multi-head attention layers
        pf_dim: the dimension of the feedforward network model
        dropout: the dropout value
        device: the device on which the model is running
        max_length: the maximum length of the input sequence
    """

    def __init__(self,  # noqa: R0913
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=100):
        """Initialize model with params."""
        super().__init__()

        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([TransformerEncoderLayer(hid_dim,
                                                             n_heads,
                                                             pf_dim,
                                                             dropout)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([hid_dim])))

    def _make_src_mask(self, batch_size, src_len, device):

        src_mask = torch.ones((batch_size, 1, 1, src_len), device=device).bool()

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def forward(self, src):
        """Run a forward pass of model over the data."""

        # src = [batch size, src len, hid_dim]

        batch_size = src.shape[0]
        src_len = src.shape[1]
        device = src.device

        src_mask = self._make_src_mask(batch_size, src_len, device)

        # src_mask = [batch size, src len]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # pos = [batch size, src len]

        src = self.dropout(src * self.scale + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        return src, src_mask


class PositionwiseFeedforwardLayer(nn.Module): # noqa: W0223
    """Fully connected feed-forward network consisting of two linear transformations with a ReLU activation in between.

    Args:
        hid_dim: the hidden size of the encoder
        pf_dim: the dimension of the feedforward network model
        dropout: the dropout value
    """

    def __init__(self, hid_dim, pf_dim, dropout):
        """Initialize model with params."""
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Run a forward pass of model over the data."""

        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class TransformerDecoderLayer(nn.Module): # noqa: W0223
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    Args:
        hid_dim: the hidden size of the encoder
        n_heads: the number of heads in the multi-head attention layers
        pf_dim: the dimension of the feedforward network model
        dropout: the dropout value
        device: the device on which the model is running
    """

    def __init__(self, # noqa: R0913
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        """Initialize model with params."""
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """Run a forward pass of model over the data."""

        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention


class TransformerDecoder(nn.Module): # noqa: W0223
    """TransformerDecoder is a stack of N decoder layers.

    Args:
        hid_dim: the hidden size of the decoder
        n_layers: the number of sub-decoder-layers in the decoder
        n_heads: the number of heads in the multi-head attention layers
        pf_dim: the dimension of the feedforward network model
        dropout: the dropout value
        pos_embedding: learned positional encoding added to the input embedding
        device: the device on which the model is running
    """

    def __init__(self, # noqa: R0913
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 pos_embedding):
        """Initialize model with params."""
        super().__init__()

        self.pos_embedding = pos_embedding

        self.layers = nn.ModuleList([TransformerDecoderLayer(hid_dim,
                                                             n_heads,
                                                             pf_dim,
                                                             dropout)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([hid_dim])))

    def _make_trg_mask(self, batch_size, trg_len, device):

        trg_mask = torch.tril(torch.ones((batch_size, 1, trg_len, trg_len), device=device)).bool()

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, trg, enc_src, src_mask):
        """Run a forward pass of model over the data."""

        # trg = [batch size, trg len, hid_dim]
        # enc_src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        device = trg.device

        trg_mask = self._make_trg_mask(batch_size, trg_len, device)

        # trg_mask = [batch size, trg len]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # pos = [batch size, trg len]

        trg = self.dropout(trg * self.scale + self.pos_embedding(pos))

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention
