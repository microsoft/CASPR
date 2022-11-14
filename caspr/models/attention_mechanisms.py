# coding: utf-8
"""Attention mechanisms base class."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module): # noqa: W0223
    def __init__(self, hid_dim, n_heads, dropout):
        """Initialize model with params."""
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([self.head_dim])))

    def forward(self, query, key, value, mask=None):
        """Run a forward pass of model over the data."""
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class MultiHeadAttentionLSTMWrapper(nn.Module): # noqa: W0223
    def __init__(self, n_head, d_model, dropout=0.1):
        """Initialize model with params."""
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.multi_head_attn = MultiHeadAttentionLayer(hid_dim=d_model, n_heads=n_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """Run a forward pass of model over the data."""
        _q, _ = self.multi_head_attn(q, k, v, mask=mask)
        # dropout, residual connection and layer norm
        q = self.self_attn_layer_norm(q + self.dropout(_q))

        context_vector = torch.sum(q, 1)
        return context_vector


class BahdanauAttention(nn.Module): # noqa: W0223
    def __init__(self, hidden_size, num_directions=1):
        """Initialize model with params."""

        super().__init__()
        self.num_directions = num_directions
        self.hidden_size = hidden_size
        self.fc_encoder = nn.Linear(self.num_directions*self.hidden_size, self.hidden_size, bias=False)
        self.attnHidden = nn.Linear(self.hidden_size, 1)

    def forward(self, enc_outputs):
        """Run a forward pass of model over the data."""
        tempX = torch.tanh(self.fc_encoder(enc_outputs))

        alignment_scores = self.attnHidden(tempX)

        attn_weights = F.softmax(alignment_scores, dim=1)
        attn_weights = attn_weights.permute(0, 2, 1)

        context_vector = torch.bmm(attn_weights, enc_outputs)

        return context_vector
