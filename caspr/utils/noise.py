"""Noise class for generating noisy data."""

import torch


class Noise(torch.nn.Module):
    """Add different types of noise to the sequential inputs for denoising autoencoder.

    Usage:
        noise = Noise(emb_dims, gau_prob, sub_prob, shuffle_dist)
        seq_cat_noisy, seq_cont_noisy = noise(seq_cat, seq_cont)
    """

    def __init__(self, emb_dims, gau_prob=0.1, sub_prob=0.1, shuffle_dist=1):
        """Initialize Noise objects with probabilities for different noise types.

        Args:
            emb_dims (List of tuples (x, y)): Embedding dimensions where x is the vocab size and
                                              y is the embedding size for every categorical variable.
            gau_prob (float): Probability of adding gaussian noise to the continuous variables.
            sub_prob (float): Probability of substituting a categorical value with another randomly selected one.
            shuffle_dist (int): The max distance that each element will be away from its original position
                                after shuffling.
        """

        super().__init__()

        self.gau_prob = gau_prob
        self.sub_prob = sub_prob
        self.shuffle_dist = shuffle_dist
        self.vocab_sizes = [dim[0] for dim in emb_dims]

    def forward(self, seq_cat_data, seq_cont_data):
        """Run a forward pass of the module over the data to add noise."""

        return self.add_noise(seq_cat_data, seq_cont_data)

    def add_noise(self, seq_cat_data, seq_cont_data):
        """Add noise to the sequential data based on the specified probabilities.

        Args:
            seq_cat_data (Tensors): Sequential categorical data.
            seq_cont_data (Tensors): Sequential continuous data.
        """

        if self.sub_prob > 0:
            seq_cat_data = self._word_substitute(seq_cat_data)

        if self.gau_prob > 0:
            seq_cont_data = self._word_gaussian(seq_cont_data)

        if self.shuffle_dist > 0:
            seq_cat_data, seq_cont_data = self._word_shuffle(seq_cat_data, seq_cont_data)

        return seq_cat_data, seq_cont_data

    def _word_shuffle(self, seq_cat_data, seq_cont_data):
        batch_size, seq_len, _ = seq_cat_data.size()
        base = torch.arange(seq_len, dtype=torch.float).repeat(batch_size, 1)
        inc = (self.shuffle_dist+1) * torch.rand((batch_size, seq_len))
        _, sigma = (base + inc).sort(dim=1)
        return (seq_cat_data[torch.arange(batch_size).unsqueeze(1), sigma],
                seq_cont_data[torch.arange(batch_size).unsqueeze(1), sigma])

    def _word_substitute(self, x):
        keep = (torch.rand(x.size(), device=x.device) > self.sub_prob)
        x_ = x.clone()
        for i in range(len(self.vocab_sizes)):
            x_[:, :, i].random_(0, self.vocab_sizes[i])
        x_[keep] = x[keep]
        return x_

    def _word_gaussian(self, x):
        gaussian = (torch.rand(x.size(), device=x.device) < self.gau_prob)
        x_ = x.clone()
        x_ += torch.randn(x.size(), device=x.device) * gaussian
        return x_
