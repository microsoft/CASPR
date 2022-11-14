"""CASPR deep embedding clustering class."""

import torch
import torch.nn as nn
from torch.nn import Parameter

from caspr.utils.preprocess import get_nonempty_tensors


class ClusterAssignment(nn.Module):  # noqa: W0223
    def __init__(self,
                 cluster_number,
                 embedding_dimension,
                 alpha=1.0,
                 cluster_centers=None):
        """Handle the soft assignment.

        For a description see in 3.1.1. in Xie/Girshick/Farhadi, where the Student's t-distribution
        is used to measure similarity between feature vector and each cluster centroid.

        Args:
            cluster_number (int): number of clusters
            embedding_dimension (int): embedding dimension of feature vectors
            alpha (float): parameter representing the degrees of freedom in the t-distribution, default 1.0
            cluster_centers (tensors): clusters centers to initialise, if None then use Xavier uniform
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number,
                self.embedding_dimension,
                dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch):
        """Run a forward pass of model over the data.

        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments for each cluster.

        Args:
            batch: FloatTensor of [batch size, embedding dimension]

        Return:
            FloatTensor of [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class DEC(nn.Module):  # noqa: W0223
    def __init__(self,
                 cluster_number,
                 hidden_dimension,
                 enc,
                 alpha=1):
        """Initialize the parts of DEC algorithm.

        as described in Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        Args:
            cluster_number (int): number of clusters
            hidden_dimension (int): hidden dimension, output of the encoder
            enc (nn.Module): # noqa: W0223 encoder to use
            alpha (float): parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super().__init__()
        self.enc = enc
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(cluster_number, self.hidden_dimension, alpha)

    def forward(self, *args):
        """Compute the cluster assignment.

        Using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        Args:
            batch: FloatTensor of [batch size, embedding dimension]

        Return:
            FloatTensor of [batch size, number of clusters]
        """
        return self.assignment(self.enc(*args))

    def run(self,  # noqa : R0913
            y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data, criterion): # noqa : W0613
        data = (seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data)
        nonempty_tensors, nonempty_idx = get_nonempty_tensors(data)
        output = self(*nonempty_tensors, nonempty_idx)
        target = _target_distribution(output).detach()
        loss = criterion(output.log(), target) / output.shape[0]
        return output, loss


def _target_distribution(batch):
    """Compute the target distribution p_ij, given the batch (q_ij).

    3.1.3 Equation 3 of Xie/Girshick/Farhadi; this used the KL-divergence loss function.

    Args:
        batch: FloatTensor of [batch size, number of clusters]

    Return:
        FloatTensor of [batch size, number of clusters]
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
