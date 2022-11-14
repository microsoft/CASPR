import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from caspr.utils.preprocess import get_nonempty_tensors


def cluster_initialize(model, dataloader, device):
    """Initialize cluster.

    Args:
        model (nn.Module): # noqa: W0223 Pretrained encoder-decoder model
        dataloader (DataLoader): Data loader that provides an iterable over the given dataset
        device ('cpu' or 'cuda'): Describes the machine on which the code is running
    """
    kmeans = KMeans(model.cluster_number, n_init=20)
    model.train()
    encoder_embs = []
    labels = []
    # form initial cluster centres
    for _, y, seq_cat_x, seq_cont_x, non_seq_cat_x, non_seq_cont_x in dataloader:
        seq_cat_x = seq_cat_x.to(device)
        seq_cont_x = seq_cont_x.to(device)
        non_seq_cat_x = non_seq_cat_x.to(device)
        non_seq_cont_x = non_seq_cont_x.to(device)

        data = (seq_cat_x, seq_cont_x, non_seq_cat_x, non_seq_cont_x)
        nonempty_tensors, nonempty_idx = get_nonempty_tensors(data)
        encoder_embs.append(model.enc(*nonempty_tensors, nonempty_idx).detach().cpu())

        labels.append(y)

    labels = torch.cat(labels).long()

    predicted = kmeans.fit_predict(torch.cat(encoder_embs).numpy())
    predicted_tensor = torch.tensor(np.copy(predicted), dtype=torch.long)
    _, accuracy = cluster_accuracy(predicted, labels.cpu().numpy())
    print('Initial Cluster Acc: ', accuracy)
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True).to(device)
    with torch.no_grad():
        # initialise the cluster centers
        model.state_dict()['assignment.cluster_centers'].copy_(cluster_centers)
    return predicted_tensor


def cluster_accuracy(y_true, y_predicted, cluster_number=None):
    """Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to determine reassignments.

    Args:
        y_true (List of int): list of true cluster numbers, an integer array 0-indexed
        y_predicted (List of int): list of predicted cluster numbers, an integer array 0-indexed
        cluster_number (int): number of clusters, if None then calculated from input
    Return:
        reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = max(y_predicted.max(), y_true.max()) + 1  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy


def cluster_predict(model, dataloader, device):
    """Predict the cluster centers for the given input data.

    Args:
        model (nn.Module): # noqa: W0223 Pretrained encoder-decoder model
        dataloader (DataLoader): Data loader that provides an iterable over the given dataset
        device ('cpu' or 'cuda'): Describes the machine on which the code is running
    """
    features = []
    labels = []
    for _, y, seq_cat_x, seq_cont_x, non_seq_cat_x, non_seq_cont_x in dataloader:
        seq_cat_x = seq_cat_x.to(device)
        seq_cont_x = seq_cont_x.to(device)
        non_seq_cat_x = non_seq_cat_x.to(device)
        non_seq_cont_x = non_seq_cont_x.to(device)

        data = (seq_cat_x, seq_cont_x, non_seq_cat_x, non_seq_cont_x)
        nonempty_tensors, nonempty_idx = get_nonempty_tensors(data)
        features.append(model(*nonempty_tensors, nonempty_idx).detach().cpu())

        labels.append(y)

    return torch.cat(features).max(1)[1], torch.cat(labels).long()
