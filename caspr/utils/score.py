import logging

import numpy as np
import torch

from caspr.data.common_dataset import id_collate
from caspr.models.factory import LSTM, TRANSFORMER
from caspr.utils.preprocess import get_nonempty_tensors

logger = logging.getLogger(__name__)

def run_autoencoder_score(autoenc, dataloader_test, device):

    embeddings = []
    tgt_ids = []

    for tgt_id, _, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data in dataloader_test:

        data = [seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data]
        if isinstance(autoenc, torch.nn.Module):
            data = [d.to(device) for d in data]

        nonempty_tensors, nonempty_idx = get_nonempty_tensors(data)

        tgt_ids.append(tgt_id)

        if get_architecture(autoenc) == TRANSFORMER:
            emb, _, _ = autoenc.unified_encoder(*nonempty_tensors, nonempty_idx)
            # Concatenate across timesteps
            emb = emb.reshape(emb.shape[0], -1)
            embeddings.append(emb.detach().cpu() if isinstance(emb, torch.Tensor) else emb)

        elif get_architecture(autoenc) == LSTM:
            _, (hn, _) = autoenc.unified_encoder(*nonempty_tensors, nonempty_idx)
            embeddings.append(hn.detach().cpu() if isinstance(hn, torch.Tensor) else hn)

    tgt_ids = np.concatenate(tgt_ids, axis=0)
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings_with_id = np.hstack((tgt_ids, embeddings))

    return embeddings_with_id

def score(dataset_test, autoenc, device, batch_size=1024):
    autoenc.eval()
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, collate_fn=id_collate)
    
    logger.info("Performing inference on given dataset")
    embeddings = run_autoencoder_score(autoenc, test_loader, device)
    return embeddings

def get_architecture(model):
    return model.__class__.__name__ if isinstance(model, torch.nn.Module) else model.model_type
