import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from petastorm.pytorch import BatchedDataLoader

from caspr.data.load import transform_and_load
from caspr.models.model_wrapper import LSTMAutoencoder, TransformerAutoEncoder
from caspr.utils.preprocess import get_nonempty_tensors
from caspr.utils.spark.preprocess import remove_underscore_in_seq_col_name_list

PS_HDFS_DRIVER = 'libhdfs3'
# lower overhead, alternative is 'process'
PS_WORKER_TYPE = 'thread'
# assuming the training relies on SSD backed dbfs:/ml, Petastorm's caching can be disabled
PS_CACHE_TYPE = None

def get_default_parallelism():
    try:
        return sc.defaultParallelism
    except NameError as _:
        # Spark Context not initialized (sc)
        return os.cpu_count()


def run_autoencoder_score_peta(autoenc, steps_per_epoch, train_dataloader_iter, device, tgt_id_col, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, time_steps):
    
    embeddings = []
    tgt_ids = []

    for _ in range(steps_per_epoch):
        pd_batch = next(train_dataloader_iter)
        tgt_id, _, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data = transform_and_load(
            pd_batch, device, remove_underscore_in_seq_col_name_list(tgt_id_col), remove_underscore_in_seq_col_name_list(seq_cols), remove_underscore_in_seq_col_name_list(non_seq_cols), output_col, remove_underscore_in_seq_col_name_list(cat_cols), remove_underscore_in_seq_col_name_list(cont_cols), time_steps)

        data = (seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data)
        nonempty_tensors, nonempty_idx = get_nonempty_tensors(data)

        tgt_ids.append(tgt_id)

        if isinstance(autoenc, TransformerAutoEncoder):
            emb, _, _ = autoenc.unified_encoder(*nonempty_tensors, nonempty_idx)
            # Concatenate across timesteps
            emb = emb.view(emb.shape[0], -1)
            embeddings.append(emb.detach().cpu())

        elif isinstance(autoenc, LSTMAutoencoder):
            _, (hn, _) = autoenc.unified_encoder(*nonempty_tensors, nonempty_idx)
            embeddings.append(hn.detach().cpu())

    tgt_ids = pd.DataFrame(np.concatenate(tgt_ids, axis=0))
    tgt_ids.columns = tgt_id_col
    embeddings = pd.DataFrame(np.concatenate(embeddings, axis=0))
    # embeddings_with_id = np.hstack((tgt_ids, embeddings))
    embeddings_with_id = pd.concat([tgt_ids, embeddings], axis=1)
    return embeddings_with_id


def score_peta(converter_test, autoenc, tgt_id, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, time_steps, batch_size=1024):
    autoenc.eval()
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")
    criterion = [nn.MSELoss(), nn.CrossEntropyLoss()]

    with converter_test.make_torch_dataloader(batch_size=batch_size, data_loader_fn=BatchedDataLoader,
                                              num_epochs=None, cache_type=PS_CACHE_TYPE,
                                              workers_count=get_default_parallelism(),
                                              reader_pool_type=PS_WORKER_TYPE,
                                              hdfs_driver=PS_HDFS_DRIVER) as test_dataloader:
        test_dataloader_iter = iter(test_dataloader)
        steps_per_epoch = max(1, len(converter_test) // (batch_size))
        embeddings = run_autoencoder_score_peta(autoenc, steps_per_epoch, test_dataloader_iter,
                                                device, tgt_id, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, time_steps) # noqa: E1121
    return embeddings
