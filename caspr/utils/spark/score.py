import logging
from math import frexp

import numpy as np
import pandas as pd
import torch
from pyspark.sql.functions import array, col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType

from caspr.data.common_dataset import CommonDataset, id_collate
from caspr.models.factory import LSTM, TRANSFORMER
from caspr.utils.preprocess import get_nonempty_tensors
from caspr.utils.score import get_architecture

logger = logging.getLogger(__name__)


def score(df, model, seq_cols, non_seq_cols, cat_cols, cont_cols, time_steps, batch_size=16*2048):
    model.eval()

    # vectorizing continuous and discrete features separately
    output = df.withColumn('cont_features', array([col(f) for f in cont_cols])).drop(*cont_cols)
    output = output.withColumn('cat_features', array([col(f) for f in cat_cols])).drop(*cat_cols)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info("Scoring on: %s" % device)

    # making sure the model is on CPU before the UDF is defined
    model.cpu()

    def calculate_embeddings(continuous, categorical):
        try:
            model.to(device)
            embeddings = []
            batch_ds = CommonDataset.for_inference(continuous, categorical, seq_cols,
                                                   non_seq_cols, cat_cols, cont_cols, time_steps)

            # nested batching to ensure Spark does not trigger CUDA OOM with larger datasets
            data_loader = torch.utils.data.DataLoader(batch_ds, batch_size=batch_size, collate_fn=id_collate)

            for _, _, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data in data_loader:

                data = [seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data]
                if isinstance(model, torch.nn.Module):
                    data = [d.to(device) for d in data]

                nonempty_tensors, nonempty_idx = get_nonempty_tensors(data)

                if get_architecture(model) == TRANSFORMER:
                    emb, _, _ = model.unified_encoder(*nonempty_tensors, nonempty_idx)
                    # Concatenate across timesteps
                    emb = emb.reshape(emb.shape[0], -1)
                    embeddings.append(emb.detach().cpu() if isinstance(emb, torch.Tensor) else emb)

                elif get_architecture(model) == LSTM:
                    _, (hn, _) = model.unified_encoder(*nonempty_tensors, nonempty_idx)
                    embeddings.append(hn.detach().cpu() if isinstance(hn, torch.Tensor) else hn)

            embeddings = pd.DataFrame(np.concatenate(embeddings, axis=0))

            return pd.Series(embeddings.values.tolist())

        finally:
            # can release resources here, if needed
            pass

    # Pandas UDF declaration with float[] return type
    score_udf = pandas_udf(calculate_embeddings, ArrayType(FloatType()))

    # Calculating the embeddings as an additional column and dropping the temporary vectors
    output = output.withColumn('embeddings', score_udf('cont_features', 'cat_features')
                               ).drop('cont_features', 'cat_features')

    return output
