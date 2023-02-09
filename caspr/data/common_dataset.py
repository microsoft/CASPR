# coding: utf-8
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import default_collate


class CommonDataset(torch.utils.data.Dataset):
    def __init__(self, df, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, time_steps, tgt_id=[]):
        self.len = df.shape[0]
        self.seq_cols = seq_cols if seq_cols else []

        self.non_seq_cols = non_seq_cols
        self.output_col = output_col

        self.seq_contX = torch.tensor(df[[item for item in seq_cols if item in cont_cols]].values, dtype=torch.float32)
        self.seq_catX = torch.tensor(df[[item for item in seq_cols if item in cat_cols]].values, dtype=torch.long)

        self.seq_contX = self.seq_contX.reshape(
            (self.seq_contX.shape[0], int(self.seq_contX.shape[1]/time_steps), time_steps))
        self.seq_contX = self.seq_contX.permute(0, 2, 1)

        self.seq_catX = self.seq_catX.reshape(
            (self.seq_catX.shape[0], int(self.seq_catX.shape[1]/time_steps), time_steps))
        self.seq_catX = self.seq_catX.permute(0, 2, 1)

        self.non_seq_catX = torch.tensor(
            df[[item for item in non_seq_cols if item in cat_cols]].values, dtype=torch.long)
        self.non_seq_contX = torch.tensor(
            df[[item for item in non_seq_cols if item in cont_cols]].values, dtype=torch.float32)

        self.y = torch.tensor(df[output_col].values, dtype=torch.float32)

        self.tgt_id = df[tgt_id].values

    @classmethod
    def for_inference(cls, continuous: pd.Series, categorical: pd.Series, seq_cols, non_seq_cols, cat_cols, cont_cols, time_steps):
        cont_df = pd.DataFrame(continuous.values.tolist(), columns=cont_cols)
        cat_df = pd.DataFrame(categorical.values.tolist(), columns=cat_cols)
        
        df = pd.concat([cont_df, cat_df], axis=1)
        return cls(df, seq_cols, non_seq_cols, [], cat_cols, cont_cols, time_steps, tgt_id=[])

    def __getitem__(self, index):
        return [self.tgt_id[index], self.y[index], self.seq_catX[index], self.seq_contX[index], self.non_seq_catX[index], self.non_seq_contX[index]]

    def __len__(self):
        return self.len


def id_collate(batch):
    ids = []
    new_batch = []
    for _batch in batch:
        ids.append(_batch[0])
        new_batch.append(_batch[1:])
    ids = np.stack(ids, axis=0)
    return tuple([ids] + default_collate(new_batch))
