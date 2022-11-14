import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from caspr.data.common_dataset import CommonDataset, id_collate


def transform_and_load(batch, device, tgt_id_cols, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, time_steps):
    """
    Transforms a batch of feature tensors from Petastorm, into input tensors for CASPR, then loads onto chosen device.
    """
    if not batch:
        raise ValueError("non-empty batch of tensors required")

    if not int(time_steps) > 0:
        raise ValueError("time_steps should be a positive integer")

    batch_size = batch[list(batch.keys())[0]].shape[0]

    seq_contX_cols = [item for item in seq_cols if item in cont_cols]
    if seq_contX_cols:
        seq_contX = torch.cat([batch[c] for c in seq_contX_cols], 0).float().to(device)
        seq_contX = seq_contX.reshape(-1, time_steps, batch_size).T
    else:
        seq_contX = torch.zeros((batch_size, time_steps, 0), device=device).float()

    seq_catX_cols = [item for item in seq_cols if item in cat_cols]
    if seq_catX_cols:
        seq_catX = torch.cat([batch[c] for c in seq_catX_cols], 0).long().to(device)
        seq_catX = seq_catX.reshape(-1, time_steps, batch_size).T
    else:
        seq_catX = torch.zeros((batch_size, time_steps, 0), device=device).long()

    non_seq_catX_cols = [item for item in non_seq_cols if item in cat_cols]
    if non_seq_catX_cols:
        non_seq_catX = torch.cat([batch[c] for c in non_seq_catX_cols], 0).long().to(device)
        non_seq_catX = non_seq_catX.reshape(len(non_seq_catX_cols), batch_size).T
    else:
        non_seq_catX = torch.zeros(batch_size, 0, device=device).long()

    non_seq_contX_cols = [item for item in non_seq_cols if item in cont_cols]
    if non_seq_contX_cols:
        non_seq_contX = torch.cat([batch[c] for c in non_seq_contX_cols], 0).float().to(device)
        non_seq_contX = non_seq_contX.reshape(len(non_seq_contX_cols), batch_size).T
    else:
        non_seq_contX = torch.zeros(batch_size, 0, device=device).float()

    if output_col:
        y = torch.cat([batch[c] for c in output_col], 0).to(device)
        y = y.reshape((len(output_col), -1)).T
    else:
        y = torch.zeros(batch_size, 0, device=device).float()

    if tgt_id_cols:
        tgt_id = torch.cat([batch[c] for c in tgt_id_cols], 0).long().cpu()
        tgt_id = tgt_id.reshape(len(tgt_id_cols), batch_size).T.numpy()
    else:
        tgt_id = torch.zeros(batch_size, 0).long().cpu().numpy()

    return tgt_id, y, seq_catX, seq_contX, non_seq_catX, non_seq_contX


def init_datasets(df, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, seq_len, test_ratio=0.2, seed=None):
    """
    Splits an incoming columnar dataframe into CASPR train and validation datasets
    """

    train_pd, val_pd = train_test_split(df, test_size=test_ratio, random_state=seed)

    print(f"train: {len(train_pd)}, val: {len(val_pd)}")

    dataset_train = CommonDataset(
        train_pd, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, seq_len)

    dataset_val = CommonDataset(
        val_pd, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, seq_len)

    return dataset_train, dataset_val


def init_loaders(ds_train, ds_val, batch_size, num_workers=0, shuffle=False, pin_memory=True, world_size=1, rank=0):
    """
    Initializes train and validation data loaders. The loaders support distributed sampling when world_size > 1.
    """

    print("Initializing dataloaders... Replica: %d of %d" % (rank + 1, world_size))

    val_sampler = DistributedSampler(ds_val,
                                     num_replicas=world_size, rank=rank, shuffle=shuffle) if world_size > 1 else None

    val_loader = DataLoader(ds_val, pin_memory=pin_memory,
                            batch_size=batch_size, num_workers=num_workers, sampler=val_sampler, collate_fn=id_collate)

    train_sampler = DistributedSampler(ds_train,
                                       num_replicas=world_size, rank=rank, shuffle=shuffle) if world_size > 1 else None

    train_loader = DataLoader(ds_train, pin_memory=pin_memory,
                              batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, collate_fn=id_collate)

    return train_loader, val_loader
