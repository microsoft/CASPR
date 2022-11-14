import logging
import time

import horovod.torch as hvd
import torch
import torch.nn as nn
from petastorm.pytorch import BatchedDataLoader
from torch import optim
from torch.utils.data.distributed import DistributedSampler

from caspr.data.load import transform_and_load
from caspr.utils.early_stopping import DistributedEarlyStopping
from caspr.utils.horovod.train import determine_early_stop
from caspr.utils.spark.large.score import get_default_parallelism
from caspr.utils.spark.preprocess import remove_underscore_in_seq_col_name_list
from caspr.utils.train import init_lr_schedulers

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, name):
    filepath = '/checkpoint-{epoch}-{model}.pth'.format(epoch=epoch, model=name)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)


def metric_average(metric, name):
    avg_tensor = hvd.allreduce(metric, name=name)
    return avg_tensor.item()


BATCH_SIZE = 1024 * 32
NUM_EPOCHS = 100
NUM_WORKERS = 4  # assume cluster consists of two workers 2x K80 each
# default loader parallism is low or None, this widens the IO bottleneck when feeding each GPU
PS_WORKERS_PER_CPU = 2
# this version is implemented in C, vs Java (slower) default
PS_HDFS_DRIVER = 'libhdfs3'
# lower overhead, alternative is 'process'
PS_WORKER_TYPE = 'thread'
# assuming the training relies on SSD backed dbfs:/ml, Petastorm's caching can be disabled
PS_CACHE_TYPE = None
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_DELTA = 1e-5
ROOT_RANK = 0


def run_autoencoder_peta(autoenc, optimizer, steps_per_epoch, train_dataloader_iter, criterion, device, tgt_id_col, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, time_steps):
    count = 0
    val_start_time = time.time()
    running_loss = 0.0
    for _ in range(steps_per_epoch):
        pd_batch = next(train_dataloader_iter)
        _, y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data = transform_and_load(
            pd_batch, device, remove_underscore_in_seq_col_name_list(tgt_id_col), remove_underscore_in_seq_col_name_list(seq_cols), remove_underscore_in_seq_col_name_list(non_seq_cols), output_col, remove_underscore_in_seq_col_name_list(cat_cols), remove_underscore_in_seq_col_name_list(cont_cols), time_steps)

        # Track history in training
        torch.set_grad_enabled(True)
        _, loss = autoenc.run(y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data, criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = (running_loss * count + loss.item()) / (count + 1)
        count = count + 1
        if count % 64 == 0:
            logger.info("Running Loss so far: " + str(running_loss))
            logger.info("Records processed so far: " + str(count*seq_cat_data.shape[0]))
            time_so_far = time.time() - val_start_time
            logger.info("Time taken since start:" + str(time_so_far))

    val_end_time = time.time()

    logger.info("Total time taken:" + str(val_end_time - val_start_time))
    logger.info("Running loss at the end of training epoch:" + str(running_loss))
    return running_loss, val_end_time - val_start_time


def run_autoencoder_val_peta(autoenc, steps_per_epoch, val_dataloader_iter, criterion, device, tgt_id_col, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, time_steps):
    criterion = [nn.MSELoss(), nn.CrossEntropyLoss()]
    count = 0
    val_start_time = time.time()
    running_loss = 0.0
    for _ in range(steps_per_epoch):
        pd_batch = next(val_dataloader_iter)
        _, y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data = transform_and_load(
            pd_batch, device, remove_underscore_in_seq_col_name_list(tgt_id_col), remove_underscore_in_seq_col_name_list(seq_cols), remove_underscore_in_seq_col_name_list(non_seq_cols), output_col, remove_underscore_in_seq_col_name_list(cat_cols), remove_underscore_in_seq_col_name_list(cont_cols), time_steps)

        # Track history in training
        torch.set_grad_enabled(False)
        _, loss = autoenc.run(y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data, criterion)

        running_loss = (running_loss * count + loss.item()) / (count + 1)

        count = count + 1
        if count % 64 == 0:
            logger.info("Running Loss so far: " + str(running_loss))
            logger.info("Records processed so far: " + str(count*seq_cat_data.shape[0]))
            time_so_far = time.time() - val_start_time
            logger.info("Time taken since start:" + str(time_so_far))

    val_end_time = time.time()

    logger.info("Total time taken:" + str(val_start_time - val_end_time))
    logger.info("Running loss at the end of validation epoch:" + str(running_loss))
    return running_loss, val_start_time - val_end_time


def train_peta_hvd(converter_train, autoenc, tgt_id, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, time_steps, batch_size=1024, epochs=10, learning_rate=0.01, warmup_epochs=5, save_model=False, path='./early_stop_model.pth'):
    autoenc.train()
    hvd.init()  # Initialize Horovod.
    logger.info("Number of workers:" + str(hvd.size()))
    # Horovod: pin GPU to local rank.
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")

    # from torch.utils.data.distributed import DistributedSampler
    # Configure the sampler such that each worker obtains a distinct sample of input dataset.
        # train_sampler = DistributedSampler(dataset_train, num_replicas=hvd.size(), rank=hvd.rank())
    # Use trian_sampler to load a different sample of data on each worker.
        # train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler)

    autoenc = autoenc.to(device)
    num_epochs = epochs

    # Effective batch size in synchronous distributed training is scaled by the number of workers.
    # An increase in learning rate compensates for the increased batch size.
    optimizer = optim.Adam(autoenc.parameters(), lr=learning_rate * hvd.size())

    # Broadcast initial parameters so all workers start with the same parameters.
    hvd.broadcast_parameters(autoenc.state_dict(), root_rank=ROOT_RANK)
    hvd.broadcast_optimizer_state(optimizer, root_rank=ROOT_RANK)

    # Wrap the optimizer with Horovod's DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=autoenc.named_parameters())

    scheduler_wu, scheduler_re = init_lr_schedulers(
        optimizer, warmup_epochs, reduce_patience=int(EARLY_STOPPING_PATIENCE/2), verbose=True)

    hvd.broadcast_object(scheduler_wu, root_rank=ROOT_RANK)
    hvd.broadcast_object(scheduler_re, root_rank=ROOT_RANK)

    criterion = [nn.MSELoss(), nn.CrossEntropyLoss()]
    early_stopper = DistributedEarlyStopping(logger, patience=EARLY_STOPPING_PATIENCE, delta=EARLY_STOPPING_DELTA)

    with converter_train.make_torch_dataloader(cur_shard=hvd.rank(), shard_count=hvd.size(),
                                               batch_size=batch_size, data_loader_fn=BatchedDataLoader,
                                               num_epochs=None, cache_type=PS_CACHE_TYPE,
                                               workers_count=PS_WORKERS_PER_CPU * get_default_parallelism(),
                                               reader_pool_type=PS_WORKER_TYPE,
                                               hdfs_driver=PS_HDFS_DRIVER) as train_dataloader:
        train_dataloader_iter = iter(train_dataloader)
        steps_per_epoch = max(1, len(converter_train) // (batch_size * hvd.size()))
        total_time = 0

        epoch = 1
        while epoch < num_epochs + 1:
            loss, epoch_time = run_autoencoder_peta(autoenc, optimizer, steps_per_epoch, train_dataloader_iter,
                                                    criterion, device, tgt_id, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, time_steps)
    #       Only save checkpoints on the first worker.
            total_time = total_time + epoch_time
            loss_averaged = metric_average(torch.tensor(loss), 'avg_loss')
            logger.info("Average overall training loss in epoch {0} is {1}".format(
                epoch, loss_averaged))

            if epoch <= warmup_epochs:
                scheduler_wu.step()
            scheduler_re.step(loss_averaged)

            epoch = determine_early_stop(early_stopper, loss_averaged, autoenc, path, epoch, num_epochs)
            if hvd.rank() == ROOT_RANK and epoch == num_epochs:
                if save_model:
                    save_checkpoint(autoenc, optimizer, epoch, 'encoder')
                return autoenc, loss_averaged, total_time
            epoch = epoch+1


def train_val_peta_hvd(converter_train, converter_val, autoenc, tgt_id, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, time_steps, batch_size=1024, epochs=10, learning_rate=0.01, warmup_epochs=5, save_model=False, path='./early_stop_model.pth'):
    autoenc.train()
    hvd.init()  # Initialize Horovod.
    logger.info("Number of workers:" + str(hvd.size()))
    # Horovod: pin GPU to local rank.
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")

    autoenc = autoenc.to(device)
    num_epochs = epochs

    # Effective batch size in synchronous distributed training is scaled by the number of workers.
    # An increase in learning rate compensates for the increased batch size.
    optimizer = optim.Adam(autoenc.parameters(), lr=learning_rate * hvd.size())

    # Broadcast initial parameters so all workers start with the same parameters.
    hvd.broadcast_parameters(autoenc.state_dict(), root_rank=ROOT_RANK)
    hvd.broadcast_optimizer_state(optimizer, root_rank=ROOT_RANK)

    # Wrap the optimizer with Horovod's DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=autoenc.named_parameters())

    scheduler_wu, scheduler_re = init_lr_schedulers(
        optimizer, warmup_epochs, reduce_patience=int(EARLY_STOPPING_PATIENCE/2), verbose=True)

    hvd.broadcast_object(scheduler_wu, root_rank=ROOT_RANK)
    hvd.broadcast_object(scheduler_re, root_rank=ROOT_RANK)

    criterion = [nn.MSELoss(), nn.CrossEntropyLoss()]
    early_stopper = DistributedEarlyStopping(logger, patience=EARLY_STOPPING_PATIENCE, delta=EARLY_STOPPING_DELTA)

    with converter_val.make_torch_dataloader(cur_shard=hvd.rank(), shard_count=hvd.size(),
                                             batch_size=batch_size, data_loader_fn=BatchedDataLoader,
                                             num_epochs=None, cache_type=PS_CACHE_TYPE,
                                             workers_count=PS_WORKERS_PER_CPU * get_default_parallelism(),
                                             reader_pool_type=PS_WORKER_TYPE,
                                             hdfs_driver=PS_HDFS_DRIVER) as val_dataloader, \
        converter_train.make_torch_dataloader(cur_shard=hvd.rank(), shard_count=hvd.size(),
                                              batch_size=batch_size, data_loader_fn=BatchedDataLoader,
                                              num_epochs=None, cache_type=PS_CACHE_TYPE,
                                              workers_count=PS_WORKERS_PER_CPU * get_default_parallelism(),
                                              reader_pool_type=PS_WORKER_TYPE,
                                              hdfs_driver=PS_HDFS_DRIVER) as train_dataloader:

        val_dataloader_iter = iter(val_dataloader)
        steps_val = max(1, len(converter_val) // (batch_size * hvd.size()))

        train_dataloader_iter = iter(train_dataloader)
        steps_per_epoch = max(1, len(converter_train) // (batch_size * hvd.size()))
        total_time = 0

        epoch = 1
        while epoch < num_epochs + 1:
            autoenc.train()
            _, epoch_time = run_autoencoder_peta(autoenc, optimizer, steps_per_epoch, train_dataloader_iter,
                                                 criterion, device, tgt_id, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, time_steps)
            autoenc.eval()
            val_loss, val_epoch_time = run_autoencoder_val_peta(autoenc, steps_val, val_dataloader_iter, criterion,
                                                                device, tgt_id, seq_cols, non_seq_cols, output_col, cat_cols, cont_cols, time_steps)
            total_time = total_time + epoch_time + val_epoch_time

            loss_averaged = metric_average(torch.tensor(val_loss), 'avg_loss')
            logger.info("Average overall training loss in epoch {0} is {1}".format(
                epoch, loss_averaged))

            if epoch <= warmup_epochs:
                scheduler_wu.step()
            scheduler_re.step(loss_averaged)

            epoch = determine_early_stop(early_stopper, loss_averaged, autoenc, path, epoch, num_epochs)
            if hvd.rank() == ROOT_RANK and epoch == num_epochs:
                if save_model:
                    save_checkpoint(autoenc, optimizer, epoch, 'encoder')
                return autoenc, loss_averaged, total_time
            epoch = epoch+1
