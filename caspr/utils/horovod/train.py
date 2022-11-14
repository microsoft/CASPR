import logging

import horovod.torch as hvd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.distributed import DistributedSampler

from caspr.data.common_dataset import id_collate
from caspr.utils.early_stopping import DistributedEarlyStopping
from caspr.utils.train import init_lr_schedulers, run_autoencoder, run_autoencoder_val

BATCH_SIZE = 1024 * 32
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_DELTA = 1e-5
ROOT_RANK = 0
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


def determine_early_stop(early_stopper: DistributedEarlyStopping, loss_averaged, model, path, epoch, num_epochs):
    # Call the distributed early stopper while passing rank info
    # Only rank 0 is allowed to checkpoint
    early_stopper(loss_averaged, model,
                  path, hvd.rank())
    if early_stopper.early_stop:
        epoch = num_epochs
    # The answer to whether to stop or not is decided by the root rank
    # The answer is then broadcased to other nodes
    epoch = hvd.broadcast_object(epoch, root_rank=ROOT_RANK)

    # The root rank loads the latest model checkpoint and broadcasts parameters
    if hvd.rank() == ROOT_RANK and epoch == num_epochs:
        model.load_state_dict(torch.load(path))
    hvd.broadcast_parameters(model.state_dict(), root_rank=ROOT_RANK)
    return epoch


def train_hvd(dataset_train, autoenc, device, batch_size=1024, epochs=10, learning_rate=0.01, warmup_epochs=5, save_model=False, path='./early_stopping_test_model.pth'):
    autoenc.train()
    hvd.init()
    logger.info("Number of workers:" + str(hvd.size()))

    if device.type == 'cuda':
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())

    # Configure the sampler such that each worker obtains a distinct sample of input dataset.
    train_sampler = DistributedSampler(dataset_train, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, sampler=train_sampler, collate_fn=id_collate)

    num_epochs = epochs

    # Effective batch size in synchronous distributed training is scaled by the number of workers.
    # An increase in learning rate compensates for the increased batch size.
    optimizer = optim.Adam(autoenc.parameters(), lr=learning_rate * hvd.size())
    # Wrap the optimizer with Horovod's DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=autoenc.named_parameters())

    scheduler_wu, scheduler_re = init_lr_schedulers(
        optimizer, warmup_epochs, reduce_patience=int(EARLY_STOPPING_PATIENCE/2), verbose=True)

    hvd.broadcast_object(scheduler_wu, root_rank=ROOT_RANK)
    hvd.broadcast_object(scheduler_re, root_rank=ROOT_RANK)

    # Broadcast initial parameters so all workers start with the same parameters.
    hvd.broadcast_parameters(autoenc.state_dict(), root_rank=ROOT_RANK)

    criterion = [nn.MSELoss(), nn.CrossEntropyLoss()]

    losses = []
    early_stopper = DistributedEarlyStopping(logger, patience=EARLY_STOPPING_PATIENCE, delta=EARLY_STOPPING_DELTA)

    epoch = 1
    while epoch < num_epochs + 1:
        losses, _ = run_autoencoder(autoenc, optimizer, train_loader, criterion, device)
        loss_averaged = metric_average(torch.tensor(losses), 'avg_loss')
        logger.info("Average overall training loss in epoch {0} is {1}".format(
            epoch, loss_averaged))

        epoch = determine_early_stop(early_stopper, loss_averaged, autoenc, path, epoch, num_epochs)

        if epoch <= warmup_epochs:
            scheduler_wu.step()
        scheduler_re.step(loss_averaged)

        if hvd.rank() == ROOT_RANK and epoch == num_epochs:
            if save_model:
                save_checkpoint(autoenc, optimizer, epoch, 'encoder')
            return autoenc, loss_averaged
        epoch = epoch+1


def train_val_hvd(dataset_train, dataset_val, autoenc, device, batch_size=1024, epochs=10, learning_rate=0.01, warmup_epochs=5, save_model=False, path='./early_stopping_test_model.pth'):
    autoenc.train()
    hvd.init()
    logger.info("Number of workers:" + str(hvd.size()))

    if device.type == 'cuda':
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())

    # Configure the sampler such that each worker obtains a distinct sample of input dataset.
    train_sampler = DistributedSampler(dataset_train, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, sampler=train_sampler, collate_fn=id_collate)

    val_sampler = DistributedSampler(dataset_val, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                             sampler=val_sampler, collate_fn=id_collate)

    num_epochs = epochs

    # Effective batch size in synchronous distributed training is scaled by the number of workers.
    # An increase in learning rate compensates for the increased batch size.
    optimizer = optim.Adam(autoenc.parameters(), lr=learning_rate * hvd.size())
    # Wrap the optimizer with Horovod's DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=autoenc.named_parameters())

    scheduler_wu, scheduler_re = init_lr_schedulers(
        optimizer, warmup_epochs, reduce_patience=int(EARLY_STOPPING_PATIENCE/2), verbose=True)

    hvd.broadcast_object(scheduler_wu, root_rank=ROOT_RANK)
    hvd.broadcast_object(scheduler_re, root_rank=ROOT_RANK)

    # Broadcast initial parameters so all workers start with the same parameters.
    hvd.broadcast_parameters(autoenc.state_dict(), root_rank=ROOT_RANK)

    criterion = [nn.MSELoss(), nn.CrossEntropyLoss()]

    losses = []
    early_stopper = DistributedEarlyStopping(logger, patience=EARLY_STOPPING_PATIENCE, delta=EARLY_STOPPING_DELTA)

    epoch = 1
    while epoch < num_epochs + 1:
        autoenc.train()
        losses, _ = run_autoencoder(autoenc, optimizer, train_loader, criterion, device)
        autoenc.eval()
        losses_val = run_autoencoder_val(autoenc, val_loader, criterion, device)
        loss_train_averaged = metric_average(torch.tensor(losses), 'avg_train_loss')
        loss_val_averaged = metric_average(torch.tensor(losses_val), 'avg_val_loss')

        logger.info("Average training loss in epoch {0} is {1}".format(epoch, loss_train_averaged))
        logger.info("Average validation loss in epoch {0} is {1}".format(epoch, loss_val_averaged))

        if epoch <= warmup_epochs:
            scheduler_wu.step()
        scheduler_re.step(loss_val_averaged)

        epoch = determine_early_stop(early_stopper, loss_val_averaged, autoenc, path, epoch, num_epochs)
        if hvd.rank() == ROOT_RANK and epoch == num_epochs:
            if save_model:
                save_checkpoint(autoenc, optimizer, epoch, 'encoder')
            return autoenc, loss_val_averaged
        epoch = epoch+1
