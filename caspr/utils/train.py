# coding: utf-8

import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from caspr.data.load import init_loaders
from caspr.models.factory import CASPRFactory
from caspr.models.model_wrapper import AutoencoderTeacherTraining, LSTMAutoencoder, TransformerAutoEncoder
from caspr.utils.early_stopping import DistributedEarlyStopping, EarlyStopping
from caspr.utils.metrics import get_metrics
from caspr.utils.onnx import ONNXWrapper
from caspr.utils.score import get_architecture

DDP_BACKEND = "nccl"
DDP_MASTER_ADDR = "localhost"
DDP_MASTER_PORT = "12355"
DDP_LOAD_WORKERS = 1
STD_LOAD_WORKERS = 0
logger = logging.getLogger(__name__)


def run_autoencoder(autoenc, optimizer, dataloader_train, criterion, device):
    count = 0
    epoch_start_time = time.time()
    running_loss = 0.0

    for _, y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data in dataloader_train:
        y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data = y.to(device), seq_cat_data.to(
            device), seq_cont_data.to(device), non_seq_cat_data.to(device), non_seq_cont_data.to(device)

        _, loss = autoenc.run(y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data, criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = (running_loss * count + loss.item()) / (count + 1)

        count = count + 1

        if count % 64 == 0:
            logger.info(loss, count*seq_cat_data.shape[0])
            time_so_far = time.time() - epoch_start_time
            logger.info("Time taken since start:" + str(time_so_far))

    epoch_end_time = time.time()
    logger.info(epoch_end_time - epoch_start_time)

    return running_loss, epoch_end_time - epoch_start_time


def run_autoencoder_val(autoenc, dataloader_val, criterion, device):
    count = 0
    running_loss = 0.0

    for _, y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data in dataloader_val:
        y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data = y.to(device), seq_cat_data.to(
            device), seq_cont_data.to(device), non_seq_cat_data.to(device), non_seq_cont_data.to(device)

        _, loss = autoenc.run(y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data, criterion)

        running_loss = (running_loss * count + loss.item()) / (count + 1)
        count = count + 1

        if count % 64 == 0:
            logger.info(loss, count*seq_cat_data.shape[0])

    return running_loss


def run_epoch(model, epoch, dataloader, criterion, device, optimizer=None, is_train=True, get_outputs=False):
    model.to(device)
    losses = []
    y_labels = []
    y_preds = []

    if isinstance(model, DDP):
        model = model.module

    for _, y, seq_cat_x, seq_cont_x, non_seq_cat_x, non_seq_cont_x in dataloader:
        if is_train:
            optimizer.zero_grad()

        seq_cat_x = seq_cat_x.to(device)
        seq_cont_x = seq_cont_x.to(device)
        non_seq_cat_x = non_seq_cat_x.to(device)
        non_seq_cont_x = non_seq_cont_x.to(device)
        y = y.to(device)

        # Forward Pass
        y_pred, loss = model.run(y, seq_cat_x, seq_cont_x, non_seq_cat_x, non_seq_cont_x, criterion=criterion)
        losses.append(loss.detach().cpu().numpy())

        if get_outputs:
            y_labels.append(y)
            y_preds.append(y_pred)

        # Backward Pass and Optimization
        if is_train:
            loss.backward()
            optimizer.step()

    if get_outputs:
        y_labels = torch.cat(y_labels, 0).detach().cpu().numpy()
        y_preds = torch.cat(y_preds, 0).detach().cpu().numpy()

    mean_loss = np.mean(np.asarray(losses))
    mode = 'training' if is_train else 'validation'
    logger.info("Average {} loss in epoch {} is {}".format(mode, epoch, mean_loss))
    return y_labels, y_preds, mean_loss


def init_lr_schedulers(optimizer, warmup_epochs, reduce_mode='min', reduce_factor=0.1, reduce_patience=4, verbose=True):
    """
    Training batch size grows proportionally with training distribution, mandating upscaling of the learning rate, which in turn reduces the probability of finding the global optimum.
    This function initializes learning rate schedulers for a given optimizer to facilitate dynamic adjustment (reduction) of learning rate during training.
    """
    
    warm_up = lambda epoch: epoch / warmup_epochs if warmup_epochs > 0 & epoch <= warmup_epochs else 1
    scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode=reduce_mode, factor=reduce_factor, patience=reduce_patience, verbose=verbose)

    return scheduler_wu, scheduler_re


def train_model(model, criterion, num_epochs, dataloader_train, dataloader_val, device, save_path, lr=1e-3, fix_module_names=None,
                should_decrease=True, patience=8, verbose=True, evaluate_downstream=False, rank=0, world_size=1, warmup_epochs=5, save_onnx=False):

    if isinstance(model, (LSTMAutoencoder, AutoencoderTeacherTraining, TransformerAutoEncoder)) and evaluate_downstream:
        raise ValueError('evaluate_downstream should be set to False when training autoencoder')

    if fix_module_names:
        fix_modules = [module for name, module in model.named_modules() if name in fix_module_names]
        for module in fix_modules:
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    scheduler_wu, scheduler_re = init_lr_schedulers(optimizer, warmup_epochs, reduce_patience=int(patience/2), verbose=verbose)

    if world_size > 1:
        early_stopping = DistributedEarlyStopping(logger, should_decrease, patience, verbose, rank=rank, save_onnx=save_onnx)
    else:
        early_stopping = EarlyStopping(logger, should_decrease, patience, verbose, save_onnx=save_onnx)

    for epoch in range(num_epochs):
        start = time.time()

        model.train()
        if fix_module_names:
            for module in fix_modules:
                module.eval()

        run_epoch(model, epoch, dataloader_train, criterion, device, optimizer)

        model.eval()
        with torch.no_grad():
            y_labels, y_preds, mean_val_loss = run_epoch(model, epoch, dataloader_val, criterion, device,
                                                         is_train=False, get_outputs=evaluate_downstream)
            if evaluate_downstream:
                get_metrics(y_labels, y_preds)

            end = time.time()
            logger.info("Time for epoch {0} is {1}\n".format(epoch, (end - start)))
            logger.info("Mean validation loss for epoch {0} is {1}\n".format(epoch, mean_val_loss))

            if epoch <= warmup_epochs:
                scheduler_wu.step()
            scheduler_re.step(mean_val_loss)

            early_stopping(mean_val_loss, model, save_path)
            if early_stopping.early_stop:
                logger.info('early stopping at epoch {}'.format(epoch))
                break

    if rank == 0:
        if save_onnx:
            model_type = get_architecture(model)
            model = ONNXWrapper(save_path, model_type)
        elif isinstance(model, DDP):
            model.module.load_state_dict(torch.load(save_path))
        else:
            model.load_state_dict(torch.load(save_path))
        return model


def __setup_ddp(rank, world_size):

    os.environ['MASTER_ADDR'] = DDP_MASTER_ADDR
    os.environ['MASTER_PORT'] = DDP_MASTER_PORT

    # initialize the process group
    dist.init_process_group(DDP_BACKEND, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def __do_train_ddp(rank, args):

    __setup_ddp(rank, args['world_size'])

    caspr_factory = args['caspr_factory']

    model = caspr_factory.create(args['caspr_arch'], **args['hyper_params'])

    model = DDP(model.cuda(), device_ids=[rank])

    train_loader, val_loader = init_loaders(args['ds_train'], args['ds_val'], args['batch_size'],
                                            num_workers=DDP_LOAD_WORKERS, world_size=args['world_size'], rank=rank)

    train_model(model, args['criterion'], args['num_epochs'], train_loader, val_loader, rank, args['save_path'],
                lr=args['lr'] * args['world_size'], rank=rank, world_size=args['world_size'], **args['kwargs'])

    dist.destroy_process_group()


def train_model_ddp(caspr_factory : CASPRFactory, caspr_arch : str, hyper_params : dict, ds_train, ds_val, criterion, num_epochs, batch_size, save_path, lr=1e-3, **kwargs):
    """
    Distributed Data Parallel implementation of CASPR training. Will use all GPUs available on the current machine.

    Arguments:
    ----------

    caspr_factory:  CASPR model factory for the specified dataset

    caspr_arch: CASPR architecture e.g. TransformerAutoEncoder

    hyper_params:  parameters for instantiating a new CASPR model with the above method

    ds_train:  CommonDataset for training

    ds_val: CommonDataset for validation

    criterion, num_epochs, batch_size, save_path, lr: self explanatory

    **kwargs: any other parameters to be passed to the train_model function by the DDP worker (e.g. evaluate, verbose or patience)

    Returns: Trained model

    """
    logger.info("Setting up model training using torch DDP")

    for arg in [caspr_factory, caspr_arch, ds_train, ds_val, criterion, num_epochs, batch_size, save_path, lr]:
        if not arg:
            raise ValueError("Illegal null argument. Check for None values and try again.")

    world_size = torch.cuda.device_count()

    if not torch.cuda.is_available() or world_size < 2:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.warn("DDP mode disabled. Training on %s..." % device)
        model = caspr_factory.create(caspr_arch, device=device, **hyper_params)
        train_loader, val_loader = init_loaders(ds_train, ds_val, batch_size, num_workers=STD_LOAD_WORKERS)
        return train_model(model, criterion, num_epochs, train_loader, val_loader, device, save_path, lr, **kwargs)

    logger.info("DDP mode enabled, will train on %d GPUs" % world_size)

    arguments = locals()

    mp.spawn(__do_train_ddp,
             args=(arguments,),
             nprocs=world_size,
             join=True)

    model = caspr_factory.create(caspr_arch, **hyper_params)
    model.load_state_dict(torch.load(save_path))
    return model


def test_model(model, dataloader_test, criterion, device):
    model.eval()
    with torch.no_grad():
        y_labels, y_preds, _ = run_epoch(
            model, 0, dataloader_test, criterion, device, is_train=False, get_outputs=True)
    return y_labels, y_preds


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
