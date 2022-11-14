"""Early stopping class for nn models."""

import logging

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from caspr.utils.onnx import export_onnx, register_custom_op


class EarlyStopping:
    """Stop the training early and save a PyTorch or ONNX model after specified iterations (patience)"""

    def __init__(self, logger, should_decrease,patience=3, verbose=True, delta=0, save_onnx=False):
        """Initialize the early stopping module.

        Args:
            logger: For logging
            should_decrease (bool): True if metrics improve by decreasing.
            patience (int): How long to wait after last time validation score improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation score improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            save_onnx (bool): If True, export the model as onnx format.
                              Default: False
        """
        self.logger = logger
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.should_decrease = should_decrease
        self.delta = delta
        self.save_onnx = save_onnx
        if self.save_onnx:
            register_custom_op()

    def __call__(self, val_score, model, path):
        """Define __call__ method.

        Args:
            val_score (float): Validation score to determine whether to early stop.
            model (nn.Module): Model being trained.
            path (str): Model save path.
        """

        if self.should_decrease:
            val_score = -val_score

        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model, path)
        elif val_score <= self.best_score + self.delta:
            self.counter += 1
            self.logger.info('EarlyStopping counter: {} out of {}\n'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.save_checkpoint(model, path)
            self.counter = 0

    def save(self, model, path):
        if self.save_onnx:
            export_onnx(model, path)
        else:
            torch.save(model.state_dict(), path)

    def save_checkpoint(self, model, path):
        """Save model when validation score improves.

        The model parameter can be a list that allows multiple models to be saved.
        """

        if self.verbose:
            self.logger.info('Validation score improved.  Saving model ...\n')
        if not isinstance(model, list):
            self.save(model, path)
        else:
            for m, p in zip(model, path):
                self.save(m, path)


class DistributedEarlyStopping(EarlyStopping):
    def __init__(self, logger, should_decrease=True, patience=3, verbose=True, delta=0, rank=None, save_onnx=False):
        super().__init__(logger, should_decrease, patience=patience, verbose=verbose, delta=delta, save_onnx=save_onnx)
        self.rank = rank
    
    def __call__(self, val_score, model, path, rank=None):
        if not rank:
            rank = self.rank

        if rank and rank > 0:
            return

        if isinstance(model, DDP):
            model = model.module
            
        return super().__call__(val_score, model, path)


if __name__ == '__main__':

    class TwoLayerNet(torch.nn.Module):
        """Simple two-layer neural network for demonstration purposes."""

        def __init__(self, D_in, H, D_out):
            """Instantiate two nn.Linear modules and assign them as member variables."""

            super().__init__()

            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, D_out)

        def forward(self, x):
            """In the forward function we accept a Tensor of input data and we must return a Tensor of output data.

            We can use Modules defined in the constructor as well as arbitrary operators on Tensors.
            """

            h_relu = self.linear1(x).clamp(min=0)
            y = self.linear2(h_relu)
            return y

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    batch_size, input_dim, hidden_dim, output_dim = 1000, 1000, 100, 10

    # Create random Tensors to hold inputs and outputs
    X = torch.randn(batch_size, input_dim)
    y_true = torch.randn(batch_size, output_dim)

    # Construct our model by instantiating the class defined above
    mlp = TwoLayerNet(input_dim, hidden_dim, output_dim)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-4)
    logger = logging.getLogger(__name__)
    early_stopping = EarlyStopping(logger, should_decrease=True, patience=3, verbose=True, delta=1e-5)
    
    for t in range(10000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = mlp(X)

        # Compute and print loss
        loss = criterion(y_pred, y_true)
        if t % 100 == 99:
            logger.info(t, loss.item())
            early_stopping(loss.item(), mlp, 'early_stopping_test_model.pth')
            if early_stopping.early_stop:
                break

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mlp.load_state_dict(torch.load('early_stopping_test_model.pth'))
    y_pred = mlp(X)
    loss = criterion(y_pred, y_true)
    logger.info('Best loss: {}'.format(loss.item()))
