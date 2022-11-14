"""CASPR Model wrappers for running functionality."""
 
import logging

import torch
import torch.nn as nn

from caspr.utils.preprocess import get_nonempty_tensors

SEQ_CAT_INDEX = 0
SEQ_CONT_INDEX = 1
NON_SEQ_CAT_INDEX = 2
NON_SEQ_CONT_INDEX = 3

CRITERIA_CONT = 0
CRITERIA_CAT = 1

logger = logging.getLogger(__name__)

def set_default_tensor_type(device):
    """Set cuda or cpu as default tensor type."""
    if device in [torch.device("cpu"), "cpu"]:
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


class LSTMAutoencoder(nn.Module):  # noqa: W0223
    """Implementation of an autoencoder with multiple criterion.

    Args:
        unified_encoder (nn.Module): # noqa: W0223 Object of the unified_encoder class
        mlp_non_seq_cat_list (list of nn.module): List of linear layers with non-linear activation funtions
        mlp_non_seq_cont (nn.Module): # noqa: W0223 linear layers with non-linear activation funtions
        lstm_decoder (nn.Module): # noqa: W0223 Object of the LSTM_attention_embedding_decoder class
    """

    def __init__(self,  # noqa: R0913
                 unified_encoder,
                 mlp_non_seq_cat_list,
                 mlp_non_seq_cont,
                 decoder):
        """Initialize model with params."""
        super().__init__()

        self.unified_encoder = unified_encoder
        self.mlp_non_seq_cat_list = mlp_non_seq_cat_list
        self.mlp_non_seq_cont = mlp_non_seq_cont
        self.decoder = decoder

    def forward(self, *args):  # noqa: R0914
        """Run a forward pass of model over the data."""
        encoder_out, (hn, cn) = self.unified_encoder(*args)
        device = hn.device
        non_sequential_cont_decoded = self.mlp_non_seq_cont(hn)
        non_sequential_cat_decoded = []
        for mlp_non_seq_cat in self.mlp_non_seq_cat_list:
            non_sequential_cat_decoded.append(mlp_non_seq_cat(hn))

        hn = torch.unsqueeze(hn, 0)
        cn = torch.unsqueeze(cn, 0)
        # decoded is the output prediction of timestep i-1 of the decoder
        decoded = torch.zeros(encoder_out.shape[0], int(
            self.unified_encoder.seq_cont_count + self.unified_encoder.no_of_embs_seq), device=device)
        seq_cont_decoded = torch.Tensor(device=device)
        seq_cat_decoded = []
        for _ in range(self.unified_encoder.seq_cat_count):
            seq_cat_decoded.append(torch.Tensor(device=device))

        for _ in range(encoder_out.shape[1]):
            decoded, (hn, cn), out_cont, out_cat = self.decoder(decoded, (hn, cn))
            # Predict all categorical columns
            out_cat_onehot = []
            if self.unified_encoder.seq_cat_count != 0:
                for idx, out in enumerate(out_cat):
                    out_cat_onehot.append(torch.argmax(out, dim=1).unsqueeze(-1))
                    seq_cat_decoded[idx] = torch.cat(
                        [seq_cat_decoded[idx], out.view(out.shape[0], 1, -1)], dim=1)
                out_cat_onehot = torch.cat(out_cat_onehot, -1)
                out_cat_embedding = self.unified_encoder.seq_emb_layers(out_cat_onehot)
                decoded = torch.cat([out_cat_embedding, out_cont], dim=-1)
            else:
                decoded = out_cont
            seq_cont_decoded = torch.cat(
                [seq_cont_decoded, out_cont.view(out_cont.shape[0], 1, -1)], dim=1)

        return non_sequential_cont_decoded, non_sequential_cat_decoded, seq_cont_decoded, seq_cat_decoded

    def run(self,  # noqa: R0913, R0914
            y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data, criterion):  # noqa: W0613
        """Run model on data and propagate loss.

        Args:
            y (iterable of tensors): List of tensors to calculate model prediction losses
                against, for example, y could be a tuple/list
                (non_seq_cont_data, non_seq_cat_data, seq_cont_data, seq_cat_data)
            seq_cat_data,  seq_cont_data, non_seq_cat_data, non_seq_cont_data, (tensors):
                List of input tensors from the dataloader
            criterion (list of torch criterion objects): The loss function to use while calculating losses
        """
        data = (seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data)
        nonempty_tensors, nonempty_idx = get_nonempty_tensors(data)
        # model_out is expected to be a list of outputs
        model_out = self(*nonempty_tensors, nonempty_idx)
        losses = []

        criteria_cont = criterion[CRITERIA_CONT]
        criteria_cat = criterion[CRITERIA_CAT]

        if self.unified_encoder.seq_cat_count != 0:
            for idx, out in enumerate(model_out[-1]):
                losses.append(criteria_cat(out.reshape(-1, out.shape[2]), seq_cat_data[:, :, idx].reshape(-1)))
        losses.append(criteria_cont(model_out[-2], seq_cont_data))
        if self.unified_encoder.non_seq_cont_count != 0:
            losses.append(criteria_cont(model_out[0], non_seq_cont_data))
        if self.unified_encoder.non_seq_cat_count != 0:
            for idx, out in enumerate(model_out[1]):
                losses.append(criteria_cat(out.reshape(-1, out.shape[1]), non_seq_cat_data[:, idx].reshape(-1)))
        loss = sum(losses)
        return model_out, loss


class ChurnModel(nn.Module):  # noqa: W0223

    def __init__(self, enc, mlp):
        """Initialize model with params."""

        super().__init__()

        self.enc = enc
        self.mlp = mlp

    def forward(self, *args):
        """Run a forward pass of model over the data."""
        _, (hn, _) = self.enc(*args)

        y_pred = self.mlp(hn)
        return y_pred

    def run(self, y, seq_cont_data, seq_cat_data, non_seq_cont_data, non_seq_cat_data, criterion):  # noqa: R0913
        """Run model on data and propagate loss.

        Args:
            y (iterable of tensors): List of tensors to calculate model prediction losses
                against, for example, y could be a tuple/list
                (non_seq_cont_data, non_seq_cat_data, seq_cont_data, seq_cat_data)
            seq_cat_data,  seq_cont_data, non_seq_cat_data, non_seq_cont_data, (tensors):
                List of input tensors from the dataloader
            criterion (list of torch criterion objects): The loss function to use while calculating losses
        """
        data = (seq_cont_data, seq_cat_data, non_seq_cont_data, non_seq_cat_data)
        nonempty_tensors, nonempty_idx = get_nonempty_tensors(data)
        y_pred = self(*nonempty_tensors, nonempty_idx)
        loss = criterion(y_pred.flatten(), y)
        return y_pred, loss


class AutoencoderTeacherTraining(nn.Module):  # noqa: W0223
    """Teacher Training based autoencoder."""

    def __init__(self, enc, dec, timestep=15, seq_len=15):  # noqa: R0913
        """Initialize model with params."""

        super().__init__()

        self.enc = enc
        self.dec = dec

        self.timestep = timestep
        self.seq_len = seq_len
        self.decoder_start = 0 if timestep == seq_len else timestep

    def forward(self, *args):
        """Run a forward pass of model over the data."""
        _, (hn, cn) = self.enc(*args)

        device = hn.device
        seq_cont_data = args[1]
        seq_cat_data = args[0]
        batch_size = seq_cont_data.shape[0]

        decoder_input_cont = torch.cat((torch.zeros(
            batch_size, 1, seq_cont_data.shape[2], device=device),
            seq_cont_data[:, self.decoder_start:self.seq_len-1, :]), 1)
        decoder_input_cat = torch.cat((torch.zeros(
            batch_size, 1, seq_cat_data.shape[2], device=device),
            seq_cat_data[:, self.decoder_start:self.seq_len-1, :]), 1)
        decoder_out_cont, decoder_out_cat, (hn, cn) = self.dec(decoder_input_cont, decoder_input_cat, hidden=(hn, cn))

        return decoder_out_cont, decoder_out_cat

    def run(self, y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data, criterion):  # noqa: R0913
        """Run model on data and propagate loss.

        Args:
            y (iterable of tensors): List of tensors to calculate model prediction losses
                against, for example, y could be a tuple/list
                (non_seq_cont_data, non_seq_cat_data, seq_cont_data, seq_cat_data)
            seq_cat_data,  seq_cont_data, non_seq_cat_data, non_seq_cont_data, (tensors):
                List of input tensors from the dataloader
            criterion (list of torch criterion objects): The loss function to use while calculating losses
        """
        data = (seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data)
        nonempty_tensors, nonempty_idx = get_nonempty_tensors(data)
        decoder_out = self(*nonempty_tensors, nonempty_idx)

        losses = []
        for idx, criteria in enumerate(criterion):
            losses.append(criteria(decoder_out[..., idx].view(-1,
                                                              decoder_out[..., idx].shape[-1]),
                                   y[..., idx].view(-1, y[..., idx].shape[-1])))
        loss = sum(losses)

        return torch.cat(decoder_out, -1), loss


class TransformerAutoEncoder(nn.Module):  # noqa: W0223
    def __init__(self,
                 unified_encoder,
                 decoder,
                 output_layer):
        """Initialize model with params."""
        super().__init__()

        self.unified_encoder = unified_encoder
        self.decoder = decoder
        self.output_layer = output_layer

    def forward(self, *args):
        """Run a forward pass of model over the data."""
        enc_src, src_mask, src_inp = self.unified_encoder(*args)
        batch_size, _, hid_dim = src_inp.shape
        device = src_inp.device
        # enc_src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        trg_inp = torch.cat((torch.zeros((batch_size, 1, hid_dim), device=device), src_inp[:, :-1, :]), 1)
        decoder_output, attention = self.decoder(trg_inp, enc_src, src_mask)

        # decoder_output = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        outputs = self.output_layer(decoder_output)

        return outputs, attention

    def run(self, y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data, criterion):  # noqa: R0913
        """Run model on data and propagate loss.

        Args:
            y (iterable of tensors): List of tensors to calculate model prediction losses
                against, for example, y could be a tuple/list
                (non_seq_cont_data, non_seq_cat_data, seq_cont_data, seq_cat_data)
            seq_cat_data,  seq_cont_data, non_seq_cat_data, non_seq_cont_data, (tensors):
                List of input tensors from the dataloader
            criterion (list of torch criterion objects): The loss function to use while calculating losses
        """
        data = (seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data)
        nonempty_tensors, nonempty_idx = get_nonempty_tensors(data)
        outputs, _ = self(*nonempty_tensors, nonempty_idx)

        losses = []

        if nonempty_idx[SEQ_CAT_INDEX] != -1:
            for i, out in enumerate(outputs[SEQ_CAT_INDEX]):
                losses.append(criterion[CRITERIA_CAT](out.reshape(-1, out.shape[-1]),
                                                      seq_cat_data[:, :, i].reshape(-1)))

        if nonempty_idx[SEQ_CONT_INDEX] != -1:
            losses.append(criterion[CRITERIA_CONT](outputs[SEQ_CONT_INDEX], seq_cont_data))

        if nonempty_idx[NON_SEQ_CAT_INDEX] != -1:
            for i, out in enumerate(outputs[NON_SEQ_CAT_INDEX]):
                losses.append(criterion[CRITERIA_CAT](out, non_seq_cat_data[:, i]))

        if nonempty_idx[NON_SEQ_CONT_INDEX] != -1:
            losses.append(criterion[CRITERIA_CONT](outputs[NON_SEQ_CONT_INDEX], non_seq_cont_data))

        loss = sum(losses)

        return outputs, loss


class TransformerChurnModel(nn.Module):  # noqa: W0223

    def __init__(self, unified_encoder, mlp):
        """Initialize model with params."""

        super().__init__()

        self.unified_encoder = unified_encoder
        self.mlp = mlp

    def forward(self, *args):
        """Run a forward pass of model over the data."""
        enc_src, _, _ = self.unified_encoder(*args)
        enc_src = enc_src.view(enc_src.shape[0], -1)
        y_pred = self.mlp(enc_src)
        return y_pred

    def run(self, y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data, criterion):  # noqa: R0913
        """Run model on data and propagate loss.

        Args:
            y (iterable of tensors): List of tensors to calculate model prediction losses
                against, for example, y could be a tuple/list
                (non_seq_cont_data, non_seq_cat_data, seq_cont_data, seq_cat_data)
            seq_cat_data,  seq_cont_data, non_seq_cat_data, non_seq_cont_data, (tensors):
                List of input tensors from the dataloader
            criterion (list of torch criterion objects): The loss function to use while calculating losses
        """
        data = (seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data)
        nonempty_tensors, nonempty_idx = get_nonempty_tensors(data)
        y_pred = self(*nonempty_tensors, nonempty_idx)
        loss = criterion(y_pred.flatten(), y)
        return y_pred, loss


class OutputLayer(nn.Module):  # noqa: W0223

    def __init__(self,  # noqa: R0913
                 hid_dim,
                 seq_cont_dim,
                 non_seq_cont_dim,
                 emb_seq_num_classes,
                 emb_non_seq_num_classes):
        """Initialize model with params."""

        super().__init__()

        self.hid_dim = hid_dim
        self.seq_cont_dim = seq_cont_dim
        self.non_seq_cont_dim = non_seq_cont_dim
        self.emb_seq_num_classes = emb_seq_num_classes
        self.emb_non_seq_num_classes = emb_non_seq_num_classes
        self.has_non_seq = len(self.emb_non_seq_num_classes) > 0 or self.non_seq_cont_dim

        self.linear_seq_cat = nn.ModuleList([nn.Linear(hid_dim, num_class) for num_class in emb_seq_num_classes])
        self.linear_seq_cont = nn.Linear(hid_dim, seq_cont_dim) if seq_cont_dim else None
        self.linear_non_seq_cat = nn.ModuleList([nn.Linear(hid_dim, num_class)
                                                 for num_class in emb_non_seq_num_classes])
        self.linear_non_seq_cont = nn.Linear(hid_dim, non_seq_cont_dim) if non_seq_cont_dim else None

    def forward(self, decoder_output):
        """Run a forward pass of model over the data."""

        # decoder_output = [batch size, trg len, hid dim]

        decoder_output_seq = decoder_output[:, :-1, :] if self.has_non_seq else decoder_output
        decoder_output_non_seq = decoder_output[:, -1, :] if self.has_non_seq else None
        device = decoder_output.device

        seq_cat = [output_layer(decoder_output_seq) for output_layer in self.linear_seq_cat]
        seq_cont = self.linear_seq_cont(
            decoder_output_seq) if self.linear_seq_cont else torch.empty((0, 0), device=device)
        non_seq_cat = [output_layer(decoder_output_non_seq) for output_layer in self.linear_non_seq_cat]
        non_seq_cont = self.linear_non_seq_cont(
            decoder_output_non_seq) if self.linear_non_seq_cont else torch.empty((0, 0), device=device)

        return seq_cat, seq_cont, non_seq_cat, non_seq_cont
