"""CASPR LSTM model."""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from caspr.models.attention_mechanisms import BahdanauAttention, MultiHeadAttentionLSTMWrapper
from caspr.models.convolutional_aggregation import ConvAggregation
from caspr.models.dense_bn_dropout import DenseBnDropout
from caspr.models.embedding_layer import CategoricalEmbedding
from caspr.models.multi_layer_lstm import MultiLayerLSTM

warnings.simplefilter('ignore')


SEQ_CAT_INDEX = 0
SEQ_CONT_INDEX = 1
NON_SEQ_CAT_INDEX = 2
NON_SEQ_CONT_INDEX = 3


class UnifiedEncoder(nn.Module):  # noqa: R0902, W0223
    """Encapsulates the basic structure to run most of our models.

    It checks the various conditions for the presence
    or the absence of data and is compatible with functionalities like
        1. Usage of pretrained embedding vectors
        2. Multi-Layered LSTM use
        3. Convolutional Aggregation for data
        4. Self-Multi-Head and Bahdanau Attention (when number of heads = 1, Bahdanau is used by default)

    In this new edition, it is compatible with the DLExplainer module and should be used if
    explainability is a requirement
    """

    def __init__(self,  # noqa: R0912, R0913, R0914, R0915
                 emb_dims_non_seq,
                 emb_dropout_non_seq,
                 emb_dims_seq,
                 emb_dropout_seq,
                 emb_lin_layer_sizes_non_seq,
                 emb_lin_layer_dropouts_non_seq,
                 emb_lin_layer_sizes_seq,
                 emb_lin_layer_dropouts_seq,
                 lstm_hidden_size,
                 output_size,
                 seq_len,
                 non_seq_cont_count,
                 seq_cat_count,
                 seq_cont_count,
                 non_seq_cat_count,
                 attention_heads=1,
                 non_seq_pretrained_embs=None,
                 freeze_non_seq_pretrained_embs=True,
                 seq_pretrained_embs=None,
                 freeze_seq_pretrained_embs=True,
                 lstm_num_layers=1,
                 lstm_bidirectional=False,
                 use_conv_agg=False,
                 kernel_size=(3, 3),
                 max_pool_size=(2, 2),
                 stride=(2, 2)):
        """Initialize of all the variables and the layers depending on the arguments supplied.

        Args:
            emb_dims_non_seq = (List of tuples (x, y)) where x is the vocab size and y is the number of dimensions
                for the respective embedding layer for every non_sequential categorical variable
            emb_dropout_non_seq = (Float) Dropout value of a layer used after the embedding layer - non_sequential
            emb_dims_seq = (List of tuples (x, y)) where x is the vocab size and y is the number of dimensions for the
                respective embedding layer for every sequential categorical variable
            emb_dropout_seq = (Float) Dropout value of a layer used after the embedding layer - sequential
            emb_lin_layer_sizes_non_seq = (List of integers) determining the sizes of the stacked linear layers
                used just after the embedding layers to learn better representations for non_sequential
                categorical variables
            emb_lin_layer_dropouts_non_seq = (List of float) values determining the p values in the dropout
                layers between linear layers
            emb_lin_layer_sizes_seq = (List of integers) determining the sizes of the stacked linear layers
                used just after the embedding layers to learn better representations for sequential
                categorical variables
            emb_lin_layer_dropouts_seq = (List of float) values determining the p values in the dropout
                layers between linear layers
            lstm_hidden_size = (Integer) determining the Hidden size of the LSTM layer used to train the sequence model
            output_size = (Integer) Size of the final embedded output by the encoder.
            seq_len = (Integer) determining the length of the sequence in input
            non_seq_cont_count = (Integer) Number of non_sequential continuous variables
            seq_cat_count = (Integer) Number of sequential categorical variables
            seq_cont_count = (Integer) Number of sequential continuous variables
            non_seq_cat_count = (Integer) Number of non_sequential categorical variables
            attention_heads = (Integer: Default = 1) Describes the number of attention heads being used after the LSTM.
                When 0 means that attention is not being used.
                When = 1 uses Bahdanau attention by default and
                When > 1 uses Multi-Head self-attention
            non_seq_pretrained_embs = (List of Tensors: Default = None) To be used as pretrained embeddings
                in the embedding layers
            freeeze_non_seq_pretrained_embs = (Boolean: Default = True) Determines if the pretrained embeddings
                are to be left untouched during backprop
            seq_pretrained_embs = (List of Tensors: Default = None) To be used as pretrained embeddings in the
                embedding layers,
            freeeze_seq_pretrained_embs = (Boolean: Default = True) Determines if the pretrained embeddings
                are to be left untouched during backprop
            lstm_num_layers = (Integer: Default = 1) The number of stacked LSTM layers used
            lstm_bidirectional = (Boolean: Default = False) Determines if the LSTM used is bidirectional
            use_conv_agg = (Boolean: Default = False) Determines  if Convolutional aggregation is to be used in
                the model or not
            kernel_size = (Tuple of Integers : Default = (3,3)) Determines the kernel size of the cnn aggregator
            max_pool_size = (Tuple of Integers : Default = (2, 2)) Determines the max_pool size of the cnn aggregator
            stride = (Tuple of Integers : Default = (2, 2)) Determines the stride of the cnn aggregator
        """
        super().__init__()

        self._explain = False
        self.non_seq_emb_layers = CategoricalEmbedding(emb_dims=emb_dims_non_seq, emb_dropout=emb_dropout_non_seq,
                                                       pretrained_vecs=non_seq_pretrained_embs,
                                                       freeze_pretrained=freeze_non_seq_pretrained_embs)
        self.seq_emb_layers = CategoricalEmbedding(emb_dims=emb_dims_seq, emb_dropout=emb_dropout_seq, is_seq=True,
                                                   pretrained_vecs=seq_pretrained_embs,
                                                   freeze_pretrained=freeze_seq_pretrained_embs)

        self.no_of_embs_non_seq = np.sum([y for x, y in emb_dims_non_seq])
        self.no_of_embs_seq = np.sum([y for x, y in emb_dims_seq])

        self.non_seq_cat_final_size = 0
        self.seq_len = seq_len
        self.hidden_size = lstm_hidden_size
        self.context_vector_size = lstm_hidden_size
        self.output_dim = output_size
        self.num_layers = lstm_num_layers
        self.num_directions = 2 if lstm_bidirectional else 1

        self.seq_cat_count = seq_cat_count
        self.seq_cont_count = seq_cont_count
        self.non_seq_cat_count = non_seq_cat_count
        self.non_seq_cont_count = non_seq_cont_count
        self.attention_heads = attention_heads

        self.use_conv_agg = use_conv_agg

        # Linear Layers for non_seq_data parallel to LSTM
        if self.no_of_embs_non_seq != 0:
            self.emb_lin_layer_non_seq = DenseBnDropout(
                lin_layer_sizes=emb_lin_layer_sizes_non_seq,
                lin_layer_dropouts=emb_lin_layer_dropouts_non_seq, input_size=self.no_of_embs_non_seq)
            self.non_seq_cat_final_size = emb_lin_layer_sizes_non_seq[-1]

        # LSTM layer
        if self.no_of_embs_seq != 0:
            self.emb_lin_layer_seq = DenseBnDropout(
                lin_layer_sizes=emb_lin_layer_sizes_seq,
                lin_layer_dropouts=emb_lin_layer_dropouts_seq, input_size=self.no_of_embs_seq)

        # LSTM layer
        if self.no_of_embs_seq != 0:
            self.emb_lin_layer_seq = DenseBnDropout(
                lin_layer_sizes=emb_lin_layer_sizes_seq,
                lin_layer_dropouts=emb_lin_layer_dropouts_seq, input_size=self.no_of_embs_seq)
            self.lstm_inp_size = emb_lin_layer_sizes_seq[-1] + seq_cont_count
        else:
            self.lstm_inp_size = seq_cont_count

        if use_conv_agg and seq_len >= kernel_size[0] and \
            (min(1, seq_cat_count)*emb_lin_layer_sizes_seq[-1] + seq_cont_count) >= kernel_size[1] and \
                int((min(1, seq_cat_count)*emb_lin_layer_sizes_seq[-1] + seq_cont_count -
                     (kernel_size[1] - 1))/stride[1]) >= max_pool_size[1] and \
                int((seq_len - (kernel_size[0] - 1))/stride[0]) >= max_pool_size[0]:
            # kernel_size[0] -> size of kernel along sequence dimension, hence must be <= seq_len
            # kernel_size[1] -> size of kernel along features dimension, hence must be <= net size of input features
            # int((min(1, seq_cat_count)*emb_lin_layer_sizes_seq[-1]
            #  + seq_cont_count - (kernel_size[i] - 1))/stride[i])
            # is the formula to calculate the final size of dimension i after the CNN filter is applied
            # the above size should be >= max_pool[i] for pooling
            self.conv_agg = ConvAggregation(
                kernel_size=kernel_size, stride=stride, max_pool_size=max_pool_size, dropout_size=0.4)
            self.lstm_inp_size = int((int((min(1, seq_cat_count)*emb_lin_layer_sizes_seq[-1] + seq_cont_count - (
                kernel_size[1] - 1) - 1)/stride[1] + 1)) / max_pool_size[1])
        else:
            self.use_conv_agg = False

        if self.lstm_inp_size > 0:
            self.lstm_layer = MultiLayerLSTM(input_size=self.lstm_inp_size, hidden_size=self.hidden_size,
                                             num_layers=self.num_layers, bidirectional=lstm_bidirectional, dropout=0.4)

        # Attention
        if self.attention_heads > 0:
            if self.attention_heads == 1:
                self.bahdanau_attention_layer = BahdanauAttention(self.hidden_size, self.num_directions)
            else:
                n_head = self.attention_heads
                d_model = self.hidden_size
                self.multi_head_attention_layer = MultiHeadAttentionLSTMWrapper(n_head, d_model, dropout=0.1)

        if self.attention_heads > 0:
            self.output_layer = nn.Linear(int(self.num_directions*self.hidden_size + self.context_vector_size +
                                              self.non_seq_cat_final_size + self.non_seq_cont_count),
                                          int(self.hidden_size))
        else:
            self.output_layer = nn.Linear(int(self.num_directions*self.hidden_size +
                                              self.non_seq_cat_final_size + self.non_seq_cont_count),
                                          int(self.hidden_size))
        nn.init.kaiming_normal_(self.output_layer.weight.data)

    def forward(self, *args):  # noqa: R0912, R0914
        """Forward function accepts multiple arguments.

        The last argument is always a list of indices representing the index (if data present)
        with -1 in places for the absence of data The indices are used to partition the data into 4 types
        - seq_cat, seq_cont, non_seq_cat, non_seq_cont
        """
        nonempty_idx = args[-1]
        data_exists = list(map(lambda x: x != -1, nonempty_idx))
        device = args[0].device
        batch_size = args[0].shape[0]

        seq_cat_data = args[nonempty_idx[SEQ_CAT_INDEX]] if data_exists[SEQ_CAT_INDEX] else torch.empty(batch_size, 0, 0, device=device)
        seq_cont_data = args[nonempty_idx[SEQ_CONT_INDEX]] if data_exists[SEQ_CONT_INDEX] else torch.empty(batch_size, 0, 0, device=device)
        non_seq_cat_data = args[nonempty_idx[NON_SEQ_CAT_INDEX]] if data_exists[NON_SEQ_CAT_INDEX] else torch.empty(batch_size, 0, device=device)
        non_seq_cont_data = args[nonempty_idx[NON_SEQ_CONT_INDEX]] if data_exists[NON_SEQ_CONT_INDEX] else torch.empty(batch_size, 0, device=device)

        if self.no_of_embs_non_seq != 0:
            non_seq_cat_inp = self.non_seq_emb_layers(non_seq_cat_data)
            non_seq_inp = self.emb_lin_layer_non_seq(non_seq_cat_inp)

            if self.non_seq_cont_count != 0:
                non_seq_inp = torch.cat((non_seq_inp.type(torch.FloatTensor).to(device),
                                         non_seq_cont_data.type(torch.FloatTensor).to(device)), 1)
        else:
            if self.non_seq_cont_count != 0:
                non_seq_inp = non_seq_cont_data.to(device)
            else:
                non_seq_inp = torch.Tensor().to(device)

        if self.no_of_embs_seq != 0:
            seq_cat_inp = self.seq_emb_layers(seq_cat_data)
            seq_inp = self.emb_lin_layer_seq(seq_cat_inp)
            if self.seq_cont_count != 0:
                seq_inp = torch.cat((seq_inp.type(torch.FloatTensor).to(device),
                                     seq_cont_data.type(torch.FloatTensor).to(device)), 2)

        elif self.seq_cont_count != 0:
            seq_inp = seq_cont_data.type(torch.FloatTensor).to(device)

        if self.no_of_embs_seq + self.seq_cont_count > 0:

            if self.use_conv_agg:
                seq_inp = self.conv_agg(seq_inp)

            output, (_, cn), seq_inp = self.lstm_layer(seq_inp)

            if self.attention_heads > 0:
                if self.attention_heads == 1:
                    context_vector = self.bahdanau_attention_layer(output)
                    context_vector = context_vector.reshape(context_vector.size()[0], context_vector.size()[2])
                else:
                    context_vector = self.multi_head_attention_layer(output, output, output)

                fin_input = torch.cat((seq_inp, context_vector), 1)
            else:
                fin_input = seq_inp

            if self.no_of_embs_non_seq + self.non_seq_cont_count > 0:
                fin_input = torch.cat((non_seq_inp.type(torch.FloatTensor).to(device), fin_input), 1)
        else:
            fin_input = non_seq_inp

        fin_output = F.relu(self.output_layer(fin_input))

        if self._explain:
            return fin_output
        return output, (fin_output, cn)

    @property
    def explain(self):
        """Getter for explain."""

        return self._explain

    def set_explain(self, value):
        """Setter for explain."""

        self._explain = value
