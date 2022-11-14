"""Bahdanau attention based LSTM encoder."""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import caspr.models

warnings.simplefilter('ignore')


class LSTM_attention_embedding_encoder_sequence(nn.Module):  # noqa: W0223
    """Luong/Bahdanau attention based LSTM encoder."""

    def __init__(self,  # noqa: R0913, R0914
                 emb_dims_non_seq,
                 emb_dims_seq,
                 lin_layer_sizes_non_sequential,
                 lin_layer_sizes_sequential,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 output_size,
                 emb_dropout_non_seq,
                 lin_layer_dropouts_non_sequential,
                 emb_dropout_seq,
                 lin_layer_dropouts_sequential,
                 lin_layer_sizes_fin,
                 lin_layer_dropouts_fin,  # noqa: W0613
                 seq_len, input_dim,
                 non_seq_cont_count, seq_cat_count, seq_cont_count, non_seq_cat_count,
                 device):
        """Initialise the pytorch LSTM layer.

        Args:
            emb_dims_non_seq, emb_dims_seq (list of int tuples):
                List of category dimension and corresponding embedding size.
            lin_layer_sizes_non_sequential,  lin_layer_sizes_sequential (list of int tuples):
                List of [m1*m2] tuples for embedding dimension reduction and non-linearity
            emb_dropout_non_seq, emb_dropout_seq (float): dropout values for embedding layers
            lin_layer_dropouts_non_seq, lin_layer_dropouts_seq (list of float):
                dropout values for linear layers corresponding to embedding layers
            hidden_size (int): Size of the hidden state
            num_layers (int): Number of stacked LSTM layers
            bidirectional (bool):  Flag for bi/uni LSTM
            output_size (int): Size of the final output layer
            lin_layer_sizes_fin (list of int tuples):
                List of [m1*m2] tuples for non-linear combination of sequential and nonsequential inputs
            seq_len (int): Length of input Sequence
        """
        super().__init__()

        self.device = device
        self.non_seq_emb_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in emb_dims_non_seq])
        self.seq_emb_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in emb_dims_seq])
        self.no_of_embs_non_seq = sum([y for x, y in emb_dims_non_seq])
        self.no_of_embs_seq = sum([y for x, y in emb_dims_seq])
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.non_seq_cont_count = non_seq_cont_count
        self.non_seq_cat_count = non_seq_cat_count
        self.context_vector_size = hidden_size
        self.output_dim = output_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.seq_cat_count = seq_cat_count
        self.seq_cont_count = seq_cont_count
        self.non_seq_cat_count = non_seq_cat_count
        self.non_seq_cont_count = non_seq_cont_count

        # Linear Layers for non_seq_data parallel to LSTM
        if self.no_of_embs_non_seq != 0:
            first_lin_layer = nn.Linear(self.no_of_embs_non_seq, lin_layer_sizes_non_sequential[0])
            self.lin_layersnon_sequential = nn.ModuleList([first_lin_layer] +
                                                          [nn.Linear(lin_layer_sizes_non_sequential[i],
                                                                     lin_layer_sizes_non_sequential[i + 1])
                                                           for i in range(len(lin_layer_sizes_non_sequential) - 1)])
            for lin_layer in self.lin_layersnon_sequential:
                nn.init.kaiming_normal_(lin_layer.weight.data)

            self.emb_dropout_layer_non_sequential = nn.Dropout(emb_dropout_non_seq)
            self.dropout_layersnon_sequential = nn.ModuleList(
                [nn.Dropout(size) for size in lin_layer_dropouts_non_sequential])
            self.bn_layersnon_sequential = nn.ModuleList(
                [nn.BatchNorm1d(size) for size in lin_layer_sizes_non_sequential])

        # Linear Layers for seq_cat_data
        if self.no_of_embs_seq != 0:
            first_lin_layer_seq = nn.Linear(self.no_of_embs_seq, lin_layer_sizes_sequential[0])
            self.lin_layers_seq = nn.ModuleList([first_lin_layer_seq] +
                                                [nn.Linear(lin_layer_sizes_sequential[i],
                                                           lin_layer_sizes_sequential[i + 1])
                                                 for i in range(len(lin_layer_sizes_sequential) - 1)])
            for lin_layer in self.lin_layers_seq:
                nn.init.kaiming_normal_(lin_layer.weight.data)

            self.emb_dropout_layer_seq = nn.Dropout(emb_dropout_seq)
            self.dropout_layers_seq = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts_sequential])
            self.bn_layers_seq = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes_sequential])

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes_fin[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # LSTM layer
        self.lstmLayer = nn.LSTM(
            self.input_dim + lin_layer_sizes_sequential[-1],
            self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional)
        # self.lstmLayer = nn.LSTM(
        #     self.input_dim+self.no_of_embs_seq,
        #     self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional)

        # Linear Layers post LSTM
        self.lin_layer_lstm_to_dense = nn.Linear(
            self.num_directions*self.hidden_size, self.hidden_size)

        # Attention
        self.bahdanau_attention_layer = caspr.models.attention_mechanisms.BahdanauAttention(
            self.hidden_size, self.num_directions)

        # self.fc_encoder = nn.Linear(
        #     self.num_directions*self.hidden_size, self.hidden_size, bias=False)

        # self.attnHidden = nn.Linear(self.hidden_size, 1)

        self.fin_layer = nn.Linear(
            self.num_directions*self.hidden_size +
            self.context_vector_size + self.no_of_embs_non_seq + self.non_seq_cont_count, hidden_size)
#         self.fin_layer = nn.Linear(
#             self.num_directions*self.hidden_size + self.context_vector_size , hidden_size)

    def forward(self, input_tensor):  # noqa : R0914
        """Run a forward pass of model over the data."""
        seq_cat_index = self.seq_len * self.seq_cat_count
        seq_cont_index = seq_cat_index + self.seq_len * self.seq_cont_count
        non_seq_cat_index = seq_cont_index + self.non_seq_cat_count
        non_seq_cont_index = non_seq_cat_index + self.non_seq_cont_count

        seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data = input_tensor[:, :seq_cat_index], \
            input_tensor[:, seq_cat_index: seq_cont_index], \
            input_tensor[:, seq_cont_index: non_seq_cat_index], \
            input_tensor[:, non_seq_cat_index: non_seq_cont_index]
        seq_cat_data = seq_cat_data.type(torch.LongTensor)
        seq_cat_data = seq_cat_data.reshape(
            seq_cat_data.shape[0], self.seq_len, int(seq_cat_data.shape[1]/self.seq_len))
        seq_cont_data = seq_cont_data.reshape(
            seq_cont_data.shape[0], self.seq_len, int(seq_cont_data.shape[1]/self.seq_len))

        if self.no_of_embs_non_seq != 0:
            non_seq_cat_data = non_seq_cat_data.type(
                torch.LongTensor).to(self.device)
            #   across all rows and column i -  useful for batches
            non_seq_cat_inp = [emb_layer(non_seq_cat_data[:, i])
                               for i, emb_layer in enumerate(self.non_seq_emb_layers)]
            non_seq_cat_inp = torch.cat(non_seq_cat_inp, 1)
            non_seq_cat_inp = self.emb_dropout_layer_non_sequential(non_seq_cat_inp)
            if self.non_seq_cont_count != 0:
                non_seq_inp = torch.cat((non_seq_cat_inp.type(torch.FloatTensor).to(
                    self.device), non_seq_cont_data.type(torch.FloatTensor).to(self.device)), 1)
            else:
                non_seq_inp = non_seq_cat_inp.type(torch.FloatTensor).to(self.device)
        elif self.non_seq_cont_count != 0:
            non_seq_inp = non_seq_cont_data.type(torch.FloatTensor).to(self.device)

        if self.no_of_embs_seq != 0:
            seq_cat_data = seq_cat_data.type(
                torch.LongTensor).to(self.device)
            #   across all rows and column i -  useful for batches
            seq_cat_inp = [emb_layer(seq_cat_data[:, :, i])
                           for i, emb_layer in enumerate(self.seq_emb_layers)]
            # shape = batchsize * seq_len * 16(emb size)
            seq_cat_inp = torch.cat(seq_cat_inp, 2)
            seq_cat_inp = self.emb_dropout_layer_seq(seq_cat_inp)
            seq_cat_inp_emb = seq_cat_inp
            for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers_seq,
                                                          self.dropout_layers_seq, self.bn_layers_seq):
                seq_cat_inp_emb = F.relu(lin_layer(seq_cat_inp_emb))
                seq_cat_inp_emb = torch.cat([bn_layer(seq_cat_inp_emb[:, i, :]).unsqueeze(1)
                                             for i in range(self.seq_len)], 1)
                seq_cat_inp_emb = dropout_layer(seq_cat_inp_emb)

        seq_cat_inp_emb = seq_cat_inp_emb.to(self.device)
        # shape seq_cat = batchsize * seq_len * emb size/lin_layers_seq[-1].shape
        # shape seq_cont = batchsize * seq_len * data

        seq_data = torch.cat([seq_cat_inp_emb, seq_cont_data], 2)

        # now the sequential data
        inp_tens = seq_data

        temp_batch_size = inp_tens.size()[0]

        h0 = torch.zeros(self.num_directions*self.num_layers, temp_batch_size, self.hidden_size).to(
            self.device).requires_grad_()
        c0 = torch.zeros(self.num_directions*self.num_layers, temp_batch_size, self.hidden_size).to(
            self.device).requires_grad_()

        output, (hn, cn) = self.lstmLayer(inp_tens, (h0, c0))
        # passes through the embedding layer to generate the required embeddings
        # attention weight calculation

        # tempX = torch.tanh(self.fc_encoder(output))
        # alignment_scores = self.attnHidden(tempX)
        # attn_weights = F.softmax(alignment_scores, dim=1)
        # attn_weights = attn_weights.permute(0, 2, 1)
        # context_vector = torch.bmm(attn_weights, output)

        context_vector = self.bahdanau_attention_layer(output)

        hn = hn.view(self.num_layers, self.num_directions, -
                     1, self.hidden_size).to(self.device)
        cn_ = cn.view(self.num_layers, self.num_directions, -
                      1, self.hidden_size).to(self.device)
        if self.num_directions > 1:
            seq_inp = self.lin_layer_lstm_to_dense(torch.cat(
                [hn[self.num_layers-1, 0], hn[self.num_layers-1, -1]], 1).unsqueeze(0))
        else:
            seq_inp = self.lin_layer_lstm_to_dense(
                hn[self.num_layers-1, 0]).unsqueeze(0)

        seq_inp = seq_inp.reshape(seq_inp.size()[1], seq_inp.size()[2])

        context_vector = context_vector.reshape(
            context_vector.size()[0], context_vector.size()[2])

        fin_input = torch.cat((non_seq_inp, seq_inp, context_vector), 1)
#         fin_input = torch.cat((seq_inp, context_vector), 1)

        hn_ = F.relu(self.fin_layer(fin_input))

        return output, (hn_, cn_[self.num_layers-1, 0, :, :].unsqueeze(0))
