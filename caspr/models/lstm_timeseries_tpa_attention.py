# coding: utf-8
"""TPA attention based LSTM encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_TPA_attention_timeseries(nn.Module):  # noqa: W0223
    """TPA attention based LSTM encoder."""

    def __init__(self,  # noqa: R0913
                 emb_dims_non_seq,
                 emb_dims_seq,
                 lin_layer_sizes_non_sequential,
                 lin_layer_sizes_sequential,
                 non_seq_cont_count,
                 hidden_size,
                 output_size,
                 emb_dropout_non_seq,
                 lin_layer_dropouts_non_sequential,
                 emb_dropout_seq,
                 lin_layer_dropouts_sequential,
                 lin_layer_sizes_fin,
                 lin_layer_dropouts_fin,
                 seq_len, input_dim, device):
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
        self.context_vector_size = hidden_size
        self.output_dim = output_size

        if self.no_of_embs_non_seq != 0:
            first_lin_layer = nn.Linear(self.no_of_embs_non_seq, lin_layer_sizes_non_sequential[0])
            self.lin_layersnon_sequential = nn.ModuleList([first_lin_layer] +
                                                          [nn.Linear(lin_layer_sizes_non_sequential[i],
                                                                     lin_layer_sizes_non_sequential[i + 1])
                                                           for i in range(len(lin_layer_sizes_non_sequential) - 1)])
            for lin_layer in self.lin_layersnon_sequential:
                nn.init.kaiming_normal_(lin_layer.weight.data)

            self.emb_dropout_layernon_sequential = nn.Dropout(emb_dropout_non_seq)
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

        for lin_layer in self.lin_layer_non_sequential:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # definitions for data parsing
        # primarily required to make sure embeddings are used for categorical data

        # LSTM layer
        self.lstm_layer = nn.LSTM(
            self.input_dim + lin_layer_sizes_non_sequential[-1], self.hidden_size, batch_first=True)

        # Linear Layers post LSTM
        self.lin_layer_lstm_to_dense = nn.Linear(
            self.hidden_size, self.hidden_size)

        # TPA attention
        self.convolution_filters = nn.ModuleList([nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=self.seq_len) for i in range(hidden_size)])
        self.tpa_linear = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False)

        self.tpa_hiddent_linear = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False)
        self.tpa_context_linear = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False)
        # Final MLP
        first_fin_layer = nn.Linear(self.hidden_size + self.context_vector_size +
                                    self.no_of_embs_non_seq + self.non_seq_cont_count, lin_layer_sizes_fin[0])

        self.lin_layers_final = nn.ModuleList([first_fin_layer] +
                                              [nn.Linear(lin_layer_sizes_fin[i],
                                                         lin_layer_sizes_fin[i + 1])
                                               for i in range(len(lin_layer_sizes_fin) - 1)])
        for lin_layer in self.lin_layers_final:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # final dropout and batch norm layers for final prediction
        self.dropout_layers_final = nn.ModuleList(
            [nn.Dropout(size) for size in lin_layer_dropouts_fin])
        self.bn_layers_final = nn.ModuleList(
            [nn.BatchNorm1d(size) for size in lin_layer_sizes_fin])

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes_fin[-1], output_size)

    def forward(self, seq_cont_data, seq_cat_data, non_seq_cat_data, non_seq_cont_data): # noqa : R0914
        """Run a forward pass of model over the data."""

        if self.no_of_embs_non_seq != 0:
            non_seq_cat_data = non_seq_cat_data.type(
                torch.LongTensor).to(self.device)
            #   across all rows and column i -  useful for batches
            non_seq_cat_inp = [emb_layer(non_seq_cat_data[:, i])
                               for i, emb_layer in enumerate(self.emb_layers)]
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
            seq_cat_inp = torch.cat(seq_cat_inp, 2)
            seq_cat_inp = self.emb_dropout_layer_seq(seq_cat_inp)

            seq_cat_inp_emb = seq_cat_inp

        seq_cat_inp_emb = seq_cat_inp_emb.to(self.device)

        seq_data = torch.cat([seq_cat_inp_emb, seq_cont_data], 2)

#       now the sequential data ------------------------------
        inp_tens = seq_data

        temp_batch_size = inp_tens.size()[0]

        h0 = torch.zeros(1, temp_batch_size, self.hidden_size).to(
            self.device).requires_grad_()
        c0 = torch.zeros(1, temp_batch_size, self.hidden_size).to(
            self.device).requires_grad_()

        output, (hn, _) = self.lstm_layer(inp_tens, (h0, c0))
        hn = hn.to(self.device)
        # passes through the embedding layer to generate the required embeddings
        # output shape batch_size * seq_len * hidden_size
        # output[:,:,i] shape batch_size * seq_len - 1st row of H matrix
        hc = torch.zeros(temp_batch_size, self.hidden_size,
                         self.hidden_size).to(self.device)

        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                hc[:, i, j] = self.convolution_filters[j](
                    output[:, :, i].unsqueeze(1)).squeeze()

        alpha = torch.zeros(temp_batch_size, self.hidden_size).to(self.device)

        for i in range(self.hidden_size):
            temp1 = self.tpa_linear(hc[:, i]).unsqueeze(1)
            temp2 = hn.squeeze().unsqueeze(2)
            temp = torch.bmm(temp1, temp2)
            alpha[:, i] = F.sigmoid(temp).squeeze()

        vt = torch.zeros(temp_batch_size, self.hidden_size).to(self.device)
        for i in range(self.hidden_size):
            temp = torch.bmm(alpha[:, i].unsqueeze(1).unsqueeze(
                2), hc[:, i].unsqueeze(1)).squeeze()
            vt += temp

        htprime = self.tpa_hiddent_linear(hn) + self.tpa_context_linear(vt)

        seq_inp = self.lin_layer_lstm_to_dense(hn)
        seq_inp = seq_inp.reshape(seq_inp.size()[1], seq_inp.size()[2])
        htprime = htprime.squeeze()

        # Linear mlp for prediction
        # fin_input = torch.cat((seq_inp, htprime), 1)
        fin_input = torch.cat((non_seq_inp, seq_inp, htprime), 1)

        x = fin_input
        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers_final, self.dropout_layers_final,
                                                      self.bn_layers_final):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = F.relu(self.output_layer(x))

        return x, fin_input
