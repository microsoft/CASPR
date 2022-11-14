import logging

import numpy as np
import torch.nn as nn

from caspr.models.dense_bn_dropout import DenseBnDropout
from caspr.models.lstm_decoder import LSTM_attention_embedding_decoder
from caspr.models.model_wrapper import LSTMAutoencoder, OutputLayer, TransformerAutoEncoder
from caspr.models.transformer import TransformerDecoder, TransformerEncoder
from caspr.models.unified_encoder import UnifiedEncoder
from caspr.models.unified_transformer_encoder import UnifiedTransformerEncoder

TRANSFORMER = 'TransformerAutoEncoder'
LSTM = 'LSTMAutoencoder'
logger = logging.getLogger(__name__)

class CASPRFactory:

    def __init__(self, cat_cols_, num_activities, cont_cols_, seq_cols_, non_seq_cols_, date_cols_=[], seq_len=15, max_emb_size=25, emb_dims_non_seq=None, emb_dims_seq=None) -> None:
        self.support = {
            TRANSFORMER : self.__create_transformer_autoencoder__,
            LSTM : self.__create_autoencoder__
        }

        if num_activities:
            self.emb_dims_non_seq, self.emb_dims_seq = self.calculate_embedding_dimensions(num_activities, seq_cols=seq_cols_,
                                                                                           non_seq_cols=non_seq_cols_,
                                                                                           max_emb_size=max_emb_size)
        else:
            self.emb_dims_non_seq = emb_dims_non_seq
            self.emb_dims_seq = emb_dims_seq

        self.seq_len = seq_len

        self.non_seq_cat_ = [x for x in cat_cols_ if x in non_seq_cols_]
        self.seq_cat_ = [x for x in cat_cols_ if x in seq_cols_]
        self.non_seq_cont_ = [x for x in cont_cols_ if x in non_seq_cols_]
        self.seq_cont_ = [x for x in cont_cols_+date_cols_ if x in seq_cols_]

        self.seq_cont_dim = len(set(seq_cols_) & set(cont_cols_)) + len(date_cols_)
        self.non_seq_cont_dim = len(set(non_seq_cols_) & set(cont_cols_))
        # Append non seq features to the end of the sequence if exist
        self.adjust_seq_len = seq_len + int(len(non_seq_cols_) > 0)

    @staticmethod
    def calculate_embedding_dimensions(num_activities, seq_cols=None, non_seq_cols=None, max_emb_size=25):
        """Calculate the emb dims for the categorical embedding layer for each categorical variable.

        Args:
            num_activities: number of unique activities for each categorical variable
            seq_cols (list): List of sequential vars
            non_seq_cols (list): List of non-sequential vars
            max_emb_size (Default = 25) : The max size of the embedding layer for a variable
                                        (needed when the possible values are very high)
        """

        # Avoid using empty lists as default values
        seq_cols = [] if seq_cols is None else seq_cols
        non_seq_cols = [] if non_seq_cols is None else non_seq_cols

        cat_seq_dims = [num_activities[c] for c in num_activities.keys() if c in seq_cols]
        cat_non_seq_dims = [num_activities[c] for c in num_activities.keys() if c in non_seq_cols]
        emb_dims_non_seq = [(x, int(np.minimum(max_emb_size, (x + 1) // 2))) for x in cat_non_seq_dims]
        emb_dims_seq = [(x, int(np.minimum(max_emb_size, (x + 1) // 2))) for x in cat_seq_dims]

        return emb_dims_non_seq, emb_dims_seq

    def __create_transformer_autoencoder__(self, device="cuda", HIDDEN_SIZE=64,
                                           NUM_LAYERS_ENC=4,
                                           NUM_LAYERS_DEC=2,
                                           NUM_HEADS_ENC=2,
                                           NUM_HEADS_DEC=4,
                                           PF_DIM_ENC=32,
                                           PF_DIM_DEC=128,
                                           DROPOUT_ENC=0.1,
                                           DROPOUT_DEC=0.1,
                                           EMBEDDING_DROPOUT_SEQUENTIAL=0.1,
                                           EMBEDDING_DROPOUT_NON_SEQUENTIAL=0.1) -> TransformerAutoEncoder:

        enc = TransformerEncoder(hid_dim=HIDDEN_SIZE, n_layers=NUM_LAYERS_ENC, n_heads=NUM_HEADS_ENC,
                                 pf_dim=PF_DIM_ENC, dropout=DROPOUT_ENC, max_length=self.adjust_seq_len)

        dec = TransformerDecoder(hid_dim=HIDDEN_SIZE, n_layers=NUM_LAYERS_DEC, n_heads=NUM_HEADS_DEC,
                                 pf_dim=PF_DIM_DEC, dropout=DROPOUT_DEC, pos_embedding=enc.pos_embedding)

        emb_seq_num_classes = [x for x, _ in self.emb_dims_seq]
        emb_non_seq_num_classes = [x for x, _ in self.emb_dims_non_seq]

        output_layer = OutputLayer(HIDDEN_SIZE, self.seq_cont_dim, self.non_seq_cont_dim,
                                   emb_seq_num_classes, emb_non_seq_num_classes)

        unified_transformer_encoder = UnifiedTransformerEncoder(enc,
                                                                self.emb_dims_non_seq,
                                                                EMBEDDING_DROPOUT_NON_SEQUENTIAL,
                                                                self.emb_dims_seq,
                                                                EMBEDDING_DROPOUT_SEQUENTIAL,
                                                                HIDDEN_SIZE,
                                                                self.seq_cont_dim,
                                                                self.non_seq_cont_dim,
                                                                non_seq_pretrained_embs=None,
                                                                freeze_non_seq_pretrained_embs=True,
                                                                seq_pretrained_embs=None,
                                                                freeze_seq_pretrained_embs=True)

        return TransformerAutoEncoder(unified_transformer_encoder, dec, output_layer).to(device)

    def __create_autoencoder__(self, device="cuda", HIDDEN_SIZE=64,
                               NUM_LAYERS=1,
                               LIN_LAYER_SIZES_NON_SEQUENTIAL=[50, 25],
                               LIN_LAYER_SIZES_SEQUENTIAL=[50, 25],
                               EMBEDDING_DROPOUT_NON_SEQUENTIAL=0.04,
                               LIN_LAYER_DROPOUTS_NON_SEQUENTIAL=[0.0001, 0.01],
                               EMBEDDING_DROPOUT_SEQUENTIAL=0.04,
                               LIN_LAYER_DROPOUTS_SEQUENTIAL=[0.001, 0.01]) -> LSTMAutoencoder:

        output_dim = len(self.seq_cont_)
        num_classes = [x for (x, _) in self.emb_dims_seq]

        # Model objects initialisation
        encoder = UnifiedEncoder(emb_dims_non_seq=self.emb_dims_non_seq,
                                 emb_dropout_non_seq=EMBEDDING_DROPOUT_NON_SEQUENTIAL,
                                 emb_dims_seq=self.emb_dims_seq,
                                 emb_dropout_seq=EMBEDDING_DROPOUT_SEQUENTIAL,
                                 emb_lin_layer_sizes_non_seq=LIN_LAYER_SIZES_NON_SEQUENTIAL,
                                 emb_lin_layer_dropouts_non_seq=LIN_LAYER_DROPOUTS_NON_SEQUENTIAL,
                                 emb_lin_layer_sizes_seq=LIN_LAYER_SIZES_SEQUENTIAL,
                                 emb_lin_layer_dropouts_seq=LIN_LAYER_DROPOUTS_SEQUENTIAL,
                                 lstm_hidden_size=HIDDEN_SIZE,
                                 output_size=output_dim,
                                 seq_len=self.seq_len,
                                 non_seq_cont_count=len(self.non_seq_cont_),
                                 seq_cat_count=len(self.seq_cat_),
                                 seq_cont_count=len(self.seq_cont_),
                                 non_seq_cat_count=len(self.non_seq_cat_))

        input_dim = int(encoder.seq_cont_count + encoder.no_of_embs_seq)

        decoder = LSTM_attention_embedding_decoder(input_dim=input_dim,
                                                   hidden_size=HIDDEN_SIZE,
                                                   num_layers=NUM_LAYERS,
                                                   output_dim=output_dim,
                                                   num_classes=num_classes)

        mlp_non_seq_cat_list = []

        for non_seq_cat, _ in self.emb_dims_non_seq:
            mlp_non_seq_cat_list.append(DenseBnDropout(LIN_LAYER_SIZES_NON_SEQUENTIAL+[
                                        non_seq_cat], LIN_LAYER_DROPOUTS_NON_SEQUENTIAL+[0], HIDDEN_SIZE))
        mlp_non_seq_cont = DenseBnDropout(
            LIN_LAYER_SIZES_NON_SEQUENTIAL, LIN_LAYER_DROPOUTS_NON_SEQUENTIAL, HIDDEN_SIZE)

        autoenc = LSTMAutoencoder(encoder, mlp_non_seq_cat_list, mlp_non_seq_cont, decoder).to(device)

        return autoenc

    def create(self, architecture: str, device="cuda", **hyperparams) -> nn.Module:
        if architecture not in self.support:
            raise ValueError("Unknown architecture specified. Model Factory currently supports: %s Requested: %s" % (str(self.support.keys()), architecture))

        constructor_f = self.support[architecture]

        logger.info("Initializing CASPR with %s architecture. Hyperparams provided: %s" % (architecture, hyperparams))

        return constructor_f(device, **hyperparams)
