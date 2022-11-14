def estimate_linear_parameters(input_dim, output_dim, bias=True):
    if input_dim > 0 and bias:
        input_dim += 1
    return input_dim * output_dim


def estimate_embedding_parameters(df, cat_cols_, max_emb_dim):
    emb_num_classes = [df.select(c).distinct().count() for c in cat_cols_]
    emb_dims = [(x, int(min(max_emb_dim, (x + 1) // 2))) for x in emb_num_classes]
    emb_size = sum([d for _, d in emb_dims])
    emb_num_param = sum([estimate_linear_parameters(v, d, bias=False) for v, d in emb_dims])
    return emb_num_param, emb_size, emb_num_classes


def estimate_transformer_parameters(hidden_dim, seq_len, pf_dim, num_layers, is_encoder=True):
    pos_emb_num_param = estimate_linear_parameters(seq_len, hidden_dim, bias=False) if is_encoder else 0
    layer_norm_num_param = hidden_dim * 2
    attn_num_param = estimate_linear_parameters(hidden_dim, hidden_dim) * 4
    layer_norm_count = 2 if is_encoder else 3
    attn_count = 1 if is_encoder else 2
    pf_num_param = estimate_linear_parameters(hidden_dim, pf_dim) + estimate_linear_parameters(pf_dim, hidden_dim)
    transformer_num_param = pos_emb_num_param + \
        (layer_norm_num_param * layer_norm_count + attn_num_param * attn_count + pf_num_param) * num_layers
    return transformer_num_param


def estimate_output_parameters(hidden_dim, emb_num_classes, cont_dim):
    output_num_param_cat = sum([estimate_linear_parameters(hidden_dim, v) for v in emb_num_classes])
    output_num_param_cont = estimate_linear_parameters(hidden_dim, cont_dim)
    output_num_param = output_num_param_cat + output_num_param_cont
    return output_num_param


def estimate_transformer_autoencoder_parameters(df, seq_cat_, seq_cont_, non_seq_cat_, non_seq_cont_,
                                                hidden_dim, pf_dim_enc, pf_dim_dec, num_layers_enc,
                                                num_layers_dec, seq_len, max_emb_dim=30):  
    emb_num_param_seq, emb_size_seq, emb_num_classes_seq = estimate_embedding_parameters(df, seq_cat_, max_emb_dim)
    emb_num_param_non_seq, emb_size_non_seq, emb_num_classes_non_seq = estimate_embedding_parameters(df, non_seq_cat_, max_emb_dim)
    emb_num_param = emb_num_param_seq + emb_num_param_non_seq

    seq_cont_dim = len(seq_cont_)
    non_seq_cont_dim = len(non_seq_cont_)
    non_seq_dim = emb_size_non_seq + non_seq_cont_dim
    linear_num_param_seq = estimate_linear_parameters(seq_cont_dim + emb_size_seq, hidden_dim)
    linear_num_param_non_seq = estimate_linear_parameters(non_seq_dim, hidden_dim)
    linear_num_param = linear_num_param_seq + linear_num_param_non_seq

    adjust_seq_len = seq_len + int(non_seq_dim > 0)
    enc_num_param = estimate_transformer_parameters(hidden_dim, adjust_seq_len, pf_dim_enc, num_layers_enc)
    dec_num_param = estimate_transformer_parameters(hidden_dim, adjust_seq_len, pf_dim_dec, num_layers_dec, is_encoder=False)
    transformer_num_param = enc_num_param + dec_num_param

    output_num_param_seq = estimate_output_parameters(hidden_dim, emb_num_classes_seq, seq_cont_dim)
    output_num_param_non_seq = estimate_output_parameters(hidden_dim, emb_num_classes_non_seq, non_seq_cont_dim)
    output_num_param = output_num_param_seq + output_num_param_non_seq

    num_param = emb_num_param + linear_num_param + transformer_num_param + output_num_param
    return num_param
