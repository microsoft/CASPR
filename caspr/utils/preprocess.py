"""Utility functions for Pandas dataframe preprocessing."""

# coding: utf-8
import logging
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTENC
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter('ignore')
logger = logging.getLogger(__name__)


def generate_combination_of_cols(cat_cols, cont_cols, seq_cols, non_seq_cols):
    """Generate combinations of columns."""

    seq_cat_cols = [item for item in seq_cols if item in cat_cols]
    seq_cont_cols = [item for item in seq_cols if item in cont_cols]
    non_seq_cat_cols = [item for item in non_seq_cols if item in cat_cols]
    non_seq_cont_cols = [item for item in non_seq_cols if item in cont_cols]

    return seq_cat_cols, seq_cont_cols, non_seq_cat_cols, non_seq_cont_cols


def encode_cat_columns(df, cat_cols, date_cols):
    """Label encode categorical data in dataframe.

    Args:
        df (pandas dataframe): Input dataframe to be encoded
        cat_cols (datetime): Categorical data column names
        date_cols (string): Datetime data column names
    """
    logger.info("Encode categorical columns using sklearn")

    label_encoders = {}
    for cat_col in cat_cols:
        label_encoders[cat_col] = LabelEncoder()
        if df[cat_col].dtype == object:
            df[cat_col] = df[cat_col].astype(str)
            label_encoders[cat_col].fit(df[cat_col])
            df[cat_col] = label_encoders[cat_col].transform(
                df[cat_col])
        else:
            label_encoders[cat_col].fit(df[cat_col])
            df[cat_col] = label_encoders[cat_col].transform(
                df[cat_col])

    for date_col in date_cols:
        df[date_col] = pd.to_datetime(df[date_col])

    return label_encoders


def flatten_multi_index(pivoted_events):
    """Flatten multilevel hierarchical indexes to a single level index."""

    pivoted_events_multi_index = pivoted_events.columns
    pivoted_events_index = pd.Index(
        [e[0] + str(e[1]) for e in pivoted_events_multi_index.tolist()])
    pivoted_events.columns = pivoted_events_index


def main_featurizer_sequential(df, cutoff_date, tgt_id, activity_date_column, n,
                               history_period, grace_period, seq_cols, reverse=False):
    """Sequentially featurize dataframe based in provided parameters.

    Args:
        df (pandas dataframe): Input dataframe to be featurized
        cutoff_date (datetime): The final date before which featurization is done
        tgt_id (string): Column name based on which to pivot dataframe
        activity_date_column (string): Column name of the datetime value of various activities
        n (int): Length of sequence history to look at
        history_period (int): Time before the cutoff date to deterimine sequence history to look at
        grace_period (int): Time before the cutoff date to deterimine end of featurization
        seq_cols (list of string): Sequential data column names
        reverse (bool): NaNs to the left if reverse=True
    """
    logger.info("Creating sequential features from pre-processed tabular data")
    def rename_cols(name, exclude=[]):
        if name not in exclude:
            return str(name) + '_'
        else:
            return str(name)

    df = df.loc[(df[activity_date_column] < cutoff_date - timedelta(days=grace_period)) &
                (df[activity_date_column] > cutoff_date - timedelta(days=grace_period) -
                 timedelta(days=history_period))]
    df.columns = list(map(str, df.columns))

    prof_cols = [col for col in df.columns.values if col not in seq_cols]
    to_pivot_cols = [col for col in df.columns.values if col in seq_cols] + [tgt_id]

    df_prof = df[prof_cols]

    df_lastn = df[to_pivot_cols].sort_values(
        activity_date_column).groupby(tgt_id).tail(n)
    df_lastn = df_lastn.fillna(0)
    df_lastn["times"] = df_lastn.sort_values(activity_date_column, ascending=(not reverse)) \
                                .groupby(tgt_id)[activity_date_column].cumcount() + 1

    df_lastn.columns = list(map(lambda x: rename_cols(x, list(tgt_id)), df_lastn.columns))
    to_pivot_cols = list(map(lambda x: rename_cols(x, list(tgt_id)), to_pivot_cols))
    pivoted_df = df_lastn.pivot(index=tgt_id, columns='times', values=[x for x in to_pivot_cols if x not in [tgt_id]])
    flatten_multi_index(pivoted_df)
    if reverse:
        pivoted_df = pivoted_df[pivoted_df.columns[::-1]]

    return pivoted_df, df_prof


def create_training_features_sliding_window(df_data, id_col, activity_col, activity_col_date, seq_cols_brief,
                                            cutoff_dates, train_end, label_end, seq_len, history_period):
    """Create training sequences based on sliding windows.

    Args:
        df_data (pandas dataframe): Input dataframe to be featurized
        cutoff_dates (list of datetime): The list of dates determining window end time
        train_end (list of datetime): The list of dates determining last date to consider for training data
        label_end (list of datetime): The list of dates determining last date to consider for labelling
        id_col (string): Column name based on which to pivot dataframe
        activity_col (string): Column name of activity features
        activity_col_date (string): Column name of the datetime value of various activities
        seq_len (int): Length of sequence history
        history_period (int): Time before the cutoff date to deterimine sequence history to look at
        seq_cols_brief (list of string): Sequential data column names
    """

    features_sequential_windows = []
    for i, cutoff_date in enumerate(cutoff_dates):
        features_sequential, features_non_sequential = main_featurizer_sequential(
            df_data, cutoff_date, id_col, 'booking_start_datetime', seq_len, history_period, 0, seq_cols_brief)
        features_sequential = features_sequential.reset_index()
        features_non_sequential = features_non_sequential.drop_duplicates([id_col])
        features_sequential = features_sequential.merge(features_non_sequential, on=id_col, how='inner')

        features_sequential_windows.append(features_sequential)

    df_activity_labels = []
    for i, label_end_i in enumerate(label_end):
        df_activities = df_data[(df_data[activity_col_date] > pd.to_datetime(train_end[i])) &
                                (df_data[activity_col_date] < pd.to_datetime(label_end_i))] \
            .pivot_table(index=id_col, columns=activity_col, aggfunc='size', fill_value=0)
        df_activities.reset_index()
        df_activities = df_activities.clip_upper(1)
        df_activity_labels.append(df_activities)

    training_set = []
    for i in range(len(label_end)):
        training_set.append(features_sequential_windows[i].merge(
            df_activity_labels[i].reset_index()[id_col].to_frame(), on=id_col, how='inner'))

        df_activity_labels[i] = df_activity_labels[i].merge(
            (features_sequential_windows[i][id_col]).to_frame(), on=id_col, how='inner')
        df_activity_labels[i] = df_activity_labels[i].drop(id_col, axis=1)

    df_activity = pd.concat(df_activity_labels)
    training_ = pd.concat(training_set)
    output_cols = df_activity.columns.values

    training_features = pd.concat(
        [training_.reset_index(), df_activity.reset_index()], axis=1)
    training_features = training_features.drop('index', axis=1)

    return training_features, output_cols


def fill_nans(df, cat_cols, cont_cols, date_cols, cutoff_dates, history_period, num_activites):
    """Fill NaNs with datetime values corresponding to start of window, null category for categorical values.

    Args:
        df (pandas dataframe): Input dataframe to be featurized
        cutoff_dates (list of datetime): The list of dates determining window end time
        history_period (int): Time before the cutoff date to deterimine sequence history to look at
        num_activites (list of int): List of cardinality of each categorical column
        cat_cols (list of string): Categorical data column names
        cont_cols (list of string): Continuous data column names
        date_cols (list of string): Datetime data column names
    """

    for date in date_cols:
        df[date] = df[date].fillna(
            cutoff_dates[0] - timedelta(days=history_period))
    for col in cat_cols:
        activity_count = num_activites[num_activites.index.str.startswith(
            col[:-2])]
        if activity_count.empty:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(activity_count[0]+1)
    for col in cont_cols:
        df[col] = df[col].fillna(0)


def num_activities(df, cat_cols_brief):
    """Get number of unique activities in the specified categorical column.

    Args:
        df (pd.DataFrame): Input dataframe
        cat_cols_brief (str): Categorical data column name
    """

    return df[cat_cols_brief].nunique()


def generate_col_lists_python(num_months, cat_cols_brief, cont_cols_brief, date_cols_brief,
                       seq_cols_brief, non_seq_cols_brief, left_pad=False):
    """Generate sequential list of columns if they have sequential featurization.

    Args:
        num_months (int): Length of sequence history
        cat_cols_brief (list of string): Categorical data column names
        cont_cols_brief (list of string): Continuous data column names
        date_cols_brief (list of string): Datetime data column names
        seq_cols_brief (list of string): Sequential data column names
        non_seq_cols_brief (list of string): Non-Sequential data column names
        left_pad (bool): Pad NaNs to the left if set to True
    """

    def number_cols(seq_len, i, left):
        return i + 1 if not left else seq_len - i

    def expand_seq_cols(cols_brief, num_months, left_pad):
        expanded_cols = []
        for item in cols_brief:
            if item in seq_cols_brief:
                for i in range(num_months):
                    expanded_cols.append(item + '_' + str(number_cols(num_months, i, left_pad)))
            else:
                expanded_cols.append(item)
        return expanded_cols

    cat_cols = expand_seq_cols(cat_cols_brief, num_months, left_pad)
    cont_cols = expand_seq_cols(cont_cols_brief, num_months, left_pad)
    seq_cols = expand_seq_cols(seq_cols_brief, num_months, left_pad)
    date_cols = expand_seq_cols(date_cols_brief, num_months, left_pad)
    non_seq_cols = non_seq_cols_brief
    return cat_cols, cont_cols, seq_cols, non_seq_cols, date_cols


def generate_col_lists(num_months, cat_cols_brief, cont_cols_brief, date_cols_brief,
                       seq_cols_brief, non_seq_cols_brief, left_pad=False):
    """Generate sequential list of columns if they have sequential featurization.

    Args:
        num_months (int): Length of sequence history
        cat_cols_brief (list of string): Categorical data column names
        cont_cols_brief (list of string): Continuous data column names
        date_cols_brief (list of string): Datetime data column names
        seq_cols_brief (list of string): Sequential data column names
        non_seq_cols_brief (list of string): Non-Sequential data column names
        left_pad (bool): Pad NaNs to the left if set to True
    """

    def number_cols(seq_len, i, left):
        return i + 1 if not left else seq_len - i

    def expand_seq_cols(cols_brief, num_months, left_pad):
        expanded_cols = []
        for item in cols_brief:
            if item in seq_cols_brief:
                for i in range(num_months):
                    expanded_cols.append(str(number_cols(num_months, i, left_pad))+ '_' + item)
            else:
                expanded_cols.append(item)
        return expanded_cols

    cat_cols = expand_seq_cols(cat_cols_brief, num_months, left_pad)
    cont_cols = expand_seq_cols(cont_cols_brief, num_months, left_pad)
    seq_cols = expand_seq_cols(seq_cols_brief, num_months, left_pad)
    date_cols = expand_seq_cols(date_cols_brief, num_months, left_pad)
    non_seq_cols = non_seq_cols_brief
    return cat_cols, cont_cols, seq_cols, non_seq_cols, date_cols


def generate_date_intervals(seq_len, date_cols, date_cols_brief, seq_cols_brief,
                            training_features, cutoff_dates, history_period):
    """Featurize datetime columns to ints and convert sequential datetime columns as an interval to first column.

    Args:
        seq_len (int): Length of sequence history
        date_cols (list of string): Datetime data column names in the pivoted df
        date_cols_brief (list of string): Datetime data column names
        seq_cols_brief (list of string): Sequential data column names
        training_features (pandas dataframe): Input dataframe post pivoting to be modified
        cutoff_dates (list of datetime): The list of dates determining window end time
        history_period (int): Time before the cutoff date to deterimine sequence history to look at
    """

    training_features[date_cols] = (training_features[date_cols] - pd.Timestamp(
        cutoff_dates[0] - timedelta(days=history_period))).astype(int) // 10e10
    for item in date_cols_brief:
        if item in seq_cols_brief:
            for i in range(2, seq_len+1):
                cur_col = item + str(i)
                prev_col = item + str(1)
                training_features[cur_col] = training_features[cur_col] - training_features[prev_col]


def normalize_numerical_data(df, cols, scaler_type='Standard'):
    """Normalize continuous and interval date-time data in dataframe.

    Args:
        df (pandas dataframe): Input dataframe to be featurized
        cols (list of string): Data column names to be normalized
        scaler_type (string): Type of numerical scaling algorithm
    """

    logger.info("Normalize numerical columns")
    if scaler_type == 'Standard':
        scaler = preprocessing.StandardScaler()
    else:
        scaler = preprocessing.MinMaxScaler()

    if cols:
        df[cols] = scaler.fit_transform(df[cols])
    return df


def split_data(train_split, val_split, df, shuffle=False):
    """Split the data into train, val and test data, and return the split data for induction into the model."""

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    num_records = df.shape[0]
    train_size = int(train_split * num_records)
    val_size = int(val_split * num_records)

    df_train = df[:train_size]
    df_val = df[train_size:train_size+val_size]
    df_test = df[train_size+val_size:]

    return df_train, df_val, df_test


def dataloader_toarray(dataloader, device):
    """Convert the dataset into a single numpy array.

    Args:
        dataloader (DataLoader): Pytorch data loader that provides an iterable over the given dataset
        device (torch.device): The device ('cpu' or 'gpu') on which a torch.Tensor is or will be allocated
    """
    all_embed = np.array([])
    for _, y, seq_cat_x, seq_cont_x, non_seq_cat_x, non_seq_cont_x in dataloader:
        non_seq_cat_x = non_seq_cat_x.to(device)
        non_seq_cont_x = non_seq_cont_x.to(device)
        seq_cont_x = seq_cont_x.to(device)
        seq_cat_x = seq_cat_x.to(device)
        y = y.to(device)

        # Forward Pass
        seq_cont_x = seq_cont_x.reshape(seq_cont_x.shape[0], seq_cont_x.shape[1] * seq_cont_x.shape[2])
        seq_cat_x = seq_cat_x.reshape(seq_cat_x.shape[0], seq_cat_x.shape[1] * seq_cat_x.shape[2])

        tens = torch.cat((seq_cat_x, seq_cont_x, non_seq_cat_x, non_seq_cont_x), dim=1).cpu().detach().numpy()
        all_embed = np.concatenate((all_embed, tens), 0) if all_embed.size else tens

    return all_embed


def smote(df, cat_cols, output_col, sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=-1):
    """Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTE-NC).

    Args:
        df (pandas dataframe): Input dataframe to be oversampled
        output_col (str): Column name of the labels
        cat_cols (list): List of column names of the categorical features
        sampling_strategy (float, str, dict or callable): Sampling information to resample the data set
        random_state (int, RandomState instance or None): Control the randomization of the algorithm
        k_neighbors (int or object): Number of nearest neighbors to used to construct synthetic samples
        n_jobs (int): Number of threads to open if possible
    """

    x = df.drop(columns=[output_col])
    y = df[output_col]
    cat_idxs = [x.columns.get_loc(c) for c in cat_cols]
    sm = SMOTENC(random_state=random_state, categorical_features=cat_idxs,
                 sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, n_jobs=n_jobs)
    x, y = sm.fit_resample(x, y)
    df_smote = pd.concat([x, y], axis=1)
    return df_smote


def get_nonempty_tensors(data):
    """Remove the empty tensors from the input and only return the non_empty ones as a Tuple.

    It also returns a list of indices (length = 4) representing the positions of each data type in the tuple.

    Args:
        data (tuple of Tensors): Input of data from the user - might contain some empty tensors
    """

    nonempty_tensors = []
    nonempty_idx = [-1] * 4
    start = 0
    for i, d in enumerate(data):
        if d.nelement() != 0:
            nonempty_tensors.append(d)
            nonempty_idx[i] = start
            start += 1
    return tuple(nonempty_tensors), nonempty_idx
