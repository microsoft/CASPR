"""Utility functions for Pyspark dataframe preprocessing."""

# coding: utf-8
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from functools import reduce

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import (PandasUDFType, col, collect_set, count, countDistinct, date_sub, datediff, first,
                                   lag, lit, pandas_udf, row_number, unix_timestamp, when)
from pyspark.sql.types import DoubleType, IntegerType, StringType

from caspr.utils.preprocess import generate_col_lists

MAX_CAT_CARDINALITY = 30000
PRUNED_ITEMS = 'pruned_product'
PRUNED_ROWS = 'pruned_rows'
logger = logging.getLogger(__name__)

def get_num_activities(cat_encoding, out_spark=False):
    """Get number of unique activities for each categorical variable.

    Args:
        cat_encoding (dict): Categorical encoding dictionary returned by encode_cat_columns
        out_spark (bool): Return a Spark dataframe if set to True
    """

    # The + 1 is added to take care of the index 0 assigned to unknown values
    num_activities_dict = {c: vals.count() + 1 for c, vals in cat_encoding.items()}

    # Outputting the spark df as the fill nans function uses it.
    # If we can get it to use the dict - we can probably omit the usage of spark df
    if out_spark:
        spark = SparkSession.builder.getOrCreate()
        pd_df = pd.DataFrame(num_activities_dict, index=[0])
        num_activities_df = spark.createDataFrame(pd_df)
        return num_activities_df

    return num_activities_dict


def petastorm_handover(df, partitions, petastorm_path="/ml/tmp/petastorm"):
    """Save the dataframe in the parquet format to the specified path for petastorm handover."""

    # Set a unique working directory for this pipeline, backed by SSD storage
    petastorm_dir = os.path.join(petastorm_path, str(uuid.uuid4()))

    os.mkdir(petastorm_dir)

    # opportunity to finetune the output for Petastorm via partitioning and block size
    parquet_path = os.path.join(petastorm_dir, "parquet")
    df.repartition(partitions).write.mode("overwrite").option("parquet.block.size", 1024 * 1024).parquet(parquet_path)

    # potentially usable with native readers
    # petastorm_dataset_url = "file://" + get_local_path(parquet_path)

    return parquet_path


def normalize_value(x, c, summary, scaling):
    """Get the normalized value for the input.

    Args:
        x (float): Input value to be normalized
        c (str): Selected numerical column
        summary (pd.DataFrame): Summary of the original distribution (if normalization reapplied)
        scaling (str): choice between "min_max" and "standard" scaling
    """

    if scaling == 'min_max':
        c_min = float(summary[c]['min'])
        c_max = float(summary[c]['max'])
        normalized_x = (x - c_min) / (c_max - c_min) if c_max != c_min else x
    elif scaling == 'standard':
        c_mean = float(summary[c]['mean'])
        c_std = float(summary[c]['stddev'])
        normalized_x = (x - c_mean) / c_std if c_std else x
    else:
        raise Exception('Scaler type not supported: {}'.format(scaling))
    return normalized_x


def fill_date_nans_sp(df, date_cols, history_period=365, prediction_date_column='prediction_date',
                      interval=False, summary_date=None, scaling='min_max'):
    """Fill date nans with history_period if interval is True else prediction_date - history_period.

    If summary_date is provided, normalize the value based on the mean/std or min/max of the corresponding date column.
    This function is generalized for non-pivoted and pivoted dataframes.
    """

    if not interval and prediction_date_column not in df.columns:
        raise Exception('{} could not be found in the dataframe'.format(prediction_date_column))
    if summary_date is None:
        if interval:
            df = df.fillna(history_period, subset=date_cols)
        else:
            for date in date_cols:
                df = df.withColumn(date, when(col(date).isNull(), unix_timestamp(
                    date_sub(col(prediction_date_column), history_period))).otherwise(col(date)))
    else:
        date_cols_ = summary_date.columns
        # Use regex pattern match to check whether the input dataframe is pivoted or not
        pattern = re.compile(r'(\d+)_(\w+)')
        if interval:
            # Fill nans with normalized history_period
            for date in date_cols:
                date_brief = date.split('_', 1)[-1] if pattern.match(date) else date
                df = df.fillna(normalize_value(history_period, date_brief, summary_date, scaling), subset=date)
        else:
            # Create an additional 'start_of_window' column which contains the start date of the window
            # for each date column in date_cols_ (these columns will be identical at this stage)
            start_window_cols_ = ['start_of_window_{}'.format(date) for date in date_cols_]
            for start_window in start_window_cols_:
                df = df.withColumn(start_window, unix_timestamp(date_sub(col(prediction_date_column), history_period)))
            # Rename the column names in summary_date to the corresponding 'start_of_window' columns
            # in order to call the normalize_columns function to get the normalized start dates
            summary_date = summary_date.rename(columns=dict(zip(date_cols_, start_window_cols_)))
            df, _ = normalize_columns(df, start_window_cols_, summary_date, scaling)
            # Fill the nan values based on the corresponding normalized 'start_of_window' column.
            # Note that for pivoted dataframes, there will be multiple timestemps for each date column,
            # e.g., 1_transactionDate, 2_transactionDate, etc.
            # All these columns with the same dateType should be filled with the same normalized value
            # given the same prediction_date, so the normalized 'start_of_window' columns are created
            # for each column in date_cols_brief instead of date_cols
            for date in date_cols:
                date_brief = date.split('_', 1)[-1] if pattern.match(date) else date
                normalized_default_date = 'start_of_window_{}'.format(date_brief)
                df = df.withColumn(date, when(col(date).isNull(), col(normalized_default_date)).otherwise(col(date)))
            df = df.drop(*start_window_cols_)
    return df


def fill_nans_sp(df, cat_cols=None, cont_cols=None, date_cols=None, history_period=365,
                 prediction_date_column='prediction_date', interval=False, summary_date=None, scaling='min_max'):
    """Fill NaNs for the given columns.

    For date columns, fill NaNs with start of window datetime if interval is False else history_period.
    For continuous and categorical columns, fill NaNs with 0.
    """

    if date_cols:
        logger.info('Filling date nans')
        df = fill_date_nans_sp(df, date_cols, history_period, prediction_date_column, interval, summary_date, scaling)

    if cont_cols:
        logger.info('Filling continuous column nans')
        df = df.fillna(0, subset=cont_cols)

    if cat_cols:
        logger.info('Filling categorical column nans')
        df = df.fillna(0, subset=cat_cols)

    return df


def normalize_columns(df, cols, summary=None, scaling="min_max"):
    """Normalize numerical columns in the dataframe.

    Args:
        df (spark dataframe): Input dataframe to be normalized
        cols (array of column names): Selected numerical columns
        summary (optional, pandas dataframe): Summary of the original distribution (if normalization reapplied)
        scaling: choice between "min_max" (default) and "standard" scaling
    """
    logger.info("Normalizing numerical columns using {} scaling".format(scaling))

    if summary is None:
        summary = df[cols].describe().toPandas().set_index('summary')
    for c in cols:
        if scaling == "min_max":
            c_min = float(summary[c]['min'])
            c_max = float(summary[c]['max'])
            if c_min == c_max:
                logger.info("No variance to normalize, consider dropping: %s" % c)
                continue

            @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
            def scaler(c):
                return pd.Series([(i-c_min)/(c_max-c_min) for i in c])
        elif scaling == "standard":
            c_mean = float(summary[c]['mean'])
            c_std = float(summary[c]['stddev'])
            if c_std == 0.0:
                logger.info("No variance to normalize, consider dropping: %s" % c)
                continue

            @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
            def scaler(c):
                return pd.Series([(i-c_mean)/c_std for i in c])
        else:
            logger.info("Scaler type not supported: %s" % scaling)
            break
        df = df.withColumn(c, scaler(c))
    return df, summary


def denormalize_columns(df, cols, summary, scaling="min_max"):
    """Reverse normalization of numerical columns in the dataframe.

    Args:
        df (spark dataframe): Input dataframe after normalization
        cols (array of column names): Selection of numerical columns (normalized)
        summary (pandas dataframe): Summary of the original distribution (before normalization)
        scaling: choice between "min_max" (default) and "standard" scaling to reverse
    """
    for c in cols:
        if scaling == "min_max":
            c_min = float(summary[c]['min'])
            c_max = float(summary[c]['max'])
            if c_min == c_max:
                continue

            @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
            def scaler(c):
                return pd.Series([i*(c_max-c_min)+c_min for i in c])
        elif scaling == "standard":
            c_mean = float(summary[c]['mean'])
            c_std = float(summary[c]['stddev'])
            if c_std == 0.0:
                continue

            @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
            def scaler(c):
                return pd.Series([i*c_std+c_mean for i in c])
        else:
            logger.info("Scaler type not supported: %s" % scaling)
            return df
        df = df.withColumn(c, scaler(c))
    return df


def encode_cat_columns_sparkrank(df, cols, encoding=None, max_cardinality=MAX_CAT_CARDINALITY):
    """Label encode categorical data in dataframe using joins.

    Args:
        df (dataframe): Input dataframe to be encoded
        cols (array): Categorical data column names
        encoding (optional, dict of spark dataframe): categorical encoding to reapply
    """

    def filter_cat_cardinality(dataframe, column, cardinality):
        window_items = Window.partitionBy().orderBy(col('count').desc(), col(column))
        top_category = dataframe.groupBy(column).count().select(
            '*', row_number().over(window_items).alias(column + '_rank'))\
            .filter(col(column + '_rank') <= cardinality)
        return top_category.select(column, column + '_rank')

    percent_pruned_items = 0
    percent_pruned_row = 0
    dataframe_filtered = df

    if encoding is None:
        encoding = {}
        i = 0
        for cat_col in cols:
            cardinality = df.select(cat_col).distinct().count()
            top_category = filter_cat_cardinality(
                dataframe_filtered.dropna(subset=(cat_col)), cat_col, max_cardinality)
            # drop na done to deal with None columns possible in the data
            # they should not be included in the groupby and cardinality calc
            encoding[cat_col] = top_category

            if cardinality > max_cardinality:
                # raise ValueError("Too many distinct values for: %s" % cat_col)
                dataframe_filtered = dataframe_filtered.join(top_category, on=cat_col, how='left_semi')
                logger.info("Number of filtered {} is {}".format(cat_col, cardinality-max_cardinality))
                percent_pruned_items = (percent_pruned_items*i + 1-max_cardinality/cardinality)/(i+1)
                i = i+1

        df_count = df.count()
        df_filtered_count = dataframe_filtered.count()
        logger.info("Number of rows with unk tokens due to high cardinality is {}".format(
            df_count-df_filtered_count))
        percent_pruned_row = 1 - df_filtered_count/df_count

    # Label encode categorical data in dataframe
    for cat_col in cols:
        enc_map = encoding[cat_col]
        dataframe_filtered = dataframe_filtered.join(enc_map, cat_col, 'left')\
                                               .drop(cat_col)\
                                               .withColumnRenamed(cat_col + '_rank', cat_col)\
                                               .fillna(0, subset=[cat_col])
    # The fillna takes care of the new or unknown strings

    return dataframe_filtered, encoding, {PRUNED_ITEMS: percent_pruned_items, PRUNED_ROWS: percent_pruned_row}


def encode_cat_columns_pandasudf(df, cols, encoding=None, max_cardinality=MAX_CAT_CARDINALITY):
    """Categorical Encoding powered by Intel Streaming SIMD extensions and pandas udf.

    When cat cardinality high, use encode_cat_columns_sparkrank() instead.
    Label encode categorical data in dataframe.

    Args:
        df (dataframe): Input dataframe to be encoded
        cols (array): Categorical data column names
        encoding (optional, dict): categorical encoding to reapply
    """
    RESERVED = "UNK"  # noqa: C0103
    logger.info("Encoding categorical columns")

    def filter_cat_cardinality(dataframe, column, cardinality):
        window_items = Window.partitionBy().orderBy(col('count').desc())
        top_category = dataframe.groupBy(column).count().select(
            '*', row_number().over(window_items).alias('rank')).filter(col('rank') <= cardinality)
        dataframe_filtered = dataframe.join(top_category.select(column), on=column, how='inner')
        return dataframe_filtered
    percent_pruned_items = 0
    percent_pruned_row = 0
    if encoding is None:
        dataframe_filtered = df
        i = 0
        for cat_col in cols:
            cardinality = df.select(cat_col).distinct().count()
            if cardinality > max_cardinality:
                # raise ValueError("Too many distinct values for: %s" % cat_col)
                dataframe_filtered = filter_cat_cardinality(dataframe_filtered, cat_col, max_cardinality)
                logger.info("Number of filtered {} is {}".format(cat_col, cardinality-max_cardinality))
                percent_pruned_items = (percent_pruned_items*i + 1-max_cardinality/cardinality)/(i+1)
                i = i+1
        encoding = {}
        logger.info("Number of rows with unk tokens due to high cardinality is {}".format(
            df.count()-dataframe_filtered.count()))
        percent_pruned_row = 1 - dataframe_filtered.count()/df.count()
        # find all distinct values per categorical column
        distinct = dataframe_filtered.select(*[collect_set(c).alias(c)
                                               for c in cols]).toPandas()
        # create a mapping array per column
        for cat_col in cols:
            encoding[cat_col] = [RESERVED]
            col_distinct = distinct[cat_col][0]
            # improve encoding determinism
            col_distinct = np.sort(col_distinct)
            encoding[cat_col].extend(col_distinct)
    # Label encode categorical data in dataframe
    for cat_col in cols:
        enc_map = encoding[cat_col]

        @pandas_udf(IntegerType(), PandasUDFType.SCALAR)
        def encoder(c):
            def get_label(val):
                if val not in enc_map:
                    val = RESERVED
                return enc_map.index(val)
            return pd.Series([get_label(v) for v in c])
        df = df.withColumn(cat_col, encoder(cat_col))
    return df, encoding, {PRUNED_ITEMS: percent_pruned_items, PRUNED_ROWS: percent_pruned_row}


def decode_cat_columns_pandasudf(df, encoding):
    """Reverse categorical encoding generated from pandasudf version in a dataframe.

    Args:
        df (dataframe): Input dataframe with encoded columns
        cols (array): Categorical columns to be decoded
        encoding (dict): categorical encoding to reverse
    """
    for cat_col in encoding.keys():
        enc_map = encoding[cat_col]

        @pandas_udf(StringType(), PandasUDFType.SCALAR)
        def decoder(c):
            return c.apply(lambda x: str(enc_map[x]))
        df = df.withColumn(cat_col, decoder(cat_col))
    return df


def append_dummy_rank(df_n, n, max_seq_len, left_pad):
    """Append dummy rank to pad sequences to user-specified sequence length.

    Args:
        df_n (spark dataframe): Input dataframe filtered to latest n timesteps
        n (int): Length of sequence history to look at
        max_seq_len (int): Max sequence length in user activities
        left_pad (Boolean): Pad missing timestamps to the left if set to True
    """

    if n - max_seq_len > 0:
        spark = SparkSession.builder.getOrCreate()
        dummy_rank = spark.range(1, n - max_seq_len + 1) if left_pad else spark.range(max_seq_len + 1, n + 1)
        dummy_rank = dummy_rank.withColumnRenamed('id', 'rank')
        for c in df_n.columns:
            if c != 'rank':
                dummy_rank = dummy_rank.withColumn(c, lit(None))
        df_n = df_n.unionByName(dummy_rank)
    return df_n


def main_featurizer_sequential_sp(df, tgt_id, n, seq_cols, left_pad=False):
    """Sequentially featurize dataframe based in provided parameters.

    Args:
        df (spark dataframe): Input dataframe to be featurized
        tgt_id (list of string): Column name based on which to pivot dataframe
        n (int): Length of sequence history to look at
        seq_cols (list of string): Sequential data column names
        non_seq_cols (list of string): Non-sequential data column names
        left_pad (Boolean): Pad missing timestamps to the left if set to True
    """
    logger.info("Creating sequential features from tabular data")

    if not all(c in df.columns for c in ['seq_len', 'rank_asc', 'rank_desc']):
        raise Exception('Call get_rank before normalize_columns and this function')

    # Take latest n timesteps
    max_seq_len = df.agg({'seq_len': 'max'}).collect()[0][0] or 0
    df = df.filter(col('rank_desc') <= min(n, max_seq_len))

    # If max_seq_len < n (all seq_len < n):
    #     left_pad: shift right (n - seq_len) steps and append dummy rank for first (n - max_seq_len) steps
    #     right_pad: no shift and append dummy rank for last (n - max_seq_len) steps
    # If max_seq_len >= n:
    #     left_pad:
    #         seq_len < n: shift right (n - seq_len) steps
    #         seq_len >= n: shift left (seq_len - n) steps
    #     right_pad:
    #         seq_len < n: do nothing
    #         seq_len >= n: shift left (seq_len - n) steps
    # Summary: shift if left_pad or n - seq_len <= 0, otherwise don't shift
    df = df.withColumn('shift', n - col('seq_len')) \
           .withColumn('shift_flag', lit(left_pad)) \
           .withColumn('shift_flag', when((col('shift') <= 0) | (col('shift_flag')), 1).otherwise(0)) \
           .withColumn('shift', col('shift_flag') * col('shift')) \
           .withColumn('rank', col('rank_asc') + col('shift'))
    df = append_dummy_rank(df, n, max_seq_len, left_pad)

    pivoted_df = df.groupby(tgt_id).pivot('rank').agg(*(first(col(c)).alias(c) for c in seq_cols))
    # drop dummy
    pivoted_df = pivoted_df.na.drop(subset=tgt_id)
    return pivoted_df


def convert_timestamps_to_intervals(df, user_id_col='user_id', datetime_col='activity_date_column',
                                    interval_col_name="interval"):
    """Convert timestamps to intervals between each transaction date."""

    new_df = df.withColumn(interval_col_name, datediff(df[datetime_col], lag(
        df[datetime_col], 1).over(Window.partitionBy(user_id_col).orderBy(datetime_col))))
    return new_df


def remove_underscore_in_seq_col_name_dataframe(df, cols):
    """Transform current featurizer outputs of the form (123)_(abc) to (abc)(123) for petastorm handover."""

    regex_replace = re.compile(r'(\d+)_(\w+)')
    for c in cols:
        renamed_col = regex_replace.sub(r'\2\1', c)
        df = df.withColumnRenamed(c, renamed_col)
    return df


def remove_underscore_in_seq_col_name_list(cols):
    """Transform list of column names of the form (123)_(abc) to (abc)(123) for petastorm handover."""

    regex_replace = re.compile(r'(\d+)_(\w+)')
    new_cols = []
    for c in cols:
        renamed_col = regex_replace.sub(r'\2\1', c)
        new_cols.append(renamed_col)
    return new_cols


def get_sliding_window_dates(df, data_rows_needed, user_id, activity_date_column, history_period,
                             overlap_percentage, latest_prediction_date=None):
    """Determine the cutoff dates for CASPR training."""

    history_period = history_period * 24 * 3600
    rows_available = df.select(user_id).distinct().count()
    sliding_window_count = int(data_rows_needed / rows_available) + 1
    if latest_prediction_date is None:
        latest_prediction_date = df.agg({activity_date_column: "max"}).collect()[0][0]

    window_interval = history_period * (1-overlap_percentage)
    cutoff_dates = [datetime.fromtimestamp(int(datetime.timestamp(latest_prediction_date) -
                                               (i * window_interval))) for i in range(sliding_window_count)]

    return cutoff_dates


def get_sequence_length(df, tgt_ids, percentile):
    """Get the user activity sequence length for the given percentile."""

    df_with_count = df.groupby(tgt_ids).count()
    entry = df_with_count.approxQuantile(['count'], [percentile], 0.001)
    return int(entry[0][0])


def get_lookback_period(df, user_id, activity_date_column, max_avg_sequence_length, latest_prediction_date=None):
    """Determine the lookback period based on the input dataframe."""

    # Find avg number of transactions in a month
    # Use time till 50 sequence length
    if latest_prediction_date is None:
        latest_prediction_date = df.agg({activity_date_column: "max"}).collect()[0][0]

    one_month_spans = [latest_prediction_date - timedelta(days=30*i) for i in range(5)]
    avg_transaction_counts = []
    # Use 4 windows to average
    # Use these windows to add more checks as well
    # For now these windows could be merged into one as well
    df = df.withColumn("date_bracket", when(
        (col(activity_date_column) <= one_month_spans[0]) &
        (col(activity_date_column) > one_month_spans[1]), 0).when(
        (col(activity_date_column) <= one_month_spans[1]) &
        (col(activity_date_column) > one_month_spans[2]), 1).when(
        (col(activity_date_column) <= one_month_spans[2]) &
        (col(activity_date_column) > one_month_spans[3]), 2).when(
        (col(activity_date_column) <= one_month_spans[3]) &
        (col(activity_date_column) > one_month_spans[4]), 3).otherwise(-1))

    count_df = df.groupby("date_bracket").agg(
        countDistinct(user_id).alias("user_count"), count(user_id).alias("total_count"))

    count_df = count_df.withColumn("avg_count", when(col("user_count") != 0,
                                                     col("total_count") / col("user_count")).otherwise(0.0))
    count_df_pd = count_df.toPandas()
    count_dict = count_df_pd.to_dict()

    avg_transaction_counts = [count_dict["avg_count"][i] for i in range(len(count_df_pd))]

    avg_transactions_per_month = np.mean(avg_transaction_counts)
    days_allowed = float(max_avg_sequence_length)*30 / float(avg_transactions_per_month)
    return int(days_allowed)


def get_rank(df, tgt_id, activity_date_column):
    """Get the ascending and descending rank for each activity in the user activity sequence."""

    window = Window.partitionBy([col(x) for x in tgt_id])
    window_desc = window.orderBy(df[activity_date_column].desc())
    df = df.withColumn('seq_len', count(activity_date_column).over(window)) \
           .withColumn('rank_desc', row_number().over(window_desc)) \
           .withColumn('rank_asc', col('seq_len') - col('rank_desc') + 1)
    return df


def pipeline(df, tgt_id, activity_date_column, prediction_date_column, cat_cols_, cont_cols_, seq_cols_,
             non_seq_cols_, date_cols_, output_col, history_period=365, seq_len=15, left_pad=False,
             interval=False, scaling='min_max', encoding=None, summary=None):
    """Featurize the dataframe for CASPR models.

    Args:
        df (spark dataframe): Input dataframe to be featurized
        tgt_id (list of string): Column name based on which to pivot dataframe
        activity_date_column (string): Column name of the datetime value of activities
        prediction_date_column (string): Column name of the prediction date before which featurization is done
        cat_cols_ (list of string): Categorical data column names
        cont_cols_ (list of string): Continuous data column names. Date columns should be excluded
        seq_cols_ (list of string): Sequential data column names. Date columns should be included
        non_seq_cols_ (list of string): Non-Sequential data column names
        date_cols_ (list of string): Datetime data column names. Note: All date columns can be put here (e.g.,
        SubStartDate, SubEndDate, ActivityDate), but at least activity_date_column should be included
        output_col (list of string): Column names of the labels
        history_period (int): Time before the cutoff date to deterimine sequence history to look at
        seq_len (int): Length of sequence history to look at
        left_pad (Boolean): Pad missing timestamps to the left if set to True
        interval: (Boolean): Convert timestamps to intervals between timestamps and cutoff dates if set to True
        scaling: choice between "min_max" (default) and "standard" scaling
        encoding (dict of Spark dataframes): Categorical encoding dictionary
        summary (pd.DataFrame): Summary of the original distribution (if normalization reapplied)
    """
    logger.info("Running pipeline in a SPARK context, to create features that caspr can consume")

    # this step can take take time if done after a lot of data wrangling
    # doing this here reduces the impact on perf as data is still raw and possibly checkpointed
    num_partitions_of_data = df.rdd.getNumPartitions()

    cat_cols, cont_cols, _, _, date_cols = generate_col_lists(
        seq_len, cat_cols_, cont_cols_, date_cols_, seq_cols_, non_seq_cols_)

    # Filter customers within active windows
    df = df.withColumn('history_period', lit(history_period * 24 * 3600)) \
           .withColumn('start_date', (unix_timestamp(col(prediction_date_column)) -
                                      col('history_period')).cast('timestamp')) \
           .filter((col(activity_date_column) < col(prediction_date_column)) &
                   (col(activity_date_column) > col('start_date')))

    # Get rank for pivoting and padding
    df = get_rank(df, tgt_id, activity_date_column)

    if interval:
        for date in date_cols_:
            df = df.withColumn(date, datediff(col(prediction_date_column), col(date)))
    else:
        for date in date_cols_:
            df = df.withColumn(date, unix_timestamp(col(date)))

    df, encoding, _ = encode_cat_columns_sparkrank(df, cat_cols_, encoding)
    # Repartitioning done outside the encode cat columns for better visibility
    # Also allows for reuse of the partition count
    df = df.repartition(num_partitions_of_data)

    df, summary = normalize_columns(df, date_cols_+cont_cols_, summary, scaling)
    summary_date = summary[date_cols_]

    df = fill_nans_sp(df, cat_cols_, cont_cols_, date_cols_, history_period,
                      prediction_date_column, interval, summary_date, scaling)

    pivoted_df = main_featurizer_sequential_sp(df, tgt_id=tgt_id, n=seq_len, seq_cols=seq_cols_, left_pad=left_pad)

    prof_df = df[tgt_id+non_seq_cols_+output_col].dropDuplicates()

    df = pivoted_df.join(prof_df, on=tgt_id, how='inner') if non_seq_cols_ + output_col else pivoted_df

    df = fill_nans_sp(df, cat_cols, cont_cols, date_cols, history_period,
                      prediction_date_column, interval, summary_date, scaling)
    return df, encoding, summary


def data_process_all_sp(train_data, val_data, test_data, tgt_id, activity_date_column, prediction_date_column,
                        cat_cols_, cont_cols_, seq_cols_, non_seq_cols_, date_cols_, output_col,
                        history_period=365, seq_len=15, left_pad=False, interval=False, scaling='min_max'):
    """Call pipeline function for train, validation, and test data."""

    train, encoding, summary = pipeline(train_data, tgt_id, activity_date_column, prediction_date_column,
                                        cat_cols_, cont_cols_, seq_cols_, non_seq_cols_, date_cols_, output_col,
                                        history_period, seq_len, left_pad, interval, scaling)

    val, _, _ = pipeline(val_data, tgt_id, activity_date_column, prediction_date_column,
                         cat_cols_, cont_cols_, seq_cols_, non_seq_cols_, date_cols_, output_col,
                         history_period, seq_len, left_pad, interval, scaling, encoding, summary)

    test, _, _ = pipeline(test_data, tgt_id, activity_date_column, prediction_date_column,
                          cat_cols_, cont_cols_, seq_cols_, non_seq_cols_, date_cols_, output_col,
                          history_period, seq_len, left_pad, interval, scaling, encoding, summary)

    return train, val, test, encoding
