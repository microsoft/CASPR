from os.path import basename, realpath, splitext
from timeit import timeit
from typing import Dict, List

import numpy as np
import pandas as pd
from captum.attr import (DeepLift, DeepLiftShap, IntegratedGradients, configure_interpretable_embedding_layer,
                         remove_interpretable_embedding_layer)
from explainer.explainer import TIME_PRECISION, Explainer, SummaryFormat, SummaryScope
from explainer.pandas_df_utility import PandasDfUtility
from pandas import DataFrame as PandasDataFrame
from torch.utils.data import DataLoader

from caspr.models.embedding_layer import CategoricalEmbedding
from caspr.utils.preprocess import get_nonempty_tensors

SEQ_CAT_INDEX = 0
SEQ_CONT_INDEX = 1
NON_SEQ_CAT_INDEX = 2
NON_SEQ_CONT_INDEX = 3

MODULE_NAME = splitext(basename(realpath(__file__)))[0]


class CASPRExplainer(Explainer):
    """Encapsulate an explainer for the CASPR model.

    Currently the explainer expects a single output from the model.
    The library currently supports 3 algorithms:
        1. DeepLift (Default)
        2. DeepLiftShap
        3. IntegratedGradients

    Deeplift is the fastest and consumes least amount of memory - It is also the one recommended for most cases
    This can be initialised in the init function.
    """

    def __init__(self,
                 logger,
                 model=None,
                 check_additivity: bool = True,
                 algorithm='DeepLift',
                 explainer_verbosity: bool = False
                 ):
        """Initialize the class.

        The parameters are:
        logger - The logger.
        model - model to be used for explanation.
        check_additivity - This runs a special validation that the sum of all SHAP Values
            and model's expected value equals to the model output.
        algorithm - which captum algorithm to be used
        explainer_verbosity - Determines if every step info is logged or not

        """

        self.logger = logger
        self.model = model
        self.check_additivity = check_additivity
        self.explainer_verbosity = explainer_verbosity
        self.interpretable_emb_seq = None
        self.interpretable_emb_non_seq = None
        self.emb_name_seq = None
        self.emb_dims_seq = None
        self.emb_name_non_seq = None
        self.emb_dims_non_seq = None

        self._find_emb_layer()
        try:
            # The constructor throws an exception if a model is of unsupported type.
            if algorithm == 'DeepLift':
                self.explainer = DeepLift(self.model)
            elif algorithm == 'IntegratedGradients':
                self.explainer = IntegratedGradients(self.model)
            elif algorithm == 'DeepLiftShap':
                self.explainer = DeepLiftShap(self.model)

        except Exception as ex:  # noqa: W0703
            self.logger.info('Failed to initialise captum explainer on the model')
            raise Exception

    def _find_emb_layer(self):
        """Search for embeddings layers in the model.

        The default function used at initialisation time to find embedding layers supporting categorical variables
        and extract embedding dimensions. These dimensions are required for aggregation.

        Assumes the embedding layer used is the Categorical Embedding layer available with the module.
        """
        if self.explainer_verbosity:
            self.logger.info("Searching for categorical embedding layers")

        for name, module in self.model.named_modules():
            if isinstance(module, CategoricalEmbedding) and module.emb_size > 0:
                if module.is_seq:
                    self.emb_name_seq = name
                    self.emb_dims_seq = module.emb_dims
                    if self.explainer_verbosity:
                        self.logger.info("Sequential categorical embedding layer found")
                else:
                    self.emb_name_non_seq = name
                    self.emb_dims_non_seq = module.emb_dims
                    if self.explainer_verbosity:
                        self.logger.info("Non Sequential categorical embedding layer found")

    def _configure_interpretable_embedding_layer(self):
        """Configure wrapper layer over embedding layer.

        This function is used for the initialisation of an Interpretable_Embedding layer provided in the Captum
        library that encapsulates the Categorical Embedding layer to allow for its explainability.
        This layer allows for the CASPRExplainer to be able to input embedding vectors corresponding to the categorical
        variables (calculated before sending them in the models' forward function) in the model while muting the
        categorical embedding layer in the models' forward function.
        """

        self.logger.info('Configuring categorical embedding layers')

        if self.emb_name_seq is not None:
            self.interpretable_emb_seq = configure_interpretable_embedding_layer(self.model, self.emb_name_seq)
        if self.emb_name_non_seq is not None:
            self.interpretable_emb_non_seq = configure_interpretable_embedding_layer(self.model, self.emb_name_non_seq)

    def _remove_interpretable_embedding_layer(self):
        """Remove wrapper layer.

        This function removes the encapsulation of Categorical Embedding layer by Interpretable_Embedding layer
        bringing back the model to its original state.
        This function is called after the attribute calculation.
        """

        if self.explainer_verbosity:
            self.logger.info("Removing categorical embedding layer")
        if self.interpretable_emb_seq:
            remove_interpretable_embedding_layer(self.model, self.interpretable_emb_seq)
        if self.interpretable_emb_non_seq:
            remove_interpretable_embedding_layer(self.model, self.interpretable_emb_non_seq)

    def _aggregate_cat_attributions_util(self, attribution, emb_dims):  # noqa: R0201
        """Aggregate attributions of each categorical variable.

        Util function used to separate out the attributes for every categorical variable by summing the attributes
        provided for every dimension of the embedding of the respective categorical variable

        Args:
            attribution (numpy array) : Attributes of all the categorical variables of a certain type (non_seq or seq)
            emb_dims (List of tuple) : emb_dims list required to separate out the attributes for every variable
        """
        if emb_dims is None:
            return attribution

        attribution_agg = []
        start = 0
        for _, emb_dim in emb_dims:
            attribution_norm = np.sum(attribution[..., start:start + emb_dim], axis=-1)
            attribution_agg.append(attribution_norm)
            start += emb_dim
        attribution_agg = np.stack(attribution_agg, axis=-1)
        return attribution_agg

    def _aggregate_cat_attributions(self, attributions, nonempty_idx):
        """Aggregate attributions of each categorical variable.

        Function used to separate out the attributes for every categorical variable by summing the attributes
        provided for every dimension of the embedding of the respective categorical variable.
        Uses the Util function.

        Args:
            attributions (tuple of Tensors) : The tuple of attributes obtained from the captum library.
                This tuple is in the exact structure of the data input.
            nonempty_idx (List of Integers) : Contains the indices representing the presence or
                absence of data points

        """
        attributions = tuple(map(lambda attr: attr.cpu().detach().numpy(), attributions))

        attribution_list = []
        if nonempty_idx[SEQ_CAT_INDEX] != -1:
            attribution_list.append(
                self._aggregate_cat_attributions_util(attributions[nonempty_idx[SEQ_CAT_INDEX]], self.emb_dims_seq))
        if nonempty_idx[SEQ_CONT_INDEX] != -1:
            attribution_list.append(attributions[nonempty_idx[SEQ_CONT_INDEX]])
        if nonempty_idx[NON_SEQ_CAT_INDEX] != -1:
            attribution_list.append(self._aggregate_cat_attributions_util(attributions[nonempty_idx[NON_SEQ_CAT_INDEX]],
                                                                          self.emb_dims_non_seq))

        if nonempty_idx[NON_SEQ_CONT_INDEX] != -1:
            attribution_list.append(attributions[nonempty_idx[NON_SEQ_CONT_INDEX]])

        attribution_agg = tuple(attribution_list)
        if self.explainer_verbosity:
            self.logger.info("Completed aggregation of categorical embedding attributions.")
        return attribution_agg

    def _join_attributions(self, attributions, nonempty_idx, add_across_time=False):
        """Join all attributions.

        This function is used to concatenate/join all attributions obtained in a format making
        it suitable for conversion to a DataFrame.
        Checks if the user wants across time aggregation and reshapes the sequential attributes
        to match the non_sequential attributes.
        The final output contains as many columns as the number of features in the input.

        Args:
            attributions (tuple of numpy arrays) : Contains the categorically aggregated attributions
            nonempty_idx (List of Integers) : Contains the indices representing the presence or
                absence of data points
            add_across_time (Boolean: Default = False) : Determines if the user wants attribute
                addition across time steps for sequential features
        """
        attribution_agg = []

        if nonempty_idx[SEQ_CAT_INDEX] != -1:
            attribution_seq_cat = attributions[nonempty_idx[SEQ_CAT_INDEX]]
            if add_across_time:
                attribution_agg.append(np.sum(attribution_seq_cat, axis=1))
            else:
                seq_cat_shape = attribution_seq_cat.shape
                attribution_agg.append(
                    attribution_seq_cat.reshape(seq_cat_shape[0], seq_cat_shape[1] * seq_cat_shape[2]))

        if nonempty_idx[SEQ_CONT_INDEX] != -1:
            attribution_seq_cont = attributions[nonempty_idx[SEQ_CONT_INDEX]]
            if add_across_time:
                attribution_agg.append(np.sum(attribution_seq_cont, axis=1))
            else:
                seq_cont_shape = attribution_seq_cont.shape
                attribution_agg.append(
                    attribution_seq_cont.reshape(seq_cont_shape[0], seq_cont_shape[1] * seq_cont_shape[2]))

        if nonempty_idx[NON_SEQ_CAT_INDEX] != -1:
            attribution_agg.append(attributions[nonempty_idx[NON_SEQ_CAT_INDEX]])

        if nonempty_idx[NON_SEQ_CONT_INDEX] != -1:
            attribution_agg.append(attributions[nonempty_idx[NON_SEQ_CONT_INDEX]])

        attribution_agg = np.concatenate(attribution_agg, axis=1)
        if self.explainer_verbosity:
            self.logger.info("Joining of attributions, converting the tuple to a numpy array completed")
        return attribution_agg

    def _indices_to_embedding(self, data: tuple):
        seq_cat, seq_cont, non_seq_cat, non_seq_cont = data
        # These functions call the Categorical Embedding layer and obtain the embeddings for every categorical variable
        # They then replace the categorical vars with their corresponding embeddiings and send them in the model
        if self.interpretable_emb_seq is not None:
            seq_cat = self.interpretable_emb_seq.indices_to_embeddings(data[SEQ_CAT_INDEX])
        if self.interpretable_emb_non_seq is not None:
            non_seq_cat = self.interpretable_emb_non_seq.indices_to_embeddings(data[NON_SEQ_CAT_INDEX])
        data, nonempty_idx = get_nonempty_tensors((seq_cat, seq_cont, non_seq_cat, non_seq_cont))

        return data, nonempty_idx

    def _explain(self, dataloader_explain: DataLoader, features: List[str], output_size: int):
        attributions_dict = {}
        for i in range(output_size):
            attributions_dict[i] = pd.DataFrame([])

        for _, _, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data in dataloader_explain:
            data = (seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data)
            nonempty_idx = None
            data, nonempty_idx = self._indices_to_embedding(data)
            for target_dim in range(output_size):
                attributions = self._calculate_SHAP_for_dataframe()(data, target_dim, nonempty_idx)
                attributions = self._join_attributions(
                    attributions=attributions, nonempty_idx=nonempty_idx, add_across_time=None)

                assert len(attributions[0]) == len(features), \
                    "Number of Shap values does not match number of features. " + \
                    str(len(attributions[0])) + " != " + str(len(features))

                # We would need to add the id column here
                attributions = pd.DataFrame(attributions, columns=features)
                attributions_dict[target_dim] = pd.concat([attributions_dict[target_dim], attributions])

        return attributions_dict

    def explain(self, dataloader_explain: DataLoader, features: List[str], output_size: int):
        """Explain the data in the dataloader.

        Main function that is called to by the user to calculate the attributions. This function handles the flow.
        Returns the dataframe with feature names as columns and corresponding attributions as data

        Args:
            dataloader_explain (torch.DataLoader) : Contains the data supplied by the user outputted
                through a torch dataloade.
            features (List of string) : Contains the feature names used for explanation. Put as the
                column names of the dataframe
            output_size (int) : Refers to the output size of the model that is being explained -
                for CASPR also equal to the embedding size

        return: A dictionary of dataframes where the keys of the dict are the embedding dim being explained.
            The dataframes contain the explanation values
        """

        # Change this to pyspark dataframe through generic_to_local method in future
        if not isinstance(dataloader_explain, DataLoader):
            self.logger.info(
                f'Invalid Type: explain function expects a dataloader object. Got: {type(dataloader_explain)}')
            raise Exception

        self.model.set_explain(True)

        categorical_emb_config_start = timeit()
        self._configure_interpretable_embedding_layer()
        categorical_emb_config_end = timeit()

        if self.explainer_verbosity:
            duration_ms = round(1000.0 * (categorical_emb_config_end - categorical_emb_config_start), TIME_PRECISION)
            self.logger.info(f'Categorical Embedding Layer Configuration finished in {duration_ms} ms')

        attribution_calculation_start = timeit()
        attributions_dict = self._explain(dataloader_explain, features, output_size)
        attribution_calculation_end = timeit()
        if self.explainer_verbosity:
            duration_ms = round(1000.0 * (attribution_calculation_end - attribution_calculation_start), TIME_PRECISION)
            self.logger.info(f'Shap Attribution calculation finished in {duration_ms} ms')

        self._remove_interpretable_embedding_layer()
        self.model.set_explain(False)
        if self.explainer_verbosity:
            self.logger.info(f'{MODULE_NAME}.explain finished running')

        return attributions_dict

    def predict(self,
                data: PandasDataFrame,
                id_column_name: str = None,
                with_explanations: bool = True):
        """Predict output of the model."""
        # predict + explain. Tracked in: 299348
        raise NotImplementedError("Not implemented.")

    def summarize(self,
                  explanations: PandasDataFrame,
                  feature_names: List[str],
                  feature_categories: List[str] = None,
                  summary_format: SummaryFormat = SummaryFormat.TABLE,
                  summary_scope: SummaryScope = SummaryScope.ALL,
                  normalize: bool = True):
        """Override the superclass method, with type of data being pandas.DataFrame."""

        return PandasDfUtility.summarize(self.logger, explanations, feature_names, MODULE_NAME,
                                         feature_categories, summary_format, summary_scope, normalize).fillna(0)

    def reduce(self, explanations: PandasDataFrame, feature_categories: Dict[str, List[str]], id_column: str = None):
        """Override the superclass method, with type of data being pandas.DataFrame."""
        return PandasDfUtility.reduce(self.logger, explanations, feature_categories, MODULE_NAME, id_column)

    def _calculate_SHAP_for_dataframe(self):
        def _internal_calculate_SHAP_for_dataframe(data, target_dim: int, nonempty_idx: List[int] = None):
            """Calculate attributions for data.

            Function used for actual attribute calculation by calling the captum library function.
            Does basic error handling.

            Args:
                data (tuple of Tensors) : Contains the data prepared for sending into the model for explanations
                target_dim (Integer) : Represents the dimension of the output being explained
                nonempty_idx (List of Integers) :  Represents the indices determing the presence or absence of
                    data as required by the models
            """
            try:
                attributions = self.explainer.attribute(inputs=data, target=target_dim,  # noqa: E1101
                                                        additional_forward_args=nonempty_idx)
            except ValueError:
                self.logger.info(
                    f'Invalid Data Type in {MODULE_NAME}._calculate_shap_for_dataframe. \
                    Expected data type: tensors of float. \
                    Actual data type: {[type(data[0]), type(data[1]), type(data[2]), type(data[3])]}')
                raise Exception

            # If a model returns N outputs, the shap_values array will have N,
            # two dimensional arrays of shapley values. Currently we only support single objective.
            # We then need to take the first element, which is the Shap Values for the first score.
            if len(attributions) != len(data):
                if self.explainer_verbosity:
                    self.logger.info("CASPRExplainer currently supports explanation for one output at a time.")
                attributions = attributions[0]

            for i, _ in enumerate(attributions):
                assert attributions[i].shape == data[i].shape, \
                    "Number of Shap values does not match number of rows. " + \
                    str(attributions[i].shape[i]) + " != " + str(data[i].shape)

            # Doing the compulsory categorical variable summing
            if isinstance(attributions, tuple):
                attributions = self._aggregate_cat_attributions(attributions, nonempty_idx)
            else:
                attributions = attributions.cpu().detach().numpy()
            return attributions

        return _internal_calculate_SHAP_for_dataframe


Explainer.register(CASPRExplainer)
